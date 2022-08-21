import numpy as np
import torch


class DummySummaryWriter:
    def nop(*args, **kwargs):
        pass

    def __getattr__(self, _):
        return self.nop


class PCL:
    def __init__(
        self,
        model,
        pi_opt,
        v_opt,
        buffer,
        num_updates_per_step=1,
        rollout_horizon=20,
        gamma=1.0,
        entropy_temp=0.5,
        target_entropy=None,
        summary_writer=DummySummaryWriter(),
    ):
        self.model = model
        self.pi_opt = pi_opt
        self.v_opt = v_opt
        self.buffer = buffer
        self.rollout_horizon = rollout_horizon
        self.gamma = gamma
        self.entropy_temp = entropy_temp
        self.target_entropy = target_entropy
        self.num_updates_per_step = num_updates_per_step
        self.summary_writer = summary_writer
        self.step_i = 0

    def compute_loss(self, obss, acts, rews, discounting, curr_horizon):
        v_curr = self.model.compute_values(obss[:, 0])
        v_last = self.model.compute_values(obss[:, -1])
        lprobs = self.model.compute_lprobs(obss, acts)
        disc_ent_reg_rews = (rews - self.entropy_temp * lprobs) * discounting
        loss = (
            -v_curr
            + self.gamma**curr_horizon * v_last
            + torch.sum(disc_ent_reg_rews, axis=-1)
        )
        return torch.mean(loss**2)

    def update(self):
        for update_i in range(self.num_updates_per_step):
            episode = self.buffer.sample()

            curr_horizon = self.rollout_horizon
            if curr_horizon > len(episode["acts"]):
                curr_horizon = len(episode["acts"])

            discounting = self.gamma ** torch.arange(curr_horizon)
            obss = []
            acts = []
            rews = []
            for segment_i in range(len(episode["acts"]) - curr_horizon + 1):
                obss.append(episode["obss"][segment_i : segment_i + curr_horizon])
                acts.append(episode["acts"][segment_i : segment_i + curr_horizon])
                rews.append(episode["rews"][segment_i : segment_i + curr_horizon])

            obss = torch.tensor(obss)
            acts = torch.tensor(acts)
            rews = torch.tensor(rews)

            self.v_opt.zero_grad()
            self.pi_opt.zero_grad()
            loss = self.compute_loss(obss, acts, rews, discounting, curr_horizon)
            loss.backward()
            self.pi_opt.step()
            self.v_opt.step()

            self.summary_writer.add_scalar(
                "loss", loss.cpu().detach().numpy(), self.step_i
            )

    def train(self, env, num_steps):
        obss = []
        acts = []
        rews = []
        truncs = []
        done = False
        ep_i = 0
        obs = env.reset()
        for self.step_i in range(num_steps):
            act = self.model.compute_acts(torch.tensor(obs))
            act = act.cpu().detach().numpy()
            next_obs, rew, done, truncated, info = env.step(act)
            obss.append(obs)
            acts.append(act)
            rews.append(rew)
            truncs.append(truncated)

            obs = next_obs

            if done:
                ep_i += 1
                obss.append(obs)
                self.buffer.add({
                    "obss": np.array(obss),
                    "acts": np.array(acts),
                    "rews": np.array(rews),
                    "truncs": np.array(truncs)
                })

                self.summary_writer.add_scalar("ep_return", np.sum(rews), ep_i)
                self.summary_writer.add_scalar("ep_length", len(rews), ep_i)

                obss = []
                acts = []
                rews = []
                truncs = []
                done = False
                obs = env.reset()

            if self.buffer.trainable:
                self.update()
