import argparse
import gym
import numpy as np
import os
import torch

from torch.utils.tensorboard import SummaryWriter

from pcl_torch.model import Model
from pcl_torch.buffer import ReplayBuffer
from pcl_torch.pcl import PCL


def set_seed(seed=None):
    if seed is None:
        seed = np.random.randint(0, 2**32 - 1)

    np.random.seed(seed)
    torch.manual_seed(seed)
    return seed


def run_pcl(args):
    print("Set seed to {}".format(set_seed(args.seed)))
    os.makedirs(args.log_dir, exist_ok=True)
    summary_writer = SummaryWriter(log_dir=os.path.join(args.log_dir, "tensorboard"))

    env = gym.make(args.env, new_step_api=True)
    env = gym.wrappers.RecordVideo(
        env, os.path.join(args.log_dir, "videos"), new_step_api=True
    )

    model = Model(env.observation_space.shape[0], env.action_space.n)
    pi_opt = torch.optim.Adam(model.pi.parameters(), lr=args.pi_lr)
    v_opt = torch.optim.Adam(model.v.parameters(), lr=args.v_lr)
    buffer = ReplayBuffer()

    target_entropy = None
    if args.automatic_temp:
        target_entropy = env.action_space.n

    agent = PCL(
        model,
        pi_opt,
        v_opt,
        buffer,
        args.num_updates_per_step,
        args.rollout_horizon,
        args.gamma,
        args.entropy_temp,
        target_entropy,
        summary_writer,
    )

    agent.train(env, args.num_steps)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Train PCL agent in environments with discrete action space."
    )
    parser.add_argument(
        "--log_dir", type=str, default=None, help="The directory for storing results"
    )
    parser.add_argument("--seed", type=int, default=0, help="The RNG seed")
    parser.add_argument(
        "--env",
        type=str,
        required=True,
        help="The environment to train the PCL agent on",
    )

    parser.add_argument(
        "--pi_lr", type=float, default=3e-4, help="The learning rate for policy updates"
    )
    parser.add_argument(
        "--v_lr", type=float, default=3e-4, help="The learning rate for value updates"
    )

    parser.add_argument(
        "--num_updates_per_step",
        type=int,
        default=1,
        help="Number of updates to perform between each interaction step",
    )
    parser.add_argument(
        "--rollout_horizon",
        type=int,
        default=20,
        help="The rollout horizon for PCL updates",
    )
    parser.add_argument("--gamma", type=float, default=1, help="The discount factor")
    parser.add_argument(
        "--entropy_temp",
        type=float,
        default=0.5,
        help="Temperature term for entropy regularized objective",
    )
    parser.add_argument(
        "--automatic_temp",
        action="store_true",
        help="Whether or not to adjust temperature term",
    )

    parser.add_argument(
        "--num_steps",
        type=int,
        default=int(1e6),
        help="Number of environment steps to take",
    )
    args = parser.parse_args()

    run_pcl(args)
