import argparse
from distutils.util import strtobool

import torch


def get_args():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument('--algo', default='ppo',
                        help='algorithm to use: a2c | ppo | acktr')
    parser.add_argument('--lr', type=float, default=0.00005,
                        help='learning rate')
    parser.add_argument('--eps', type=float, default=1e-5,
                        help='RMSprop optimizer epsilon (default: 1e-5)')
    parser.add_argument('--alpha', type=float, default=0.99,
                        help='RMSprop optimizer apha (default: 0.99)')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='discount factor for rewards (default: 0.99)')
    parser.add_argument('--use_gae', action='store_true', default=False,
                        help='use generalized advantage estimation')
    parser.add_argument('--tau', type=float, default=0.95,
                        help='gae parameter (default: 0.95)')
    parser.add_argument('--entropy_coef', type=float, default=0.001,
                        help='entropy term coefficient')
    parser.add_argument('--value_loss_coef', type=float, default=0.5,
                        help='value loss coefficient (default: 0.5)')
    parser.add_argument('--max_grad_norm', type=float, default=0.5,
                        help='max norm of gradients (default: 0.5)')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed (default: 1)')
    parser.add_argument('--num_processes', type=int, default=16,
                        help='how many training CPU processes to use')
    parser.add_argument('--num_steps', type=int, default=5,
                        help='number of forward steps in A2C (default: 5)')
    parser.add_argument('--ppo_epoch', type=int, default=4,
                        help='number of ppo epochs (default: 4)')
    parser.add_argument('--num_mini-batch', type=int, default=16,
                        help='number of batches for ppo')
    parser.add_argument('--clip_param', type=float, default=0.2,
                        help='ppo clip parameter (default: 0.2)')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='log interval, one log per n updates (default: 10)')
    parser.add_argument('--save_interval', type=int, default=100,
                        help='save interval, one save per n updates (default: 100)')
    parser.add_argument('--eval_interval', type=int, default=None,
                        help='eval interval, one eval per n updates (default: None)')
    parser.add_argument('--vis_interval', type=int, default=100,
                        help='vis interval, one log per n updates (default: 100)')
    parser.add_argument('--num_frames', type=int, default=10e6,
                        help='number of frames to train (default: 10e6)')
    parser.add_argument('--env_name', default='PongNoFrameskip-v4',
                        help='environment to train on (default: PongNoFrameskip-v4)')
    parser.add_argument('--log_dir', default='/tmp/gym/',
                        help='directory to save agent logs (default: /tmp/gym)')
    parser.add_argument('--save_dir', default='./trained_models/',
                        help='directory to save agent logs (default: ./trained_models/)')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--add_timestep', action='store_true', default=False,
                        help='add timestep to observations')
    parser.add_argument('--recurrent_policy', default=False,
                        type = lambda x: bool(strtobool(x)),
                        help='use a recurrent policy')
    parser.add_argument('--vis', action='store_true', default=False,
                        help='enable visdom visualization')
    parser.add_argument('--port', type=int, default=8097,
                        help='port to run the server on (default: 8097)')
    parser.add_argument('--eps_explore', type = float, default = 0.15)
    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    return args
