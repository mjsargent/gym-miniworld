import copy
import glob
import os
import time
import types
import random
from collections import deque

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#torch.autograd.set_detect_anomaly(True)

import algo
from arguments import get_args
from envs import make_vec_envs
from model import Policy, SFPolicy, QPolicy, SFConditionedPolicy, XPolicy, SFConditionedXPolicy
from storage import RolloutStorage, TemporallyExtendedRolloutStorage
#from visualize import visdom_plot

import wandb

from multiprocessing import set_start_method

args = get_args()

assert args.algo in ['a2c', 'ppo', 'acktr', 'sf', "q", "a2csf", "a2cx", "a2csfx"]
if args.recurrent_policy:
    assert args.algo in ['a2c', 'ppo', "sf", "q", "a2csf", "a2cx", "a2csfx"], \
        'Recurrent policy is not implemented for ACKTR'

num_updates = int(args.num_frames) // args.num_steps // args.num_processes

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

try:
    os.makedirs(args.log_dir)
except OSError:
    files = glob.glob(os.path.join(args.log_dir, '*.monitor.csv'))
    for f in files:
        os.remove(f)

eval_log_dir = args.log_dir + "_eval"

try:
    os.makedirs(eval_log_dir)
except OSError:
    files = glob.glob(os.path.join(eval_log_dir, '*.monitor.csv'))
    for f in files:
        os.remove(f)

def gather_steps(tensor, steps):
    """
    function for taking the ith step for a given tensor assuming steps
    are one-off from the required index
    """
    return tensor[steps - 1, torch.arange(tensor.shape[1]), :]

def make_w(task_rewards, num_processes, device):
    # feature is  box ball key
    w = [task_rewards["box"], task_rewards["ball"], task_rewards["key"]]
    w = torch.tensor(w).unsqueeze(0).unsqueeze(0).to(device)
    return w.repeat(1,num_processes,1)

def make_gpi_set(gpi_set: str, num_processes, device, training_set = None, test_task = None):
    if gpi_set == "training_set":
        z = torch.cat([make_w(task, num_processes, device) for task in training_set] , dim = 1)
    elif gpi_set == "test_task":
        z = make_w(test_task, num_processes, device)
    elif gpi_set == "training_set_and_test_task":
        z_train = torch.cat([make_w(task, num_processes, device) for task in training_set] , dim = 1)
        z_test = make_w(test_task, num_processes, device)
        z = torch.cat([z_train, z_test], dim = 1).float()
    return z

def multitask():
    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")

    """
    if args.vis:
        from visdom import Visdom
        viz = Visdom(port=args.port)
        win = None
    """

    if args.feature_size > 1:
        feature_size = args.feature_size
        learn_phi = True
    else:
        feature_size = 3
        learn_phi = False
    envs = make_vec_envs(args.env_name, args.seed, args.num_processes,
                        args.gamma, args.log_dir, args.add_timestep, device, False)

    if args.algo == "a2csf":
        actor_critic = SFConditionedPolicy(envs.observation_space.shape, envs.action_space, feature_size = feature_size, learn_phi = learn_phi,
            multitask = True, base_kwargs={'recurrent': args.recurrent_policy})
        actor_critic.to(device)
    elif args.algo == "a2cx":
        actor_critic = XPolicy(envs.observation_space.shape, envs.action_space, feature_size = feature_size,
            max_repeat = args.max_repeat, multitask = True, base_kwargs={'recurrent': args.recurrent_policy})
        actor_critic.to(device)
    elif args.algo == "a2csfx":
        actor_critic = SFConditionedXPolicy(envs.observation_space.shape, envs.action_space, feature_size = feature_size,
            max_repeat = args.max_repeat, multitask = True, base_kwargs={'recurrent': args.recurrent_policy})
        actor_critic.to(device)
    else:
        actor_critic = Policy(envs.observation_space.shape, envs.action_space, feature_size = feature_size,
            base_kwargs={'recurrent': args.recurrent_policy})
        actor_critic.to(device)


    if args.algo == 'a2c':
        agent = algo.A2C_ACKTR(actor_critic, args.value_loss_coef,
                               args.entropy_coef, lr=args.lr,
                               eps=args.eps, alpha=args.alpha,
                               max_grad_norm=args.max_grad_norm,
                               feature_size = feature_size)

    elif args.algo == 'a2csf':
        agent = algo.A2C_SF(actor_critic, args.value_loss_coef,
                               args.entropy_coef, lr_psi=args.lr,
                               lr_policy = args.lr, lr_w = 1,
                               eps=args.eps, alpha=args.alpha,
                               max_grad_norm=args.max_grad_norm,
                               feature_size = feature_size, gamma=args.gamma,
                                learn_phi = learn_phi, multitask = True)
    elif args.algo == "a2cx":
        agent = algo.A2CX(actor_critic, args.value_loss_coef,
                               args.entropy_coef, lr = args.lr,
                               eps=args.eps, alpha=args.alpha,
                               max_grad_norm=args.max_grad_norm,
                               feature_size = feature_size, gamma=args.gamma,
                                )

    elif args.algo == 'ppo':
        agent = algo.PPO(actor_critic, args.clip_param, args.ppo_epoch, args.num_mini_batch,
                         args.value_loss_coef, args.entropy_coef, lr=args.lr,
                               eps=args.eps,
                               max_grad_norm=args.max_grad_norm)
    elif args.algo == 'acktr':
        agent = algo.A2C_ACKTR(actor_critic, args.value_loss_coef,
                               args.entropy_coef, acktr=True)
    elif args.algo == 'a2csfx':
        agent = algo.A2CSFX(actor_critic, args.value_loss_coef,
                               args.entropy_coef, lr = args.lr,
                               eps=args.eps, alpha=args.alpha,
                               lr_w = 1,
                               max_grad_norm=args.max_grad_norm,
                               feature_size = feature_size, gamma=args.gamma)

    use_a2csf_storage = True if args.algo == "a2csf"  or args.algo == "a2csfx" else False


    # temporally extended methods
    if args.algo == "a2cx" or args.algo == "a2csfx":
        # num steps is overwitten to mean numboer macro decisions - results
        # are still report wrt the number of steps taken in the envs
        rollouts = TemporallyExtendedRolloutStorage(args.num_steps, args.num_processes,
                        envs.observation_space.shape, envs.action_space,
                        actor_critic.recurrent_hidden_state_size, feature_dim = feature_size,
                                  a2csf = use_a2csf_storage, z_samples = args.z_samples)
    else:
        rollouts = RolloutStorage(args.num_steps, args.num_processes,
                        envs.observation_space.shape, envs.action_space,
                        actor_critic.recurrent_hidden_state_size, feature_dim = feature_size,
                                  a2csf = use_a2csf_storage, z_samples = args.z_samples)
    num_processes = args.num_processes

    obs = envs.reset()

    # create a dummy feature
    dummy_feature = torch.zeros([args.num_processes, feature_size])

    rollouts.features[0].copy_(dummy_feature)
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    episode_rewards = deque(maxlen=100)

    start = time.time()
    wandb.init(project = "tSFMultitask", config = args)

    task_switch_counter = 0

    training_set = [{"ball": 1, "box": 0, "key": 0},
                    {"ball": 0, "box": 1, "key": 0},
                    {"ball": 0, "box": 0, "key": 1}
                    ]
    test_set = [["110",{"ball": 1, "box": 1, "key": 0}],
                ["011",{"ball": 0, "box": 1, "key": 1}],
                ["101",{"ball": 1, "box": 0, "key": 1}],
                ["1-10",{"ball": 1, "box": -1, "key": 0}],
                ["10-1",{"ball": 1, "box": 0, "key": -1}],
                ["1-1-1",{"ball": 1, "box": -1, "key": -1}],
                ["-110",{"ball": -1, "box": 1, "key": 0}],
                ["01-1",{"ball": 0, "box": 1, "key": -1}],
                ["-11-1",{"ball": -1, "box": 1, "key": -1}],
                ["-101",{"ball": -1, "box": 0, "key": 1}],
                ["0-11",{"ball": 0, "box": -1, "key": 1}],
                ["-1-11",{"ball": -1, "box": -1, "key": 1}],
                ]

    episode_rewards_acc = torch.zeros(num_processes, requires_grad = False)

    if args.algo == "a2csf":
        task_rewards = random.choice(training_set)
        w = make_w(task_rewards, num_processes, device)
        rollouts.w_storage[0].copy_(w[0])
        obs = envs.switch(task_rewards)
        for j in range(num_updates):
            for step in range(args.num_steps):
                # sample a training task:

                # Sample actions
                with torch.no_grad():
                    value, action, action_log_prob, recurrent_hidden_states, psi, feature, z = actor_critic.act(
                            rollouts.obs[step],
                            rollouts.recurrent_hidden_states[step],
                            rollouts.masks[step],
                            rollouts.features[step],
                            w = rollouts.w_storage[step],
                            z_n = args.z_samples)
                # Obser reward and next obs
                obs, reward, done, infos = envs.step(action)

                # info is a tuple of dicts
                if not actor_critic.learn_phi:
                    _feature = []
                    for info in infos:
                        if "feature" in info.keys():
                            _feature.append(info["feature"])

                    feature = torch.FloatTensor(np.stack(_feature, axis = 0)).to(device)

                estimated_reward = actor_critic.evaluate_rewards(feature)
                """
                for info in infos:
                    if 'episode' in info.keys():
                        print(reward)
                        episode_rewards.append(info['episode']['r'])
                """
                episode_rewards_acc += reward.squeeze(1)
                for idx, eps_done in enumerate(done):
                    if eps_done:
                        episode_rewards.append(np.array(episode_rewards_acc[idx]))
                        episode_rewards_acc[idx] = 0
                        new_task = random.choice(training_set)
                        new_w = make_w(new_task, 1, device)
                        if step < args.num_steps - 1:
                            # a bit hacky - if an episode ends on the last step
                            # of the rollout will we end up using the same w
                            # again
                            rollouts.w_storage[step + 1][idx].copy_(new_w[0][0])


                # If done then clean the history of observations.
                masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
                rollouts.insert(obs, recurrent_hidden_states, action, action_log_prob, value, reward, masks, feature, psi, estimated_reward, w = w[0], z = z[0])

            with torch.no_grad():
                next_value, next_psi, next_phi= actor_critic.get_value(rollouts.obs[-1],
                                                    rollouts.recurrent_hidden_states[-1],
                                                    rollouts.masks[-1],
                                                    rollouts.features[-1],
                                                    w = rollouts.w_storage[-1])


            #rollouts.compute_returns(next_value, args.use_gae, args.gamma, args.tau, sf = True)
            rollouts.compute_psi_returns(next_psi,args.gamma, next_phi)

            value_loss, action_loss, dist_entropy, psi_loss, w_loss, phi_loss = agent.update(rollouts)

            rollouts.after_update()

            if j % args.save_interval == 0 and args.save_dir != "":
                print('Saving model')
                print()

                save_path = os.path.join(args.save_dir, args.algo)
                try:
                    os.makedirs(save_path)
                except OSError:
                    pass

                # A really ugly way to save a model to CPU
                save_model = actor_critic
                if args.cuda:
                    save_model = copy.deepcopy(actor_critic).cpu()

                save_model = [save_model, hasattr(envs.venv, 'ob_rms') and envs.venv.ob_rms or None]

                torch.save(save_model, os.path.join(save_path, args.env_name + ".pt"))

            total_num_steps = (j + 1) * args.num_processes * args.num_steps

            task_switch_counter = (j + 1) * args.num_processes * args.num_steps
            if j % args.log_interval == 0 and len(episode_rewards) > 1:
                end = time.time()
                print("Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.2f}/{:.2f}, min/max reward {:.2f}/{:.2f}, success rate {:.2f}\n".
                    format(
                        j, total_num_steps,
                        int(total_num_steps / (end - start)),
                        len(episode_rewards),
                        np.mean(episode_rewards),
                        np.median(episode_rewards),
                        np.min(episode_rewards),
                        np.max(episode_rewards),
                        np.count_nonzero(np.greater(episode_rewards, 0)) / len(episode_rewards)
                    )
                )
                wandb.log({"mean_reward": np.mean(episode_rewards),
                           "success_rate": np.count_nonzero(np.greater(episode_rewards, 0)) / len(episode_rewards),
                           "num_updates": j,
                           "value_loss": float(value_loss),
                           "action_loss": float(action_loss),
                           "dist_entropy": float(dist_entropy),
                           "psi_loss": float(psi_loss),
                           "w_loss": float(w_loss),
                           "phi_loss": float(phi_loss),
                           }, step = total_num_steps)


            if args.eval_interval is not None and len(episode_rewards) > 1 and j % args.eval_interval == 0:
                eval_envs = make_vec_envs(args.env_name, args.seed + args.num_processes, args.num_processes,
                                    args.gamma, eval_log_dir, args.add_timestep, device, True)


                for test_task in test_set:
                    test_reward_dict = test_task[1]
                    test_w = make_w(test_reward_dict, num_processes, device).float()
                    gpi_z = make_gpi_set(args.gpi_eval, num_processes, device, training_set, test_reward_dict)
                    gpi_z = gpi_z.float()
                    obs = eval_envs.switch(test_reward_dict)

                    if eval_envs.venv.__class__.__name__ == "VecNormalize":
                        eval_envs.venv.ob_rms = envs.venv.ob_rms

                        # An ugly hack to remove updates
                        def _obfilt(self, obs):
                            if self.ob_rms:
                                obs = np.clip((obs - self.ob_rms.mean) / np.sqrt(self.ob_rms.var + self.epsilon), -self.clipob, self.clipob)
                                return obs
                            else:
                                return obs

                        eval_envs.venv._obfilt = types.MethodType(_obfilt, envs.venv)

                    eval_episode_rewards = []
                    eval_episode_rewards_acc = torch.zeros(num_processes)

                    obs = eval_envs.reset()
                    eval_recurrent_hidden_states = torch.zeros(args.num_processes,
                                    actor_critic.recurrent_hidden_state_size, device=device)
                    eval_masks = torch.zeros(args.num_processes, 1, device=device)

                    # create a dummy feature
                    eval_features = torch.zeros([args.num_processes, feature_size])


                    while len(eval_episode_rewards) < 50:
                        with torch.no_grad():
                            _, action, _, eval_recurrent_hidden_states, _, _, _ = actor_critic.act(
                                obs, eval_recurrent_hidden_states, eval_masks, eval_features, deterministic=True,
                                 w = test_w[0], z = gpi_z[0]
                            )
                        # Obser reward and next obs
                        obs, reward, done, infos = eval_envs.step(action)
                        eval_episode_rewards+= reward.squeeze(1)
                        _feature = []
                        for info in infos:
                            if "feature" in info.keys():
                                _feature.append(info["feature"])
                        eval_features = torch.FloatTensor(np.stack(_feature, axis = 0)).to(device)

                        for idx, eps_done in enumerate(done):
                            if eps_done:
                                eval_episode_rewards.append(np.array(eval_episode_rewards_acc[idx]))
                                eval_episode_rewards_acc[idx] = 0
                        eval_masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done]).to(device)


                    print(" Evaluation of task {} using {} episodes: mean reward {:.5f}\n".format(
                        test_task[0],
                        len(eval_episode_rewards),
                        np.mean(eval_episode_rewards)
                                    ))
                    wandb.log({f"mean_eval_reward, {test_task[0]}": np.mean(eval_episode_rewards),
                            }, step = total_num_steps)

                eval_envs.close()

            """
            if args.vis and j % args.vis_interval == 0:
                try:
                    # Sometimes monitor doesn't properly flush the outputs
                    win = visdom_plot(viz, win, args.log_dir, args.env_name,
                                    args.algo, args.num_frames)
                except IOError:
                    pass
            """

        envs.close()

    elif args.algo == "a2cx":
        total_num_steps = 0
        #wandb.watch(actor_critic.base)
        for j in range(num_updates):
            action_only = True if j < (num_updates * args.action_only_interval) else False
            for step in range(args.num_steps):

                with torch.no_grad():
                    value, action, action_log_prob, recurrent_hidden_states, repeat, repeat_log_prob = actor_critic.act(
                            rollouts.obs[step],
                            rollouts.recurrent_hidden_states[step],
                            rollouts.masks[step],
                            rollouts.features[step])
                # Obser reward and next obs
                #action = torch.randint_like(action, low = 0, high = 2)
                #repeat_env = repeat.clone()

                # if we are in the non_repeated phase of learning,
                # overwrite repeats with zeros
                if action_only:
                    repeat = torch.zeros_like(repeat)
                obs, reward, done, infos = envs.step(action, repeat)

                # info is a tuple of dicts
                _feature = []
                _steps_taken = []
                for info in infos:
                    if "feature" in info.keys():
                        _feature.append(info["feature"])
                    if "steps_taken" in info.keys():
                        _steps_taken.append(info["steps_taken"])

                feature = torch.tensor(np.stack(_feature, axis = 0)).to(device)
                steps_taken = torch.tensor(np.stack(_steps_taken, axis = 0)).to(device).unsqueeze(1)
                """
                for info in infos:
                    if 'episode' in info.keys():
                        print(reward)
                        episode_rewards.append(info['episode']['r'])
                """

                # FIXME: works only for environments with sparse rewards
                for idx, eps_done in enumerate(done):
                    if eps_done:
                        episode_rewards.append(np.array(reward[idx]))
                # If done then clean the history of observations.
                masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
                rollouts.insert(obs, recurrent_hidden_states,
                    action, action_log_prob, value, reward, masks, feature,
                    psi = None, estimated_reward = None,
                    repeat = repeat, repeat_log_prob = repeat_log_prob,
                    steps_taken = steps_taken)

            with torch.no_grad():
                next_value = actor_critic.get_value(rollouts.obs[-1],
                                                    rollouts.recurrent_hidden_states[-1],
                                                    rollouts.masks[-1],
                                                    rollouts.features[-1]).detach()

            rollouts.compute_returns(next_value, args.use_gae, args.gamma, args.tau)

            value_loss, action_loss, dist_entropy, \
                repeat_loss, repeat_dist_entropy = agent.update(rollouts, action_only)

            total_num_steps = total_num_steps + rollouts.steps_taken.sum()

            total_num_des = (j + 1) * args.num_processes * args.num_steps
            task_switch_counter = total_num_steps + rollouts.steps_taken.sum()
            rollouts.after_update()

            if j % args.save_interval == 0 and args.save_dir != "":
                print('Saving model')
                print()

                save_path = os.path.join(args.save_dir, args.algo)
                try:
                    os.makedirs(save_path)
                except OSError:
                    pass

                # A really ugly way to save a model to CPU
                save_model = actor_critic
                if args.cuda:
                    save_model = copy.deepcopy(actor_critic).cpu()

                save_model = [save_model, hasattr(envs.venv, 'ob_rms') and envs.venv.ob_rms or None]

                torch.save(save_model, os.path.join(save_path, args.env_name + ".pt"))


            if j % args.log_interval == 0 and len(episode_rewards) > 1:
                end = time.time()
                print("Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.2f}/{:.2f}, min/max reward {:.2f}/{:.2f}, success rate {:.2f}\n".
                    format(
                        j, total_num_steps,
                        int(total_num_steps / (end - start)),
                        len(episode_rewards),
                        np.mean(episode_rewards),
                        np.median(episode_rewards),
                        np.min(episode_rewards),
                        np.max(episode_rewards),
                        np.count_nonzero(np.greater(episode_rewards, 0)) / len(episode_rewards)
                    )
                )
                repeat_histogram = wandb.Histogram(rollouts.repeats.flatten().cpu().numpy(), num_bins = args.max_repeat)
                wandb.log({"mean_reward": np.mean(episode_rewards),
                           "success_rate": np.count_nonzero(np.greater(episode_rewards, 0)) / len(episode_rewards),
                           "num_updates": j,
                           "value_loss": float(value_loss),
                           "action_loss": float(action_loss),
                           "dist_entropy": float(dist_entropy),
                           "repeat_loss": float(repeat_loss),
                           "repeat_dist_entropy": float(repeat_dist_entropy),
                           "repeat_histogram": repeat_histogram,
                           "total_num_des": total_num_des
                           }, step = total_num_steps)

            if args.task_switch_interval > 0 and task_switch_counter > args.task_switch_interval:
                obs = envs.switch()
                task_switch_counter = 0

            if args.eval_interval is not None and len(episode_rewards) > 1 and j % args.eval_interval == 0:
                eval_envs = make_vec_envs(args.env_name, args.seed + args.num_processes, args.num_processes,
                                    args.gamma, eval_log_dir, args.add_timestep, device, True)

                if eval_envs.venv.__class__.__name__ == "VecNormalize":
                    eval_envs.venv.ob_rms = envs.venv.ob_rms

                    # An ugly hack to remove updates
                    def _obfilt(self, obs):
                        if self.ob_rms:
                            obs = np.clip((obs - self.ob_rms.mean) / np.sqrt(self.ob_rms.var + self.epsilon), -self.clipob, self.clipob)
                            return obs
                        else:
                            return obs

                    eval_envs.venv._obfilt = types.MethodType(_obfilt, envs.venv)

                eval_episode_rewards = []

                obs = eval_envs.reset()
                eval_recurrent_hidden_states = torch.zeros(args.num_processes,
                                actor_critic.recurrent_hidden_state_size, device=device)
                eval_masks = torch.zeros(args.num_processes, 1, device=device)

                # create a dummy feature
                eval_features = torch.zeros([args.num_processes, feature_size]).to(device)

                while len(eval_episode_rewards) < 10:
                    with torch.no_grad():
                        _, action, _, eval_recurrent_hidden_states, repeat, _ = actor_critic.act(
                            obs, eval_recurrent_hidden_states, eval_masks, eval_features, deterministic=True)

                    # Obser reward and next obs
                    obs, reward, done, infos = eval_envs.step(action, repeat)

                    _feature = []
                    for info in infos:
                        if "feature" in info.keys():
                            _feature.append(info["feature"])
                    eval_feature = np.stack(_feature, axis = 0)
                    eval_feature = torch.tensor(feature).to(device)

                    eval_masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done]).to(device)

                    for idx, eps_done in enumerate(done):
                        if eps_done:
                            eval_episode_rewards.append(np.array(reward[idx]))

                eval_envs.close()

                print(" Evaluation using {} episodes: mean reward {:.5f}\n".format(
                    len(eval_episode_rewards),
                    np.mean(eval_episode_rewards)
                                   ))
                wandb.log({"mean_eval_reward": np.mean(eval_episode_rewards),
                           }, step = total_num_steps)

            # for temporally extended algs that take multiple time steps
            if total_num_steps > args.num_frames:
                break

            """
            if args.vis and j % args.vis_interval == 0:
                try:
                    # Sometimes monitor doesn't properly flush the outputs
                    win = visdom_plot(viz, win, args.log_dir, args.env_name,
                                    args.algo, args.num_frames)
                except IOError:
                    pass
            """

        envs.close()

    elif args.algo == "a2csfx":

        total_num_steps = 0
        #wandb.watch(actor_critic.base)
        for j in range(num_updates):
            action_only = True if j < (num_updates * args.action_only_interval) else False
            for step in range(args.num_steps):

                with torch.no_grad():
                    value, action, action_log_prob, recurrent_hidden_states, repeat, repeat_log_prob, psi = actor_critic.act(
                            rollouts.obs[step],
                            rollouts.recurrent_hidden_states[step],
                            rollouts.masks[step],
                            rollouts.features[step])
                # Obser reward and next obs
                #action = torch.randint_like(action, low = 0, high = 2)
                #repeat_env = repeat.clone()

                # if we are in the non_repeated phase of learning,
                # overwrite repeats with zeros
                if action_only:
                    repeat = torch.zeros_like(repeat)
                obs, reward, done, infos = envs.step(action, repeat)

                # info is a tuple of dicts
                _feature = []
                _steps_taken = []
                for info in infos:
                    if "feature" in info.keys():
                        _feature.append(info["feature"])
                    if "steps_taken" in info.keys():
                        _steps_taken.append(info["steps_taken"])

                feature = torch.tensor(np.stack(_feature, axis = 0)).to(device)
                steps_taken = torch.tensor(np.stack(_steps_taken, axis = 0)).to(device).unsqueeze(1)
                """
                for info in infos:
                    if 'episode' in info.keys():
                        print(reward)
                        episode_rewards.append(info['episode']['r'])
                """

                # FIXME: works only for environments with sparse rewards
                for idx, eps_done in enumerate(done):
                    if eps_done:
                        episode_rewards.append(np.array(reward[idx]))
                # If done then clean the history of observations.
                masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
                rollouts.insert(obs, recurrent_hidden_states,
                    action, action_log_prob, value, reward, masks, feature,
                    psi = None, estimated_reward = None,
                    repeat = repeat, repeat_log_prob = repeat_log_prob,
                    steps_taken = steps_taken)

            with torch.no_grad():
                next_value, next_psi = actor_critic.get_value(rollouts.obs[-1],
                                                    rollouts.recurrent_hidden_states[-1],
                                                    rollouts.masks[-1],
                                                    rollouts.features[-1])

            rollouts.compute_returns(next_value, args.use_gae, args.gamma, args.tau)
            rollouts.compute_psi_returns(next_psi, args.gamma)

            value_loss, action_loss, dist_entropy, \
                repeat_loss, repeat_dist_entropy, psi_loss, w_loss = agent.update(rollouts, action_only)

            total_num_steps = total_num_steps + rollouts.steps_taken.sum()
            task_switch_counter = total_num_steps + rollouts.steps_taken.sum()
            total_num_des = (j + 1) * args.num_processes * args.num_steps

            rollouts.after_update()

            if j % args.save_interval == 0 and args.save_dir != "":
                print('Saving model')
                print()

                save_path = os.path.join(args.save_dir, args.algo)
                try:
                    os.makedirs(save_path)
                except OSError:
                    pass

                # A really ugly way to save a model to CPU
                save_model = actor_critic
                if args.cuda:
                    save_model = copy.deepcopy(actor_critic).cpu()

                save_model = [save_model, hasattr(envs.venv, 'ob_rms') and envs.venv.ob_rms or None]

                torch.save(save_model, os.path.join(save_path, args.env_name + ".pt"))


            if j % args.log_interval == 0 and len(episode_rewards) > 1:
                end = time.time()
                print("Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.2f}/{:.2f}, min/max reward {:.2f}/{:.2f}, success rate {:.2f}\n".
                    format(
                        j, total_num_steps,
                        int(total_num_steps / (end - start)),
                        len(episode_rewards),
                        np.mean(episode_rewards),
                        np.median(episode_rewards),
                        np.min(episode_rewards),
                        np.max(episode_rewards),
                        np.count_nonzero(np.greater(episode_rewards, 0)) / len(episode_rewards)
                    )
                )
                repeat_histogram = wandb.Histogram(rollouts.repeats.flatten().cpu().numpy(), num_bins = args.max_repeat)
                wandb.log({"mean_reward": np.mean(episode_rewards),
                           "success_rate": np.count_nonzero(np.greater(episode_rewards, 0)) / len(episode_rewards),
                           "num_updates": j,
                           "value_loss": float(value_loss),
                           "action_loss": float(action_loss),
                           "dist_entropy": float(dist_entropy),
                           "repeat_loss": float(repeat_loss),
                           "repeat_dist_entropy": float(repeat_dist_entropy),
                           "repeat_histogram": repeat_histogram,
                           "total_num_des": total_num_des,
                           "psi_loss" : psi_loss,
                           "w_loss": w_loss
                           }, step = total_num_steps)

            if args.task_switch_interval > 0 and task_switch_counter > args.task_switch_interval:
                obs = envs.switch()
                task_switch_counter = 0

            if args.eval_interval is not None and len(episode_rewards) > 1 and j % args.eval_interval == 0:
                eval_envs = make_vec_envs(args.env_name, args.seed + args.num_processes, args.num_processes,
                                    args.gamma, eval_log_dir, args.add_timestep, device, True)

                if eval_envs.venv.__class__.__name__ == "VecNormalize":
                    eval_envs.venv.ob_rms = envs.venv.ob_rms

                    # An ugly hack to remove updates
                    def _obfilt(self, obs):
                        if self.ob_rms:
                            obs = np.clip((obs - self.ob_rms.mean) / np.sqrt(self.ob_rms.var + self.epsilon), -self.clipob, self.clipob)
                            return obs
                        else:
                            return obs

                    eval_envs.venv._obfilt = types.MethodType(_obfilt, envs.venv)

                eval_episode_rewards = []

                obs = eval_envs.reset()
                eval_recurrent_hidden_states = torch.zeros(args.num_processes,
                                actor_critic.recurrent_hidden_state_size, device=device)
                eval_masks = torch.zeros(args.num_processes, 1, device=device)

                # create a dummy feature
                eval_features = torch.zeros([args.num_processes, feature_size]).to(device)

                while len(eval_episode_rewards) < 10:
                    with torch.no_grad():
                        _, action, _, eval_recurrent_hidden_states, repeat, _, _ = actor_critic.act(
                            obs, eval_recurrent_hidden_states, eval_masks, eval_features, deterministic=True)

                    # Obser reward and next obs
                    obs, reward, done, infos = eval_envs.step(action, repeat)

                    _feature = []
                    for info in infos:
                        if "feature" in info.keys():
                            _feature.append(info["feature"])
                    eval_feature = np.stack(_feature, axis = 0)
                    eval_feature = torch.tensor(feature).to(device)

                    eval_masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done]).to(device)

                    for idx, eps_done in enumerate(done):
                        if eps_done:
                            eval_episode_rewards.append(np.array(reward[idx]))

                eval_envs.close()

                print(" Evaluation using {} episodes: mean reward {:.5f}\n".format(
                    len(eval_episode_rewards),
                    np.mean(eval_episode_rewards)
                                   ))
                wandb.log({"mean_eval_reward": np.mean(eval_episode_rewards),
                           }, step = total_num_steps)

            # for temporally extended algs that take multiple time steps
            if total_num_steps > args.num_frames:
                break


            """
            if args.vis and j % args.vis_interval == 0:
                try:
                    # Sometimes monitor doesn't properly flush the outputs
                    win = visdom_plot(viz, win, args.log_dir, args.env_name,
                                    args.algo, args.num_frames)
                except IOError:
                    pass
            """

        envs.close()

    else:

        for j in range(num_updates):
            for step in range(args.num_steps):
                # Sample actions
                with torch.no_grad():

                    value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                            rollouts.obs[step],
                            rollouts.recurrent_hidden_states[step],
                            rollouts.masks[step],
                            rollouts.features[step])

                # Obser reward and next obs
                obs, reward, done, infos = envs.step(action)

                # info is a tuple of dicts
                _feature = []
                for info in infos:
                    if "feature" in info.keys():
                        _feature.append(info["feature"])

                feature = torch.tensor(np.stack(_feature, axis = 0)).to(device)

                """
                for info in infos:
                    if 'episode' in info.keys():
                        print(reward)
                        episode_rewards.append(info['episode']['r'])
                """

                # FIXME: works only for environments with sparse rewards
                for idx, eps_done in enumerate(done):
                    if eps_done:
                        episode_rewards.append(np.array(reward[idx]))

                # If done then clean the history of observations.
                masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
                rollouts.insert(obs, recurrent_hidden_states, action, action_log_prob, value, reward, masks, feature, psi = None, estimated_reward = None)

            with torch.no_grad():
                next_value = actor_critic.get_value(rollouts.obs[-1],
                                                    rollouts.recurrent_hidden_states[-1],
                                                    rollouts.masks[-1],
                                                    rollouts.features[-1]).detach()

            rollouts.compute_returns(next_value, args.use_gae, args.gamma, args.tau)

            value_loss, action_loss, dist_entropy = agent.update(rollouts)

            rollouts.after_update()

            if j % args.save_interval == 0 and args.save_dir != "":
                print('Saving model')
                print()

                save_path = os.path.join(args.save_dir, args.algo)
                try:
                    os.makedirs(save_path)
                except OSError:
                    pass

                # A really ugly way to save a model to CPU
                save_model = actor_critic
                if args.cuda:
                    save_model = copy.deepcopy(actor_critic).cpu()

                save_model = [save_model, hasattr(envs.venv, 'ob_rms') and envs.venv.ob_rms or None]

                torch.save(save_model, os.path.join(save_path, args.env_name + ".pt"))

            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            task_switch_counter = (j + 1) * args.num_processes * args.num_steps

            if j % args.log_interval == 0 and len(episode_rewards) > 1:
                end = time.time()
                print("Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.2f}/{:.2f}, min/max reward {:.2f}/{:.2f}, success rate {:.2f}\n".
                    format(
                        j, total_num_steps,
                        int(total_num_steps / (end - start)),
                        len(episode_rewards),
                        np.mean(episode_rewards),
                        np.median(episode_rewards),
                        np.min(episode_rewards),
                        np.max(episode_rewards),
                        np.count_nonzero(np.greater(episode_rewards, 0)) / len(episode_rewards)
                    )
                )
                wandb.log({"mean_reward": np.mean(episode_rewards),
                           "success_rate": np.count_nonzero(np.greater(episode_rewards, 0)) / len(episode_rewards),
                           "num_updates": j,
                           "value_loss": float(value_loss),
                           "action_loss": float(action_loss),
                           "dist_entropy": float(dist_entropy)
                           }, step = total_num_steps)

            if args.task_switch_interval > 0 and task_switch_counter > args.task_switch_interval:
                obs = envs.switch()
                task_switch_counter = 0

            if args.eval_interval is not None and len(episode_rewards) > 1 and j % args.eval_interval == 0:
                eval_envs = make_vec_envs(args.env_name, args.seed + args.num_processes, args.num_processes,
                                    args.gamma, eval_log_dir, args.add_timestep, device, True)

                if eval_envs.venv.__class__.__name__ == "VecNormalize":
                    eval_envs.venv.ob_rms = envs.venv.ob_rms

                    # An ugly hack to remove updates
                    def _obfilt(self, obs):
                        if self.ob_rms:
                            obs = np.clip((obs - self.ob_rms.mean) / np.sqrt(self.ob_rms.var + self.epsilon), -self.clipob, self.clipob)
                            return obs
                        else:
                            return obs

                    eval_envs.venv._obfilt = types.MethodType(_obfilt, envs.venv)

                eval_episode_rewards = []

                obs = eval_envs.reset()
                eval_recurrent_hidden_states = torch.zeros(args.num_processes,
                                actor_critic.recurrent_hidden_state_size, device=device)
                eval_masks = torch.zeros(args.num_processes, 1, device=device)

                # create a dummy feature
                eval_features = torch.zeros([args.num_processes, feature_size]).to(device)

                while len(eval_episode_rewards) < 10:
                    with torch.no_grad():
                        _, action, _, eval_recurrent_hidden_states = actor_critic.act(
                            obs, eval_recurrent_hidden_states, eval_masks, eval_features, deterministic=True)

                    # Obser reward and next obs
                    obs, reward, done, infos = eval_envs.step(action)

                    obs = torch.tensor(obs).to(device)
                    _feature = []
                    for info in infos:
                        if "feature" in info.keys():
                            _feature.append(info["feature"])
                    eval_feature = np.stack(_feature, axis = 0)
                    eval_feature = torch.tensor(eval_feature).to(device)

                    eval_masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done]).to(device)
                    for idx, eps_done in enumerate(done):
                        if eps_done:
                            eval_episode_rewards.append(np.array(reward[idx]))

                eval_envs.close()
                del obs
                del eval_feature
                del eval_masks


                print(" Evaluation using {} episodes: mean reward {:.5f}\n".format(
                    len(eval_episode_rewards),
                    np.mean(eval_episode_rewards)
                                   ))
                wandb.log({"mean_eval_reward": np.mean(eval_episode_rewards),
                           }, step = total_num_steps)


            """
            if args.vis and j % args.vis_interval == 0:
                try:
                    # Sometimes monitor doesn't properly flush the outputs
                    win = visdom_plot(viz, win, args.log_dir, args.env_name,
                                    args.algo, args.num_frames)
                except IOError:
                    pass
            """

        envs.close()

if __name__ == "__main__":

    set_start_method('forkserver')
    multitask()
