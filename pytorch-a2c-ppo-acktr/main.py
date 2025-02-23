import copy
import glob
import os
import time
import types
from collections import deque

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import algo
from arguments import get_args
from envs import make_vec_envs
from model import Policy, SFPolicy, QPolicy, SFConditionedPolicy
from storage import RolloutStorage
#from visualize import visdom_plot

import wandb

args = get_args()

assert args.algo in ['a2c', 'ppo', 'acktr', 'sf', "q", "a2csf"]
if args.recurrent_policy:
    assert args.algo in ['a2c', 'ppo', "sf", "q", "a2csf"], \
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


def main():
    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")

    """
    if args.vis:
        from visdom import Visdom
        viz = Visdom(port=args.port)
        win = None
    """

    feature_size = 2
    envs = make_vec_envs(args.env_name, args.seed, args.num_processes,
                        args.gamma, args.log_dir, args.add_timestep, device, False)

    if args.algo == 'sf':
        policy= SFPolicy(envs.observation_space.shape, envs.action_space, feature_size = 2,
            base_kwargs={'recurrent': args.recurrent_policy})
        policy.to(device)
    elif args.algo == "q":
        policy= QPolicy(envs.observation_space.shape, envs.action_space, feature_size = 2,
            base_kwargs={'recurrent': args.recurrent_policy})
        policy.to(device)

    elif args.algo == "a2csf":
        actor_critic = SFConditionedPolicy(envs.observation_space.shape, envs.action_space, feature_size = 2,
            base_kwargs={'recurrent': args.recurrent_policy})
        actor_critic.to(device)
    else:
        actor_critic = Policy(envs.observation_space.shape, envs.action_space, feature_size = 2,
            base_kwargs={'recurrent': args.recurrent_policy})
        actor_critic.to(device)


    if args.algo == 'a2c':
        agent = algo.A2C_ACKTR(actor_critic, args.value_loss_coef,
                               args.entropy_coef, lr=args.lr,
                               eps=args.eps, alpha=args.alpha,
                               max_grad_norm=args.max_grad_norm,
                               feature_size = 2)

    elif args.algo == 'a2csf':
        agent = algo.A2C_SF(actor_critic, args.value_loss_coef,
                               args.entropy_coef, lr_psi=args.lr,
                               lr_policy = args.lr, lr_w = 1,
                               eps=args.eps, alpha=args.alpha,
                               max_grad_norm=args.max_grad_norm,
                               feature_size = 2, gamma=args.gamma)
    elif args.algo == 'ppo':
        agent = algo.PPO(actor_critic, args.clip_param, args.ppo_epoch, args.num_mini_batch,
                         args.value_loss_coef, args.entropy_coef, lr=args.lr,
                               eps=args.eps,
                               max_grad_norm=args.max_grad_norm)
    elif args.algo == 'acktr':
        agent = algo.A2C_ACKTR(actor_critic, args.value_loss_coef,
                               args.entropy_coef, acktr=True)
    elif args.algo == 'sf':
        agent = algo.SF(policy, feature_size = feature_size,
                        phi_lr=3e-4, psi_lr=3e-4, eps=args.eps_explore)

    elif args.algo == 'q':
        agent = algo.QLearning(policy, feature_size = feature_size,
                        lr=args.lr, eps=args.eps_explore)

    use_a2csf_storage = True if args.algo == "a2csf" else False

    if args.algo == "sf" or args.algo == "q":
        rollouts = RolloutStorage(args.num_steps, args.num_processes,
                        envs.observation_space.shape, envs.action_space,
                        policy.recurrent_hidden_state_size, feature_dim = feature_size)

    else:
        rollouts = RolloutStorage(args.num_steps, args.num_processes,
                        envs.observation_space.shape, envs.action_space,
                        actor_critic.recurrent_hidden_state_size, feature_dim = feature_size,
                                  a2csf = use_a2csf_storage)


    obs = envs.reset()

    # create a dummy feature
    dummy_feature = torch.zeros([args.num_processes, feature_size])

    rollouts.features[0].copy_(dummy_feature)
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    episode_rewards = deque(maxlen=100)

    start = time.time()
    wandb.init(project = "tSF")

    if args.algo == "sf":
        for j in range(num_updates):
            for step in range(args.num_steps):
                # Sample actions
                with torch.no_grad():
                    _, _, action, _ ,  recurrent_hidden_states = policy.act(
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

                # FIXME: works only for environments with sparse rewards
                for idx, eps_done in enumerate(done):
                    if eps_done:
                        episode_rewards.append(np.array(reward[idx]))

                # If done then clean the history of observations.
                masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
                rollouts.insert(obs = obs,recurrent_hidden_states =  recurrent_hidden_states,
                                action_log_probs = None, value_preds = None,
                                actions = action, rewards = reward, masks = masks,
                                feature = feature)

            psi_loss, phi_loss, w_loss = agent.update(rollouts)

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
                save_model = policy
                if args.cuda:
                    save_model = copy.deepcopy(policy).cpu()

                save_model = [save_model, hasattr(envs.venv, 'ob_rms') and envs.venv.ob_rms or None]

                torch.save(save_model, os.path.join(save_path, args.env_name + ".pt"))

            total_num_steps = (j + 1) * args.num_processes * args.num_steps

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
                           "psi_loss": float(psi_loss),
                           "phi_loss": float(phi_loss),
                           "w_loss": float(w_loss)
                           }, step = total_num_steps)

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
                eval_recurrent_hidden_states = torch.zeros(args.num_processes, actor_critic.recurrent_hidden_state_size, device=device)
                eval_masks = torch.zeros(args.num_processes, 1, device=device)
                # create a dummy feature
                eval_features = torch.zeros([args.num_processes, feature_size])
                while len(eval_episode_rewards) < 10:
                    with torch.no_grad():
                        _, action, _, _, eval_recurrent_hidden_states = policy.act( obs, eval_recurrent_hidden_states, eval_masks, eval_features, deterministic=True)
                    # Obser reward and next obs
                    obs, reward, done, infos = eval_envs.step(action)
                    _feature = []
                    for info in infos:
                        if "feature" in info.keys():
                            _feature.append(info["feature"])
                    eval_feature = np.stack(_feature, axis = 0)

                    eval_masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
                    for info in infos:
                        if 'episode' in info.keys():
                            eval_episode_rewards.append(info['episode']['r'])

                eval_envs.close()

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

    elif args.algo == "q":
        for j in range(num_updates):
            for step in range(args.num_steps):
                # Sample actions
                with torch.no_grad():
                    _, action, _, recurrent_hidden_states = policy.act(
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

                # FIXME: works only for environments with sparse rewards
                for idx, eps_done in enumerate(done):
                    if eps_done:
                        episode_rewards.append(np.array(reward[idx]))

                # If done then clean the history of observations.
                masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
                rollouts.insert(obs = obs,recurrent_hidden_states =  recurrent_hidden_states,
                                action_log_probs = None, value_preds = None,
                                actions = action, rewards = reward, masks = masks,
                                feature = feature)


            q_loss = agent.update(rollouts)

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
                save_model = policy
                if args.cuda:
                    save_model = copy.deepcopy(policy).cpu()

                save_model = [save_model, hasattr(envs.venv, 'ob_rms') and envs.venv.ob_rms or None]

                torch.save(save_model, os.path.join(save_path, args.env_name + ".pt"))

            total_num_steps = (j + 1) * args.num_processes * args.num_steps

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
                           "q_loss": float(q_loss),
                           }, step = total_num_steps)

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
                eval_recurrent_hidden_states = torch.zeros(args.num_processes, actor_critic.recurrent_hidden_state_size, device=device)
                eval_masks = torch.zeros(args.num_processes, 1, device=device)
                # create a dummy feature
                eval_features = torch.zeros([args.num_processes, feature_size])
                while len(eval_episode_rewards) < 10:
                    with torch.no_grad():
                        _, action, _, eval_recurrent_hidden_states = policy.act( obs, eval_recurrent_hidden_states, eval_masks, eval_features, deterministic=True)
                    # Obser reward and next obs
                    obs, reward, done, infos = eval_envs.step(action)
                    _feature = []
                    for info in infos:
                        if "feature" in info.keys():
                            _feature.append(info["feature"])
                    eval_feature = np.stack(_feature, axis = 0)

                    eval_masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
                    for info in infos:
                        if 'episode' in info.keys():
                            eval_episode_rewards.append(info['episode']['r'])

                eval_envs.close()

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

    elif args.algo == "a2csf":
        for j in range(num_updates):
            for step in range(args.num_steps):
                # Sample actions
                with torch.no_grad():

                    value, action, action_log_prob, recurrent_hidden_states, psi = actor_critic.act(
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

                feature = torch.FloatTensor(np.stack(_feature, axis = 0)).to(device)
                estimated_reward = actor_critic.evaluate_rewards(feature)
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
                rollouts.insert(obs, recurrent_hidden_states, action, action_log_prob, value, reward, masks, feature, psi, estimated_reward)

            with torch.no_grad():
                next_value, next_psi = actor_critic.get_value(rollouts.obs[-1],
                                                    rollouts.recurrent_hidden_states[-1],
                                                    rollouts.masks[-1],
                                                    rollouts.features[-1])


            rollouts.compute_returns(next_value, args.use_gae, args.gamma, args.tau, sf = True)
            rollouts.compute_psi_returns(next_psi,args.gamma)

            value_loss, action_loss, dist_entropy, psi_loss, w_loss = agent.update(rollouts)

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
                           "w_loss": float(w_loss)
                           }, step = total_num_steps)



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
                eval_features = torch.zeros([args.num_processes, feature_size])

                while len(eval_episode_rewards) < 10:
                    with torch.no_grad():
                        _, action, _, eval_recurrent_hidden_states = actor_critic.act(
                            obs, eval_recurrent_hidden_states, eval_masks, eval_features, deterministic=True)

                    # Obser reward and next obs
                    obs, reward, done, infos = eval_envs.step(action)

                    _feature = []
                    for info in infos:
                        if "feature" in info.keys():
                            _feature.append(info["feature"])
                    eval_feature = np.stack(_feature, axis = 0)

                    eval_masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
                    for info in infos:
                        if 'episode' in info.keys():
                            eval_episode_rewards.append(info['episode']['r'])

                eval_envs.close()

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
                if j > 5:
                    env_mask = np.array([0, 1, 0, 0])
                else:
                    env_mask = np.array([0, 0, 0, 0])
                obs, reward, done, infos = envs.step(action, env_mask)
                print(obs[:,0,0,0])

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
                eval_features = torch.zeros([args.num_processes, feature_size])

                while len(eval_episode_rewards) < 10:
                    with torch.no_grad():
                        _, action, _, eval_recurrent_hidden_states = actor_critic.act(
                            obs, eval_recurrent_hidden_states, eval_masks, eval_features, deterministic=True)

                    # Obser reward and next obs
                    obs, reward, done, infos = eval_envs.step(action)

                    _feature = []
                    for info in infos:
                        if "feature" in info.keys():
                            _feature.append(info["feature"])
                    eval_feature = np.stack(_feature, axis = 0)

                    eval_masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
                    for info in infos:
                        if 'episode' in info.keys():
                            eval_episode_rewards.append(info['episode']['r'])

                eval_envs.close()

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
    main()
