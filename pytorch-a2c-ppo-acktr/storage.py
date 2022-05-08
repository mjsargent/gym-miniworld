import torch
import numpy as np
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler


def _flatten_helper(T, N, _tensor):
    return _tensor.view(T * N, *_tensor.size()[2:])


class RolloutStorage(object):
    def __init__(self, num_steps, num_processes, obs_shape, action_space, recurrent_hidden_state_size, feature_dim, a2csf = False, learn_phi = False, z_samples = 0):
        self.a2csf = a2csf
        self.obs = torch.zeros(num_steps + 1, num_processes, *obs_shape)
        self.recurrent_hidden_states = torch.zeros(num_steps + 1, num_processes, recurrent_hidden_state_size)
        self.rewards = torch.zeros(num_steps, num_processes, 1)

        self.returns = torch.zeros(num_steps + 1, num_processes, 1)
        self.feature_dim = feature_dim
        if z_samples > 0:
            self.psi_returns = torch.zeros(num_steps + 1, num_processes * (z_samples + 1), feature_dim)
            self.multitask = True
            self.value_preds = torch.zeros(num_steps + 1, num_processes * (z_samples + 1), 1)
        else:
            self.psi_returns = torch.zeros(num_steps + 1, num_processes, feature_dim)
            self.multitask = False
            self.value_preds = torch.zeros(num_steps + 1, num_processes, 1)
        self.action_log_probs = torch.zeros(num_steps, num_processes, 1)
        self.learn_phi = learn_phi
        # if learning the features, note that what is inserted is the embedding
        # of the current state, therefore we need to shift this by one
        if self.a2csf:
            if self.learn_phi:
                self.features = torch.zeros(num_steps + 1, num_processes, feature_dim)
            else:
                self.features = torch.zeros(num_steps, num_processes, feature_dim)
            if self.multitask:
                self.psis = torch.zeros(num_steps, num_processes * (z_samples + 1), feature_dim)
            else:
                self.psis = torch.zeros(num_steps, num_processes, feature_dim)
        else:
            self.features = torch.zeros(num_steps + 1 , num_processes, feature_dim)

        # psi has dim feature dim not feature_dim * num_actions as it is used
        # for the policy gradient version of the algorithm not the value based self.psis = torch.zeros(num_steps, num_processes, feature_dim)
        self.estimated_rewards = torch.zeros(num_steps, num_processes, 1)
        if action_space.__class__.__name__ == 'Discrete':
            action_shape = 1
        else:
            action_shape = action_space.shape[0]
        self.actions = torch.zeros(num_steps, num_processes, action_shape)
        if action_space.__class__.__name__ == 'Discrete':
            self.actions = self.actions.long()
        self.masks = torch.ones(num_steps + 1, num_processes, 1)

        self.num_steps = num_steps
        self.step = 0

        # multitask storage
        self.z_samples = z_samples
        if self.multitask:
            self.w_storage = torch.zeros(num_steps, num_processes, self.feature_dim)
            self.z_storage = torch.zeros(num_steps, num_processes, self.z_samples, self.feature_dim)


    def to(self, device):
        self.obs = self.obs.to(device)
        self.recurrent_hidden_states = self.recurrent_hidden_states.to(device)
        self.rewards = self.rewards.to(device)
        self.value_preds = self.value_preds.to(device)
        self.returns = self.returns.to(device)
        self.psi_returns = self.psi_returns.to(device)
        self.action_log_probs = self.action_log_probs.to(device)
        self.actions = self.actions.to(device)
        self.masks = self.masks.to(device)
        self.features = self.features.to(device)
        if self.a2csf:
            self.psis = self.psis.to(device)
        self.estimated_rewards = self.estimated_rewards.to(device)
        if self.multitask:
            self.w_storage = self.w_storage.to(device)
            self.z_storage = self.z_storage.to(device)

    def insert(self, obs, recurrent_hidden_states, actions, action_log_probs, value_preds, rewards, masks, feature, psi, estimated_reward, w = None,z = None):
        self.obs[self.step + 1].copy_(obs)
        self.recurrent_hidden_states[self.step + 1].copy_(recurrent_hidden_states)
        self.actions[self.step].copy_(actions)
        if action_log_probs != None:
            self.action_log_probs[self.step].copy_(action_log_probs)
        if value_preds != None:
            self.value_preds[self.step].copy_(value_preds)
        self.rewards[self.step].copy_(rewards)
        self.masks[self.step + 1].copy_(masks)
        if self.a2csf:
            self.features[self.step].copy_(feature)
        else:
            self.features[self.step + 1].copy_(feature)
        if psi != None:
            self.psis[self.step].copy_(psi)
        if estimated_reward != None:
            self.estimated_rewards[self.step].copy_(estimated_reward)
        self.step = (self.step + 1) % self.num_steps
        if self.multitask > 0:
            self.w_storage[self.step].copy_(w)
            self.z_storage[self.step].copy_(z)

    def after_update(self):
        self.obs[0].copy_(self.obs[-1])
        self.recurrent_hidden_states[0].copy_(self.recurrent_hidden_states[-1])
        self.masks[0].copy_(self.masks[-1])
        # TODO double check if this is needed
        if not self.a2csf:
            self.features[0].copy_(self.features[-1])
        self.w_storage[0].copy_(self.w_storage[-1])

    def compute_returns(self, next_value, use_gae, gamma, tau, sf = False):
        if use_gae:
            self.value_preds[-1] = next_value
            gae = 0
            for step in reversed(range(self.rewards.size(0))):
                delta = self.rewards[step] + gamma * self.value_preds[step + 1] * self.masks[step + 1] - self.value_preds[step]
                gae = delta + gamma * tau * self.masks[step + 1] * gae
                self.returns[step] = gae + self.value_preds[step]
        else:
            self.returns[-1] = next_value
            if sf:
                for step in reversed(range(self.estimated_rewards.size(0))):
                    self.returns[step] = self.returns[step + 1] * \
                        gamma * self.masks[step + 1] + self.estimated_rewards[step]
            else:
                for step in reversed(range(self.rewards.size(0))):
                    self.returns[step] = self.returns[step + 1] * \
                        gamma * self.masks[step + 1] + self.rewards[step]

    def compute_psi_returns(self, next_psi, gamma, final_phi = None):

        self.psi_returns[-1] = next_psi
        if self.learn_phi:
            self.feature[-1] = final_phi
            for step in reversed(range(self.psis.size(0))):
                self.psi_returns[step] = self.psi_returns[step + 1] * \
                gamma * self.masks[step+1].repeat(1,self.psi_returns.shape[-1]) + self.features[step+1]
        else:
            interleave_factor = self.psi_returns[0].shape[0] // self.features[0].shape[0]
            if self.multitask:
                for step in reversed(range(self.psis.size(0))):
                    self.psi_returns[step] = self.psi_returns[step + 1] * \
                    gamma * torch.repeat_interleave(self.masks[step+1].repeat(1,self.psi_returns.shape[-1]), interleave_factor, dim = 0) + \
                    torch.repeat_interleave(self.features[step], interleave_factor, dim = 0)
            else:
                for step in reversed(range(self.psis.size(0))):
                    self.psi_returns[step] = self.psi_returns[step + 1] * \
                    gamma * self.masks[step+1].repeat(1,self.psi_returns.shape[-1]) + \
                    self.features[step]


        # function to use if computing the returns where the critic is
        # based on a successor feature

    def feed_forward_generator(self, advantages, num_mini_batch):
        num_steps, num_processes = self.rewards.size()[0:2]
        batch_size = num_processes * num_steps
        assert batch_size >= num_mini_batch, (
            "PPO requires the number of processes ({}) "
            "* number of steps ({}) = {} "
            "to be greater than or equal to the number of PPO mini batches ({})."
            "".format(num_processes, num_steps, num_processes * num_steps, num_mini_batch))
        mini_batch_size = batch_size // num_mini_batch
        sampler = BatchSampler(SubsetRandomSampler(range(batch_size)), mini_batch_size, drop_last=False)
        for indices in sampler:
            obs_batch = self.obs[:-1].view(-1, *self.obs.size()[2:])[indices]
            recurrent_hidden_states_batch = self.recurrent_hidden_states[:-1].view(-1,
                self.recurrent_hidden_states.size(-1))[indices]
            actions_batch = self.actions.view(-1, self.actions.size(-1))[indices]
            return_batch = self.returns[:-1].view(-1, 1)[indices]
            masks_batch = self.masks[:-1].view(-1, 1)[indices]
            old_action_log_probs_batch = self.action_log_probs.view(-1, 1)[indices]
            adv_targ = advantages.view(-1, 1)[indices]

            yield obs_batch, recurrent_hidden_states_batch, actions_batch, \
                return_batch, masks_batch, old_action_log_probs_batch, adv_targ

    def recurrent_generator(self, advantages, num_mini_batch):
        num_processes = self.rewards.size(1)
        assert num_processes >= num_mini_batch, (
            "PPO requires the number of processes ({}) "
            "to be greater than or equal to the number of "
            "PPO mini batches ({}).".format(num_processes, num_mini_batch))
        num_envs_per_batch = num_processes // num_mini_batch
        perm = torch.randperm(num_processes)
        for start_ind in range(0, num_processes, num_envs_per_batch):
            obs_batch = []
            recurrent_hidden_states_batch = []
            actions_batch = []
            return_batch = []
            masks_batch = []
            old_action_log_probs_batch = []
            adv_targ = []

            for offset in range(num_envs_per_batch):
                ind = perm[start_ind + offset]
                obs_batch.append(self.obs[:-1, ind])
                recurrent_hidden_states_batch.append(self.recurrent_hidden_states[0:1, ind])
                actions_batch.append(self.actions[:, ind])
                return_batch.append(self.returns[:-1, ind])
                masks_batch.append(self.masks[:-1, ind])
                old_action_log_probs_batch.append(self.action_log_probs[:, ind])
                adv_targ.append(advantages[:, ind])

            T, N = self.num_steps, num_envs_per_batch
            # These are all tensors of size (T, N, -1)
            obs_batch = torch.stack(obs_batch, 1)
            actions_batch = torch.stack(actions_batch, 1)
            return_batch = torch.stack(return_batch, 1)
            masks_batch = torch.stack(masks_batch, 1)
            old_action_log_probs_batch = torch.stack(old_action_log_probs_batch, 1)
            adv_targ = torch.stack(adv_targ, 1)

            # States is just a (N, -1) tensor
            recurrent_hidden_states_batch = torch.stack(recurrent_hidden_states_batch, 1).view(N, -1)

            # Flatten the (T, N, ...) tensors to (T * N, ...)
            obs_batch = _flatten_helper(T, N, obs_batch)
            actions_batch = _flatten_helper(T, N, actions_batch)
            return_batch = _flatten_helper(T, N, return_batch)
            masks_batch = _flatten_helper(T, N, masks_batch)
            old_action_log_probs_batch = _flatten_helper(T, N, \
                    old_action_log_probs_batch)
            adv_targ = _flatten_helper(T, N, adv_targ)

            yield obs_batch, recurrent_hidden_states_batch, actions_batch, \
                return_batch, masks_batch, old_action_log_probs_batch, adv_targ

# use a different buffer for temporally extended algorithms -
# does not include the ppo specific methods
class TemporallyExtendedRolloutStorage(object):
    def __init__(self, num_steps, num_processes, obs_shape, action_space, recurrent_hidden_state_size, feature_dim, a2csf = False):
        self.a2csf = a2csf
        self.obs = torch.zeros(num_steps + 1, num_processes, *obs_shape)
        self.recurrent_hidden_states = torch.zeros(num_steps + 1, num_processes, recurrent_hidden_state_size)
        self.rewards = torch.zeros(num_steps, num_processes, 1)
        self.value_preds = torch.zeros(num_steps + 1, num_processes, 1)
        self.returns = torch.zeros(num_steps + 1, num_processes, 1)
        self.psi_returns = torch.zeros(num_steps + 1, num_processes, feature_dim)
        self.action_log_probs = torch.zeros(num_steps, num_processes, 1)
        self.repeat_log_probs = torch.zeros(num_steps, num_processes, 1)
        # define two variables related to action reps -
        # one for keep tracking of what we output, and one for keeping
        # track of the actual number of actions repeated for discounting
        self.repeats = torch.zeros(num_steps, num_processes, 1).long()
        self.steps_taken = torch.zeros(num_steps, num_processes, 1)
        if self.a2csf:
            self.psis =  torch.zeros(num_steps, num_processes, feature_dim)
            self.features = torch.zeros(num_steps, num_processes, feature_dim)
        else:
            self.features = torch.zeros(num_steps + 1 , num_processes, feature_dim)

        # psi has dim feature dim not feature_dim * num_actions as it is used
        # for the policy gradient version of the algorithm not the value based self.psis = torch.zeros(num_steps, num_processes, feature_dim)
        self.estimated_rewards = torch.zeros(num_steps, num_processes, 1)
        if action_space.__class__.__name__ == 'Discrete':
            action_shape = 1
        else:
            action_shape = action_space.shape[0]
        self.actions = torch.zeros(num_steps, num_processes, action_shape)
        if action_space.__class__.__name__ == 'Discrete':
            self.actions = self.actions.long()
        self.masks = torch.ones(num_steps + 1, num_processes, 1)

        self.num_steps = num_steps
        self.step = 0

    def to(self, device):
        self.obs = self.obs.to(device)
        self.recurrent_hidden_states = self.recurrent_hidden_states.to(device)
        self.rewards = self.rewards.to(device)
        self.returns = self.returns.to(device)
        self.action_log_probs = self.action_log_probs.to(device)
        self.repeat_log_probs = self.repeat_log_probs.to(device)
        self.actions = self.actions.to(device)
        self.masks = self.masks.to(device)
        self.features = self.features.to(device)
        if self.a2csf:
            self.psis = self.psis.to(device)
            self.estimated_rewards = self.estimated_rewards.to(device)#
            self.psi_returns = self.psi_returns.to(device)
        self.repeats = self.repeats.to(device)
        self.steps_taken = self.steps_taken.to(device)

    def insert(self, obs, recurrent_hidden_states, actions, action_log_probs, value_preds, rewards, masks, feature, psi, estimated_reward, repeat, repeat_log_prob, steps_taken):
        self.obs[self.step + 1].copy_(obs)
        self.recurrent_hidden_states[self.step + 1].copy_(recurrent_hidden_states)
        self.actions[self.step].copy_(actions)
        if action_log_probs != None:
            self.action_log_probs[self.step].copy_(action_log_probs)
        if value_preds != None:
            self.value_preds[self.step].copy_(value_preds)
        if repeat_log_prob != None:
            self.repeat_log_probs[self.step].copy_(repeat_log_prob)
        self.rewards[self.step].copy_(rewards)
        self.masks[self.step + 1].copy_(masks)
        if self.a2csf:
            self.features[self.step].copy_(feature)
        else:
            self.features[self.step + 1].copy_(feature)
        if psi != None:
            self.psis[self.step].copy_(psi)
        if estimated_reward != None:
            self.estimated_rewards[self.step].copy_(estimated_reward)
        self.steps_taken[self.step].copy_(steps_taken.detach())
        self.repeats[self.step].copy_(repeat.detach())

        self.step = (self.step + 1) % self.num_steps


    def after_update(self):
        self.obs[0].copy_(self.obs[-1])
        self.recurrent_hidden_states[0].copy_(self.recurrent_hidden_states[-1])
        self.masks[0].copy_(self.masks[-1])
        # TODO double check if this is needed
        if not self.a2csf:
            self.features[0].copy_(self.features[-1])

    def compute_returns(self, next_value, use_gae, gamma, tau, sf = False):
        if use_gae:
            self.value_preds[-1] = next_value
            gae = 0
            for step in reversed(range(self.rewards.size(0))):
                delta = self.rewards[step] + gamma * self.value_preds[step + 1] * self.masks[step + 1] - self.value_preds[step]
                gae = delta + gamma * tau * self.masks[step + 1] * gae
                self.returns[step] = gae + self.value_preds[step]
        else:
            self.returns[-1] = next_value
            if sf:
                for step in reversed(range(self.estimated_rewards.size(0))):
                    self.returns[step] = self.returns[step + 1] * \
                        gamma * self.masks[step + 1] + self.estimated_rewards[step]
            else:
                for step in reversed(range(self.rewards.size(0))):
                    self.returns[step] = self.returns[step + 1] * \
                        (gamma ** (self.repeats[step] + 1)) *  self.masks[step + 1] + self.rewards[step]
                        #(gamma ** self.steps_taken[step]) *  self.masks[step + 1] + self.rewards[step]

    def compute_psi_returns(self, next_psi, gamma):
        self.psi_returns[-1] = next_psi
        for step in reversed(range(self.psis.size(0))):
            self.psi_returns[step] = self.psi_returns[step + 1] * \
                (gamma ** self.steps_taken[step]) * self.masks[step+1].repeat(1,self.psi_returns.shape[-1]) + self.features[step]

        # function to use if computing the returns where the critic is
        # based on a successor feature


