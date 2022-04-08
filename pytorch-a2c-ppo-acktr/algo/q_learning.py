import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class QLearning():
    def __init__(self,
                 q_net,
                 feature_size=2,
                 lr = 3e-4,
                 eps=None,
                 alpha=None,
                 max_grad_norm=10):

        self.policy = q_net
        self.max_grad_norm = max_grad_norm

        self.feature_size = feature_size

        self.optimizer = torch.optim.Adam(self.policy.q_net.parameters(), lr=lr)

    def update(self, rollouts):
        obs_shape = rollouts.obs.size()[2:]
        action_shape = rollouts.actions.size()[-1]
        num_steps, num_processes, _ = rollouts.rewards.size()

        #  update psi net
        dims = [num_steps+1, num_processes]
        q_loss = self.policy.evaluate_actions(
            rollouts.obs.view(-1, *obs_shape),
            rollouts.recurrent_hidden_states.view(-1, self.policy.recurrent_hidden_state_size),
            rollouts.masks.view(-1, 1),
            rollouts.actions.view(-1, action_shape),
            rollouts.rewards.view(-1, 1),
            rollouts.features.view(-1, self.feature_size),
            dims = dims)

        # update psi
        self.optimizer.zero_grad()
        q_loss.backward()
        nn.utils.clip_grad_norm_(self.policy.q_net.parameters(),
                                     self.max_grad_norm)
        self.optimizer.step()


        return q_loss.item()
