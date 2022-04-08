import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class SF():
    def __init__(self,
                 psi_net,
                 feature_size=2,
                 phi_lr=3e-4,
                 psi_lr=3e-4,
                 w_lr = 1,
                 eps=None,
                 alpha=None,
                 max_grad_norm=10):

        self.policy = psi_net
        self.max_grad_norm = max_grad_norm

        self.feature_size = feature_size

        self.psi_optimizer = torch.optim.Adam(self.policy.psi_net.parameters(), lr=psi_lr)
        if self.policy.phi_net != None:
            self.phi_optimizer = torch.optim.Adam(self.policy.phi_net.parameters(), lr=phi_lr)

        self.w_optimizer = torch.optim.SGD([self.policy.estimated_w], lr = w_lr)

    def update(self, rollouts):
        obs_shape = rollouts.obs.size()[2:]
        action_shape = rollouts.actions.size()[-1]
        num_steps, num_processes, _ = rollouts.rewards.size()

        #  update psi net
        dims = [num_steps+1, num_processes]
        psi_loss, phi_loss, w_loss = self.policy.evaluate_actions(
            rollouts.obs.view(-1, *obs_shape),
            rollouts.recurrent_hidden_states.view(-1, self.policy.recurrent_hidden_state_size),
            rollouts.masks.view(-1, 1),
            rollouts.actions.view(-1, action_shape),
            rollouts.rewards.view(-1, 1),
            rollouts.features.view(-1, self.feature_size),
            dims = dims)

        # update psi
        self.psi_optimizer.zero_grad()
        psi_loss.backward()
        nn.utils.clip_grad_norm_(self.policy.psi_net.parameters(),
                                     self.max_grad_norm)
        self.psi_optimizer.step()

        # update phi if needed
        if self.policy.phi_net != None:
            self.phi_optimizer.zero_grad()
            phi_loss.backward()
            nn.utils.clip_grad_norm_(self.policy.phi_net.parameters(),
                             self.max_grad_norm)

            self.phi_optimizer.step()
        else:
            phi_loss = torch.tensor([-1])

        # update w
        self.w_optimizer.zero_grad()
        w_loss.backward()
        self.w_optimizer.step()

        return psi_loss.item(), phi_loss.item(), w_loss.item()
