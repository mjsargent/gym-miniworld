import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from .kfac import KFACOptimizer


class A2CSFX():
    def __init__(self,
                 actor_critic,
                 value_loss_coef,
                 entropy_coef,
                 feature_size=2,
                 lr=None,
                 lr_w=1,
                 eps=None,
                 alpha=None,
                 max_grad_norm=None,
                 gamma = 0.99
                ):
        self.actor_critic = actor_critic

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        self.max_grad_norm = max_grad_norm
        self.feature_size = feature_size

        self.optimizer = optim.RMSprop(
            actor_critic.parameters(), lr, eps=eps, alpha=alpha)
        self.policy_optimizer = optim.RMSprop(actor_critic.parameters(), lr=lr,eps = eps, alpha =alpha)
        self.w_optimizer = optim.SGD([actor_critic.base.w], lr=lr_w)

    def update(self, rollouts, action_only = False):
        obs_shape = rollouts.obs.size()[2:]
        action_shape = rollouts.actions.size()[-1]
        num_decisions, num_processes, _ = rollouts.rewards.size()
        #

        values, action_log_probs, dist_entropy, _, repeat_log_probs, repeat_dist_entropy, psi, _ = self.actor_critic.evaluate_actions(
            rollouts.obs[:-1].view(-1, *obs_shape),
            rollouts.recurrent_hidden_states[0].view(-1, self.actor_critic.recurrent_hidden_state_size),
            rollouts.masks[:-1].view(-1, 1),
            rollouts.actions.view(-1, action_shape),
            rollouts.repeats.view(-1, 1),
            rollouts.features[:-1].view(-1, self.feature_size))
        values = values.view(num_decisions, num_processes, 1)
        current_psi = psi.view(num_decisions, num_processes, -1)

        action_log_probs = action_log_probs.view(num_decisions, num_processes, 1)
        repeat_log_probs = repeat_log_probs.view(num_decisions, num_processes, 1)

        psi_advantages = rollouts.psi_returns[:-1] - current_psi
        psi_loss =psi_advantages.pow(2).sum(-1).mean()

        #value_loss = advantages.pow(2).mean()
        predicted_rewards = torch.matmul(rollouts.features[:-1].view(-1, self.feature_size),
                                         self.actor_critic.base.w.t())

        w_loss = F.mse_loss(rollouts.rewards[:-1].view(-1,1), predicted_rewards)
        action_log_probs = action_log_probs.view(num_decisions, num_processes, 1)
        repeat_log_probs = repeat_log_probs.view(num_decisions, num_processes, 1)

        #action_loss = -(advantages.detach() * action_log_probs).mean()
        #repeat_loss = -(advantages.detach() * repeat_log_probs).mean()

        advantages = (psi_advantages * self.actor_critic.base.w.squeeze(0)).sum(-1).unsqueeze(-1)
        if action_only:
            action_loss = -(advantages.detach() * action_log_probs ).mean()
        else:
            action_loss = -(advantages.detach() * (action_log_probs + repeat_log_probs) ).mean()

        self.policy_optimizer.zero_grad()
        self.w_optimizer.zero_grad()
        if action_only:
            p_loss = action_loss - \
                dist_entropy * self.entropy_coef
        else:
            p_loss = action_loss - \
                dist_entropy * self.entropy_coef -  \
                repeat_dist_entropy * self.entropy_coef

        p_loss = p_loss + psi_loss
        nn.utils.clip_grad_norm_(self.actor_critic.parameters(),
                                 self.max_grad_norm)
        p_loss.backward()

        self.policy_optimizer.step()

        w_loss.backward()

        self.w_optimizer.step()

        value_loss = psi_loss

        return value_loss.item(), action_loss.item(), dist_entropy.item(), action_loss.item(), repeat_dist_entropy.item(), psi_loss.item(), w_loss.item()
