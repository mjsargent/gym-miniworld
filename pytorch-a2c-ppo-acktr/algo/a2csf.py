import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class A2C_SF():
    def __init__(self,
                 actor_critic,
                 value_loss_coef,
                 entropy_coef,
                 feature_size=2,
                 lr_policy=None,
                 lr_psi=None,
                 lr_phi=3e-3,
                 lr_w=None,
                 eps=None,
                 alpha=None,
                 max_grad_norm=None,
                 gamma=0.99,
                 learn_phi = False,
                 multitask = False
                ):

        self.actor_critic = actor_critic

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        self.max_grad_norm = max_grad_norm
        self.feature_size = feature_size
        self.gamma = gamma
        self.learn_phi = learn_phi
        self.multitask = multitask

            #self.policy_optimizer = optim.RMSprop(
            #[*actor_critic.base.main.parameters(), *actor_critic.base.gru.parameters(), \
             #*actor_critic.dist.parameters(), *actor_critic.base.sf.parameters()], lr=lr_policy, eps=eps, alpha=alpha)

        self.policy_optimizer = optim.RMSprop(actor_critic.parameters(),lr=lr_policy, eps=eps, alpha=alpha)
        self.w_optimizer = optim.SGD([actor_critic.base.w], lr=lr_w)

        if learn_phi:
            self.phi_optimizer = optim.RMSprop(actor_critic.phi_net.parameters(), lr=lr_phi, eps=eps, alpha=alpha)


    def update(self, rollouts):
        obs_shape = rollouts.obs.size()[2:]
        action_shape = rollouts.actions.size()[-1]
        num_steps, num_processes, _ = rollouts.rewards.size()

        values, action_log_probs, dist_entropy, _, psi = self.actor_critic.evaluate_actions(
            rollouts.obs[:-1].view(-1, *obs_shape),
            rollouts.recurrent_hidden_states[0].view(-1, self.actor_critic.recurrent_hidden_state_size),
            rollouts.masks[:-1].view(-1, 1),
            rollouts.actions.view(-1, action_shape),
            rollouts.features[:-1].view(-1, self.feature_size))

        # compute psi loss
        current_psi = psi.view(num_steps, num_processes, -1)
        psi_advantages = rollouts.psi_returns[:-1] - current_psi
        psi_loss = psi_advantages.pow(2).sum(-1).mean()

        #psi_loss = F.mse_loss(current_psi, psi_target)
        #psi_loss = psi_loss.mean()

        # compute policy gradient
        values = values.view(num_steps, num_processes, 1)
        action_log_probs = action_log_probs.view(num_steps, num_processes, 1)


        #advantages = rollouts.returns[:-1] - values.detach()
        advantages = (psi_advantages * self.actor_critic.base.w.squeeze(0)).sum(-1)
        #value_loss = advantages.pow(2).mean()

        action_loss = -(advantages.detach().unsqueeze(-1) * action_log_probs).mean()

        #p_loss = (value_loss * self.value_loss_coef + action_loss -
        # dist_entropy * self.entropy_coef)

        self.policy_optimizer.zero_grad()

        p_loss = (action_loss - dist_entropy * self.entropy_coef)

        p_loss = p_loss +  psi_loss
        p_loss.backward()

        nn.utils.clip_grad_norm_(self.actor_critic.parameters(),
                                     self.max_grad_norm)

        self.policy_optimizer.step()

        #psi_loss = value_loss
        value_loss = psi_loss

        if self.learn_phi:
            #inputs = rollouts.obs.reshape(num_steps, num_processes, *obs_shape)
            inputs = rollouts.obs[:-1].reshape(-1, *obs_shape)
            next_inputs = rollouts.obs[1:].reshape(-1, *obs_shape)

            prediction, phi = self.actor_critic.phi_net(inputs.view(-1, *obs_shape), rollouts.actions.reshape(-1, 1))

            phi_loss = F.mse_loss(prediction, next_inputs)

            predicted_rewards = torch.matmul(phi, self.actor_critic.base.w.t().detach())
            reward_loss = F.mse_loss(rollouts.rewards.view(-1,1), predicted_rewards)

            phi_loss = phi_loss + reward_loss

            self.phi_optimizer.zero_grad()
            phi_loss.backward()
            self.phi_optimizer.step()

        else:
            phi_loss = torch.tensor([0])


        # compute w loss

        if not self.multitask:
            predicted_rewards = torch.matmul(rollouts.features.view(-1, self.feature_size), \
                                         self.actor_critic.base.w.t())


            w_loss = F.mse_loss(rollouts.rewards.view(-1,1), predicted_rewards)

            w_loss.backward()

            self.w_optimizer.step()

            self.w_optimizer.zero_grad()
        else:
            w_loss = torch.tensor([0])

        return value_loss.item(), action_loss.item(), dist_entropy.item(), \
            psi_loss.item(), w_loss.item(), phi_loss.item()
