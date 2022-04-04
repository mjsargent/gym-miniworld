import torch
import torch.nn as nn
import torch.nn.functional as F

from distributions import Categorical, DiagGaussian
from utils import init, init_normc_


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Policy(nn.Module):
    def __init__(self, obs_shape, action_space, feature_size, base_kwargs=None):
        super(Policy, self).__init__()
        if base_kwargs is None:
            base_kwargs = {}

        if len(obs_shape) == 3:
            self.base = CNNBase(obs_shape[0], feature_size, **base_kwargs)
        elif len(obs_shape) == 1:
            self.base = MLPBase(obs_shape[0], feature_size, **base_kwargs)
        else:
            raise NotImplementedError

        if action_space.__class__.__name__ == "Discrete":
            num_outputs = action_space.n
            self.dist = Categorical(self.base.output_size, num_outputs)
        elif action_space.__class__.__name__ == "Box":
            num_outputs = action_space.shape[0]
            self.dist = DiagGaussian(self.base.output_size, num_outputs)
        else:
            raise NotImplementedError

    @property
    def is_recurrent(self):
        return self.base.is_recurrent

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return self.base.recurrent_hidden_state_size

    def forward(self, inputs, rnn_hxs, masks, features = None):
        raise NotImplementedError

    def act(self, inputs, rnn_hxs, masks, features, deterministic=False):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks, features)
        dist = self.dist(actor_features)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action, action_log_probs, rnn_hxs

    def get_value(self, inputs, rnn_hxs, masks, features = None):
        value, _, _ = self.base(inputs, rnn_hxs, masks, features)
        return value

    def evaluate_actions(self, inputs, rnn_hxs, masks, action, features = None):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks, features)
        dist = self.dist(actor_features)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy, rnn_hxs


class SFPolicy(nn.Module):
    def __init__(self, obs_shape, action_space, feature_size, learnt_phi = False, use_target_network = False, eps = 0.05, gamma = 0.99, base_kwargs=None):
        super(SFPolicy, self).__init__()
        if base_kwargs is None:
            base_kwargs = {}

        self.phi_net = None
        self.gamma = gamma
        self.eps = eps
        self.estimated_w = torch.nn.Parameter(torch.randn(feature_size))
        self.use_target_network = use_target_network
        self.feature_size =feature_size

        if action_space.__class__.__name__ == "Discrete":
            self.num_actions = action_space.n

        if len(obs_shape) == 3:
            self.psi_net = CNNBase(obs_shape[0], feature_size, SF=True,num_actions=action_space.n, **base_kwargs)
            if use_target_network:
                self.target_psi_net = CNNBase(obs_shape[0], feature_size,SF=True, num_actions=action_space.n, **base_kwargs)
                self.target_psi_net.load_state_dict(self.psi_net.state_dict())

            if learnt_phi == True:
                self.phi_net = CNNBase(obs_shape[0], feature_size = 0, **base_kwargs)

        elif len(obs_shape) == 1:
            self.psi_net = MLPBase(obs_shape[0], feature_size,num_actions=action_space.n, **base_kwargs)
            if use_target_network:
                self.target_psi_net = MLPBase(obs_shape[0], feature_size, SF=True,num_actions=action_space.n, **base_kwargs)
                self.target_psi_net.load_state_dict(self.psi_net.state_dict())

            if learnt_phi == True:
                self.phi_net = MLPBase(obs_shape[0], feature_size = 0, **base_kwargs)
        else:
            raise NotImplementedError

        #if action_space.__class__.__name__ == "Discrete":
        #    num_outputs = action_space.n
        #    self.dist = Categorical(self.psi_net.output_size, num_outputs)
        #elif action_space.__class__.__name__ == "Box":
        #    num_outputs = action_space.shape[0]
        #    self.dist = DiagGaussian(self.psi_net.output_size, num_outputs)
        #else:
        #    raise NotImplementedError

    @property
    def is_recurrent(self):
        return self.psi_net.is_recurrent

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return self.psi_net.recurrent_hidden_state_size

    def forward(self, inputs, rnn_hxs, masks, features = None):
        raise NotImplementedError

    def act(self, inputs, rnn_hxs, masks, features, deterministic=False):
        # return q values
        # psi: NXB, A*|phi|
        psi, _,  rnn_hxs = self.psi_net(inputs, rnn_hxs, masks, features)
        with torch.no_grad():
            psi = psi.reshape(-1,self.num_actions, self.feature_size)
            q = psi * self.estimated_w.unsqueeze(0).repeat(psi.shape[0], psi.shape[1], 1)
            q = q.sum(-1)

        if deterministic:
            r = torch.rand(psi.shape[0]).to(psi.device)
            action = torch.where(r < self.eps, torch.randint(q.shape[1], (psi.shape[0])).to(self.psi_net.device), torch.argmax(q, dim = -1))
        else:
            action = torch.argmax(q, dim = -1, keepdims = True)

        return q, psi, action, None, rnn_hxs

    def get_value(self, inputs, rnn_hxs, masks, features = None):
        # overloaded to mean q values
        q, _, _ = self.psi_net(inputs, rnn_hxs, masks, features)
        return q

    def evaluate_actions(self, inputs, rnn_hxs, masks, action, rewards, features = None, dims = None):
        # hacky, but we need to know the rollout length and the batch size in
        # order to align the states into t and t+1
        T = dims[0]
        B = dims[1]

        psi, _ , rnn_hxs = self.psi_net(inputs, rnn_hxs, masks, features)
        psi = psi.reshape(T, B, -1)

        s = inputs.reshape(T, B, -1)[:-1]
        next_s = inputs.reshape(T, B, -1)[1:]

        current_psi = psi[:-1].reshape((T-1)*B, self.num_actions, -1)

        _phi = features.reshape(T, B, -1)[1:]

        if self.use_target_network:
            #TODO is a target network needed here? USFA does not use one
            # as far as I can tell. VISR does use one.
            with torch.no_grad():
                # get t+1 of each input to the psi net
                next_rnn_hxs = rnn_hxs.reshape(T, B, -1)[1:].reshape((T-1)*B, -1)
                next_features= _phi.reshape((T-1)*B, B, -1)

                _ ,next_action, next_psi, _, _ = self.target_psi_net.act(next_s.reshape((T-1)*B, -1), rnn_hxs, features, deterministic = True)

            next_psi = next_psi.clone().detach().reshape((T-1)*B,self.num_actions,-1)
        else:
            next_psi = psi[1:].clone().detach().reshape((T-1)*B,self.num_actions,-1)
            next_q = next_psi * self.estimated_w.unsqueeze(0).unsqueeze(0).repeat(next_psi.shape[0], next_psi.shape[1], 1)
            next_q = next_q.sum(-1)
            next_action = torch.argmax(next_q, -1)


        # index with actions and best next actions
        current_psi = current_psi[range(B*(T-1)), action.reshape(-1)]
        next_psi = next_psi[range(B*(T-1)), next_action.reshape(-1)]

        with torch.no_grad():
            masks = masks.reshape(T, B, -1)
            masks = masks[1:].reshape((T-1)*B, -1).repeat(1, next_psi.shape[-1])
            target = _phi.reshape(next_psi.shape) + self.gamma *masks * next_psi

        # compute psi loss
        psi_loss = F.mse_loss(current_psi, target)

        # compute phi loss
        if self.phi_net != None:
            pass
        else:
            phi_loss = None

        # compute w_loss
        predicted_rewards = torch.matmul(_phi,self.estimated_w.unsqueeze(1))
        rewards = rewards.reshape(T-1, B, -1).reshape(-1,1)
        predicted_rewards = predicted_rewards.reshape(-1,1)
        w_loss = F.mse_loss(predicted_rewards, rewards)

        return psi_loss, phi_loss, w_loss


class NNBase(nn.Module):

    def __init__(self, recurrent, recurrent_input_size, hidden_size, feature_size):
        super(NNBase, self).__init__()

        self._hidden_size = hidden_size
        self._recurrent = recurrent
        self._feature_size = feature_size


        if recurrent:
            self.gru = nn.GRUCell(recurrent_input_size, hidden_size)
            nn.init.orthogonal_(self.gru.weight_ih.data)
            nn.init.orthogonal_(self.gru.weight_hh.data)
            self.gru.bias_ih.data.fill_(0)
            self.gru.bias_hh.data.fill_(0)

    @property
    def is_recurrent(self):
        return self._recurrent

    @property
    def recurrent_hidden_state_size(self):
        if self._recurrent:
            return self._hidden_size
        return 1

    @property
    def output_size(self):
        return self._hidden_size + self._feature_size

    def _forward_gru(self, x, hxs, masks):
        if x.size(0) == hxs.size(0):
            x = hxs = self.gru(x, hxs * masks)
        else:
            # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
            N = hxs.size(0)
            T = int(x.size(0) / N)

            # unflatten
            x = x.view(T, N, x.size(1))

            # Same deal with masks
            masks = masks.view(T, N, 1)

            outputs = []
            for i in range(T):
                hx = hxs = self.gru(x[i], hxs * masks[i])
                outputs.append(hx)

            # assert len(outputs) == T
            # x is a (T, N, -1) tensor
            x = torch.stack(outputs, dim=0)
            # flatten
            x = x.view(T * N, -1)

        return x, hxs


class Print(nn.Module):
    def __init__(self):
        super(Print, self).__init__()

    def forward(self, x):
        print('layer input:', x.shape)
        return x


class CNNBase(NNBase):
    def __init__(self, num_inputs, feature_size = 2, recurrent=False, hidden_size=128, SF=False, num_actions = 3):
        super(CNNBase, self).__init__(recurrent, hidden_size + feature_size, hidden_size, feature_size)
        self.feature_size = feature_size
        self.SF = SF

        init_ = lambda m: init(m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0),
            nn.init.calculate_gain('relu'))

        # For 80x60 input
        self.main = nn.Sequential(
            init_(nn.Conv2d(num_inputs, 32, kernel_size=5, stride=2)),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            init_(nn.Conv2d(32, 32, kernel_size=5, stride=2)),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            init_(nn.Conv2d(32, 32, kernel_size=4, stride=2)),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            #Print(),
            Flatten(),

            #nn.Dropout(0.2),

            init_(nn.Linear(32 * 7 * 5, hidden_size)),
            nn.ReLU()
        )

        init_ = lambda m: init(m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0))
        if SF:
            self.critic_linear = init_(nn.Linear(hidden_size + feature_size, feature_size*num_actions))
        else:

            self.critic_linear = init_(nn.Linear(hidden_size + feature_size, 1))
        self.train()

    def forward(self, inputs, rnn_hxs, masks, features = None):
        #print(inputs.size())

        x = inputs / 255.0
        #print(x.size())

        x = self.main(x)
        if self.feature_size > 0:

            x = torch.cat([x, features], axis = -1)
        #print(x.size())

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        return self.critic_linear(x), x, rnn_hxs


class MLPBase(NNBase):
    def __init__(self, num_inputs, feature_size = 2, recurrent=False, hidden_size=64, SF=False, num_actions = 3):
        super(MLPBase, self).__init__(recurrent, num_inputs + feature_size, hidden_size, feature_size = 2)
        self.SF = SF
        if recurrent:
            num_inputs = hidden_size

        init_ = lambda m: init(m,
            init_normc_,
            lambda x: nn.init.constant_(x, 0))

        self.actor = nn.Sequential(
            init_(nn.Linear(num_inputs + feature_size, hidden_size)),
            nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh()
        )

        self.critic = nn.Sequential(
            init_(nn.Linear(num_inputs + feature_size, hidden_size)),
            nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh()
        )

        if SF:

            self.critic_linear = init_(nn.Linear(hidden_size + feature_size, feature_size*num_actions))
        else:
            self.critic_linear = init_(nn.Linear(hidden_size + feature_size, 1))

        self.train()

    def forward(self, inputs, rnn_hxs, masks, features = None):
        x = inputs
        if self.feature_size > 0:
            x = torch.cat([inputs, features], axis = -1 )


        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)
        hidden_critic = self.critic(x)
        hidden_actor = self.actor(x)

        return self.critic_linear(hidden_critic), hidden_actor, rnn_hxs
