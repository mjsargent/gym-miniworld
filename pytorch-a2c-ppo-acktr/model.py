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
    def __init__(self, obs_shape, action_space, feature_size, learnt_phi = False, eps = 0.05, base_kwargs=None):
        super(Policy, self).__init__()
        if base_kwargs is None:
            base_kwargs = {}

        self.phi_net = None
        self.eps = eps
        self.estimated_w = torch.randn(feature_size)

        if len(obs_shape) == 3:
            self.psi_net = CNNBase(obs_shape[0], feature_size, **base_kwargs)
            if learnt_phi = True:
                self.phi_net = CNNBase(obs_shape[0], feature_size = 0, **base_kwargs)

        elif len(obs_shape) == 1:
            self.base = MLPBase(obs_shape[0], feature_size, **base_kwargs)
            if learnt_phi = True:
                self.phi_net = CNNBase(obs_shape[0], feature_size = 0, **base_kwargs)
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
        # return q values
        # psi: NXB, |phi|
        psi, rnn_hxs = self.base(inputs, rnn_hxs, masks, features)
        with torch.no_grad:
            q = psi * self.estimated_w.unsqueeze(0).repeat(psi.shape[0], 1)

        if deterministic:
            r = torch.rand(psi.shape[0]).to(psi.device)
            action = torch.where(r < self.eps, torch.randint(q.shape[1], (psi.shape[0])).to(self.psi_net.device), torch.argmax(q, dim = -1))
        else:
            action = torch.argmax(q, dim = -1)

        return q, action, None, rnn_hxs

    def get_value(self, inputs, rnn_hxs, masks, features = None):
        # overloaded to mean q values
        q _, _ = self.base(inputs, rnn_hxs, masks, features)
        return q

    def evaluate_actions(self, inputs, rnn_hxs, masks, action, features = None):
        psi, rnn_hxs = self.base(inputs, rnn_hxs, masks, features)
        psi = psi.reshape(
        s = inputs[:-1]
        next_s = inputs[1:]

        psi = psi[:-1]
        next_psi = next_psi[1:].clone().detach()

        _phi = features[1:]

        # index with actions and best next actions
        psi  =
        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return psi_loss, phi_loss, w_loss, rnn_hxs



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
    def __init__(self, num_inputs, feature_size = 2, recurrent=False, hidden_size=128):
        super(CNNBase, self).__init__(recurrent, hidden_size + feature_size, hidden_size, feature_size)
        self.feature_size = feature_size

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
    def __init__(self, num_inputs, feature_size = 2, recurrent=False, hidden_size=64):
        super(MLPBase, self).__init__(recurrent, num_inputs + feature_size, hidden_size, feature_size = 2)

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
