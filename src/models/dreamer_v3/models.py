"""
Actor, Critic and World Models used for Dreamer V3 and Dreamer V3 + MLE regularisation.
This work is based on the Electric Sheep implementation of Dreamer V3 [1]

[1]: https://github.com/Eclectic-Sheep/sheeprl/tree/main/sheeprl/algos/dreamer_v3
"""

from typing import Optional, Type, List, Dict, Any, Tuple

import torch as th
import torch.nn.functional as F
from torch import nn
from torch.distributions.utils import probs_to_logits
from torch.distributions import (
    Distribution,
    Independent,
    Normal,
    TanhTransform,
    TransformedDistribution,
)
from torch.optim.adam import Adam as Adam
from torch.optim.optimizer import Optimizer as Optimizer

from src.models.dreamer_v3.distributions import TruncatedNormal
from src.models.dreamer_v3.utils import symlog, make_mlp, compute_stochastic_state


class Encoder(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        net_arch: List[int],
        layer_norm: bool = True,
        activation_fn: Type[nn.Module] = nn.SiLU,
        symlog_inputs: bool = True,
    ) -> None:
        super().__init__()
        self.symlog_inputs = symlog_inputs

        # create model
        self.model = make_mlp(
            [obs_dim] + net_arch,
            layer_norm=layer_norm,
            activation_fn=activation_fn,
        )

    def forward(self, obs: th.Tensor) -> th.Tensor:
        obs = symlog(obs) if self.symlog_inputs else obs
        return self.model(obs)


class Decoder(nn.Module):
    def __init__(
        self,
        output_dim: int,
        latent_state_size: int,
        net_arch: List[int],
        layer_norm: bool = True,
        activation_fn: Optional[Type[nn.Module]] = nn.SiLU,
    ) -> None:
        super().__init__()
        self.output_dim = output_dim

        # create model
        self.model = make_mlp(
            net_arch=[latent_state_size] + net_arch,
            layer_norm=layer_norm,
            activation_fn=activation_fn,
        )
        self.head = nn.Linear(net_arch[-1], output_dim)

    def forward(self, latent_state: th.Tensor) -> th.Tensor:
        features = self.model(latent_state)
        return self.head(features)


class LayerNormGRUCell(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        batch_first: bool = False,
        layer_norm: bool = False,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.batch_first = batch_first
        self.linear = nn.Linear(
            input_size + hidden_size, 3 * hidden_size, bias=self.bias
        )
        if layer_norm:
            self.layer_norm = th.nn.LayerNorm(3 * hidden_size)
        else:
            self.layer_norm = nn.Identity()

    def forward(
        self,
        sequence_input: th.Tensor,
        hidden_input: Optional[th.Tensor] = None,
    ) -> th.Tensor:
        is_3d = sequence_input.dim() == 3
        if is_3d:
            if sequence_input.shape[int(self.batch_first)] == 1:
                sequence_input = sequence_input.squeeze(int(self.batch_first))
            else:
                raise AssertionError(
                    "LayerNormGRUCell: Expected input to be 3-D with sequence length equal to 1 but received "
                    f"a sequence of length {sequence_input.shape[int(self.batch_first)]}"
                )
        if hidden_input.dim() == 3:
            hidden_input = hidden_input.squeeze(0)
        assert sequence_input.dim() in (
            1,
            2,
        ), f"LayerNormGRUCell: Expected input to be 1-D or 2-D but received {sequence_input.dim()}-D tensor"

        is_batched = sequence_input.dim() == 2
        if not is_batched:
            sequence_input = sequence_input.unsqueeze(0)

        if hidden_input is None:
            hidden_input = th.zeros(
                sequence_input.size(0),
                self.hidden_size,
                dtype=sequence_input.dtype,
                device=sequence_input.device,
            )
        else:
            hidden_input = hidden_input.unsqueeze(0) if not is_batched else hidden_input

        sequence_input = th.cat((hidden_input, sequence_input), -1)
        x = self.linear(sequence_input)
        x = self.layer_norm(x)
        reset, candidate, update = th.chunk(x, 3, -1)
        reset = th.sigmoid(reset)
        candidate = th.tanh(reset * candidate)
        update = th.sigmoid(update - 1)
        hidden_input = update * candidate + (1 - update) * hidden_input

        if not is_batched:
            hidden_input = hidden_input.squeeze(0)
        elif is_3d:
            hidden_input = hidden_input.unsqueeze(0)

        return hidden_input


class Recurrent_Model(nn.Module):
    def __init__(
        self,
        input_size: int,
        recurrent_state_size: int,
        n_hidden_units: int,
        activation_fn: nn.Module = nn.SiLU,
        layer_norm: bool = True,
    ) -> None:
        super().__init__()

        self.mlp = make_mlp(
            net_arch=[input_size, n_hidden_units],
            layer_norm=layer_norm,
            activation_fn=activation_fn,
        )
        self.rnn = LayerNormGRUCell(
            n_hidden_units,
            recurrent_state_size,
            bias=False,
            batch_first=False,
            layer_norm=True,
        )

    def forward(self, input: th.Tensor, recurrent_state: th.Tensor) -> th.Tensor:
        feat = self.mlp(input)
        return self.rnn(feat, recurrent_state)


class RSSM(nn.Module):
    def __init__(
        self,
        recurrent_model: nn.Module,
        representation_model: nn.Module,
        dynamics_predictor: nn.Module,
        discrete_rep_n_classes: int = 32,
        uni_mix: float = 0.01,
    ) -> None:
        super().__init__()
        self.recurrent_model = recurrent_model
        self.representation_model = representation_model
        self.dynamics_predictor = dynamics_predictor
        self.discrete_rep_n_classes = discrete_rep_n_classes
        self.uni_mix = uni_mix

    def dynamic(
        self,
        posterior: th.Tensor,
        recurrent_state: th.Tensor,
        action: th.Tensor,
        embedded_obs: th.Tensor,
        is_first: th.Tensor,
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor, th.Tensor]:

        # use action or zero action if first
        action = (1 - is_first) * action

        # use recurrent state (h1) or zero recurrent state if first
        recurrent_state = (1 - is_first) * recurrent_state + is_first * th.tanh(
            th.zeros_like(recurrent_state)
        )

        # use true posterior or sampled prior if first
        posterior = posterior.view(*posterior.shape[:-2], -1)
        posterior = (1 - is_first) * posterior + is_first * self._transition(
            recurrent_state, sample_state=False
        )[1].view_as(posterior)

        # estimate next recurrent state
        next_recurrent_state = self.recurrent_model(
            th.cat((posterior, action), -1), recurrent_state
        )

        # estimate next prior and logits
        next_prior_logits, next_prior = self._transition(next_recurrent_state)

        # estimate true next posterior and logits
        next_posterior_logits, next_posterior = self._representation(
            next_recurrent_state, embedded_obs
        )

        return (
            next_recurrent_state,
            next_posterior,
            next_prior,
            next_posterior_logits,
            next_prior_logits,
        )

    def _uniform_mix(self, logits: th.Tensor) -> th.Tensor:
        dim = logits.dim()
        if dim == 3:
            logits = logits.view(*logits.shape[:-1], -1, self.discrete_rep_n_classes)
        elif dim != 4:
            raise RuntimeError(
                f"The logits expected shape is 3 or 4: received a {dim}D tensor"
            )
        if self.uni_mix > 0.0:
            probs = logits.softmax(dim=-1)
            uniform = th.ones_like(probs) / self.discrete_rep_n_classes
            probs = (1 - self.uni_mix) * probs + self.uni_mix * uniform
            logits = probs_to_logits(probs)
        logits = logits.view(*logits.shape[:-2], -1)
        return logits

    def _representation(
        self,
        recurrent_state: th.Tensor,
        embedded_obs: th.Tensor,
        deterministic: bool = False,
    ) -> Tuple[th.Tensor, th.Tensor]:
        """
        Args:
            recurrent_state (Tensor): the recurrent state of the recurrent model, i.e.,
                what is called h or deterministic state in
                [https://arxiv.org/abs/1811.04551](https://arxiv.org/abs/1811.04551).
            embedded_obs (Tensor): the embedded real observations provided by the environment.

        Returns:
            logits (Tensor): the logits of the distribution of the posterior state.
            posterior (Tensor): the sampled posterior stochastic state.
        """
        logits: th.Tensor = self.representation_model(
            th.cat((recurrent_state, embedded_obs), -1)
        )
        logits = self._uniform_mix(logits)
        return logits, compute_stochastic_state(
            logits,
            discrete=self.discrete_rep_n_classes,
            sample=not deterministic,
        )

    def _transition(
        self, recurrent_state: th.Tensor, sample_state=True
    ) -> Tuple[th.Tensor, th.Tensor]:
        """
        Args:
            recurrent_out (Tensor): the output of the recurrent model, i.e., the deterministic part of the latent space.
            sampler_state (bool): whether or not to sample the stochastic state.
                Default to True

        Returns:
            logits (Tensor): the logits of the distribution of the prior state.
            prior (Tensor): the sampled prior stochastic state.
        """

        logits: th.Tensor = self.dynamics_predictor(recurrent_state)
        logits = self._uniform_mix(logits)
        return logits, compute_stochastic_state(
            logits,
            discrete=self.discrete_rep_n_classes,
            sample=sample_state,
        )

    def imagination(
        self, prior: th.Tensor, recurrent_state: th.Tensor, actions: th.Tensor
    ) -> Tuple[th.Tensor, th.Tensor]:
        """
        One-step imagination of the next latent state.
        It can be used several times to imagine trajectories in the latent space (Transition Model).

        Args:
            prior (Tensor): the prior state.
            recurrent_state (Tensor): the recurrent state of the recurrent model.
            actions (Tensor): the actions taken by the agent.

        Returns:
            The imagined prior state (Tuple[Tensor, Tensor]): the imagined prior state.
            The recurrent state (Tensor).
        """
        recurrent_state = self.recurrent_model(
            th.cat((prior, actions), -1), recurrent_state
        )
        _, imagined_prior = self._transition(recurrent_state)
        return imagined_prior, recurrent_state


class Actor(nn.Module):
    def __init__(
        self,
        latent_state_size: int,
        actions_dim: int,
        net_arch: List[int],
        distribution: str = "auto",
        init_std: float = 0.0,
        min_std: float = 0.1,
        activation_fn: nn.Module = nn.SiLU,
        layer_norm: bool = True,
        uni_mix: float = 0.01,
        explore_amount: float = 0.0,
    ) -> None:
        super().__init__()

        self.distribution = distribution
        if self.distribution not in (
            "auto",
            "normal",
            "tanh_normal",
            "trunc_normal",
        ):
            raise ValueError(
                "The distribution must be on of: `auto`, `discrete`, "
                "`normal`, `tanh_normal` and `trunc_normal`. "
                f"Found: {self.distribution}"
            )
        if self.distribution == "auto":
            self.distribution = "trunc_normal"

        # set up actor network
        self.model = make_mlp(
            [latent_state_size] + net_arch,
            layer_norm=layer_norm,
            activation_fn=activation_fn,
        )
        self.head = nn.Linear(net_arch[-1], actions_dim * 2)

        self.actions_dim = actions_dim
        self.init_std = th.tensor(init_std)
        self.min_std = min_std
        self._uni_mix = uni_mix
        self._explore_amount = explore_amount

    @property
    def explore_amount(self) -> float:
        return self._explore_amount

    @explore_amount.setter
    def explore_amount(self, explore_amount: float):
        self._explore_amount = explore_amount

    def forward(
        self,
        latent_sate: th.Tensor,
        mask: Optional[Dict[str, th.Tensor]] = None,
        deterministic: bool = False,
    ) -> Tuple[th.Tensor, Distribution]:

        # get mean and standard deviation
        features: th.Tensor = self.model(latent_sate)
        mean_and_std = self.head(features)
        mean, std = th.chunk(mean_and_std, 2, -1)

        # get action distribution
        if self.distribution == "tanh_normal":
            mean = 5 * th.tanh(mean / 5)
            std = F.softplus(std + self.init_std) + self.min_std
            actions_dist = Normal(mean, std)
            actions_dist = Independent(
                TransformedDistribution(
                    actions_dist,
                    TanhTransform(),
                    validate_args=False,
                ),
                1,
                validate_args=False,
            )
        elif self.distribution == "normal":
            actions_dist = Normal(mean, std, validate_args=False)
            actions_dist = Independent(actions_dist, 1, validate_args=False)
        elif self.distribution == "trunc_normal":
            std = 2 * th.sigmoid((std + self.init_std) / 2) + self.min_std
            dist = TruncatedNormal(th.tanh(mean), std, -1, 1, validate_args=False)
            actions_dist = Independent(dist, 1, validate_args=False)

        # get sampled action
        if deterministic:
            actions = actions_dist.mean
        elif self.training:
            actions = actions_dist.rsample()
        else:
            actions = actions_dist.sample()
            # sample = actions_dist.sample((100,))
            # log_prob: th.Tensor = actions_dist.log_prob(sample)
            # actions = sample[log_prob.argmax(0)].view(1, 1, -1)

        return actions, actions_dist

    def add_exploration_noise(
        self, actions: th.Tensor, mask: Optional[Dict[str, th.Tensor]] = None
    ) -> th.Tensor:
        if self._explore_amount > 0.0:
            return th.clip(Normal(actions, self._explore_amount).sample(), -1, 1)

        return actions


class World_Model(nn.Module):
    def __init__(
        self,
        encoder: Encoder,
        rssm: RSSM,
        decoder: Decoder,
        reward_model: nn.Module,
        continue_model: nn.Module,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.rssm = rssm
        self.decoder = decoder
        self.reward_model = reward_model
        self.continue_model = continue_model


class Moments(nn.Module):
    def __init__(
        self,
        decay: float = 0.99,
        max_: float = 1e8,
        percentile_low: float = 0.05,
        percentile_high: float = 0.95,
    ) -> None:
        super().__init__()
        self._decay = decay
        self._max = th.tensor(max_)
        self._percentile_low = percentile_low
        self._percentile_high = percentile_high
        self.register_buffer("low", th.zeros((), dtype=th.float32))
        self.register_buffer("high", th.zeros((), dtype=th.float32))

    def forward(self, lambda_values: th.Tensor) -> Any:
        with th.no_grad():
            low = th.quantile(lambda_values.detach(), self._percentile_low)
            high = th.quantile(lambda_values.detach(), self._percentile_high)
            self.low = self._decay * self.low + (1 - self._decay) * low
            self.high = self._decay * self.high + (1 - self._decay) * high
            invscale = th.max(1 / self._max, self.high - self.low)
            return self.low.detach(), invscale.detach()
