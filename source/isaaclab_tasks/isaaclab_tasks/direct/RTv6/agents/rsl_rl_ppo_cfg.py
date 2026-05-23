# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class RTv6RoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 20
    max_iterations = 3000
    save_interval = 50
    experiment_name = "RTv6_rough_direct"
    empirical_normalization = True
    obs_groups = {"policy": ["policy"], "critic": ["policy"]}
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=True,
        critic_obs_normalization=False,
        actor_hidden_dims=[4],
        critic_hidden_dims=[4],
        activation="identity",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=10,
        num_mini_batches=8,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.995,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class RTv6FlatPPORunnerCfg(RTv6RoughPPORunnerCfg):
    def __post_init__(self) -> None:
        super().__post_init__()  # type: ignore
        self.experiment_name = "RTv6_flat_direct"
