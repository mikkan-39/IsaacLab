from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class RTv5RoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 32
    max_iterations = 3000
    save_interval = 50
    experiment_name = "RTv4_rough"
    empirical_normalization = True
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        # actor_hidden_dims=[150, 75, 30],
        # critic_hidden_dims=[150, 75, 30],
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.008,
        # entropy_coef=0.002,
        num_learning_epochs=10,
        num_mini_batches=8,
        # learning_rate=1.0e-4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class RTv5FlatPPORunnerCfg(RTv5RoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__() # type: ignore
        self.experiment_name = "RTv4_flat"
