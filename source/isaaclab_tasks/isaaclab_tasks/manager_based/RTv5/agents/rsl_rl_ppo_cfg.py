from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class RTv5RoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 96
    max_iterations = 3000
    save_interval = 50
    experiment_name = "RTv5_rough"
    # Tier-3 #14: bake observation normalization stats into the exported policy
    # so the deployed network receives well-scaled inputs even when the obs
    # distribution at training time isn't exactly zero-mean unit-var. The
    # running mean/std are exported as constants in the ONNX graph (rsl_rl
    # handles this), so the deploy code just feeds raw obs as during training.
    # CRITICAL: at deployment, replicate every analytic pre-scaling step from
    # this env exactly (joint pos, ang vel units, gravity-bias convention).
    # Empirical normalization corrects only the residual scale/bias; it cannot
    # rescue a unit-mismatch or axis-flip.
    empirical_normalization = True
    # fp16 = True 
    obs_groups = {"policy": ["policy"], "critic": ["policy"]}
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        # actor_hidden_dims=[150, 75, 30],
        # critic_hidden_dims=[150, 75, 30],
        actor_obs_normalization=True,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        # noise_std_type="log",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=10,
        num_mini_batches=8,
        # learning_rate=1.0e-4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.995,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class RTv5FlatPPORunnerCfg(RTv5RoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__() # type: ignore
        self.experiment_name = "RTv5_flat"



@configclass
class RTv5TestPPORunnerCfg(RTv5RoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__() # type: ignore
        self.experiment_name = "RTv5_test"
        # self.policy.init_noise_std = 0.001
        self.policy = RslRlPpoActorCriticCfg(
            init_noise_std=1.0,
            actor_obs_normalization=False,
            critic_obs_normalization=False, 
            actor_hidden_dims=[14],
            critic_hidden_dims=[14],
            activation="elu",
        )
