"""PPO configuration for the K1 velocity examples (example3-example10)."""

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)

# Every course run logs to one W&B project; experiment_name tells examples apart.
WANDB_PROJECT = "eir_course"


def k1_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """PPO runner for the Booster K1 velocity task.

  Same network and hyperparameters as Pumas-Velocity-*-Booster-K1-PPO in
  mjlab-biped-playground (which match mjlab's upstream G1 velocity task).
  """
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.01,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="k1_velocity",
    wandb_project="k1_velocity",
    save_interval=1_000,
    num_steps_per_env=24,
    max_iterations=6_000,
  )


def course_ppo_runner_cfg(example: int, max_iterations: int) -> RslRlOnPolicyRunnerCfg:
  """K1 PPO runner logging to the course W&B project as ``example<N>``."""
  cfg = k1_ppo_runner_cfg()
  cfg.experiment_name = f"example{example}"
  cfg.wandb_project = WANDB_PROJECT
  cfg.max_iterations = max_iterations
  cfg.save_interval = 500
  return cfg
