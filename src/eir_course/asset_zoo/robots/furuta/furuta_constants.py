"""Furuta pendulum entity, driven by the Booster K1 knee motor."""

from pathlib import Path

import mujoco
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

from eir_course.asset_zoo.robots.k1.actuators import MotorPositionActuatorCfg
from eir_course.asset_zoo.robots.k1.k1_constants import ACTUATOR_E6416

FURUTA_XML: Path = Path(__file__).parent / "furuta.xml"
assert FURUTA_XML.exists()

# The K1 knee pitch motor: kp = 80 N*m/rad, kd = 4 N*m*s/rad, +-112 N*m and
# 0.095625 kg*m^2 of armature. Same motor config as K1_ACTUATOR_E6416, without
# the random actuation delay: a tabular policy has no memory to cope with it.
FURUTA_ARM_ACTUATOR = MotorPositionActuatorCfg(
  motor=ACTUATOR_E6416,
  target_names_expr=("arm",),
  effort_limit=ACTUATOR_E6416.effort_limit,
  armature=ACTUATOR_E6416.armature,
  stiffness=ACTUATOR_E6416.stiffness,
  damping=ACTUATOR_E6416.damping,
)

# Same rule as every K1 joint: 0.25 * effort_limit / stiffness = 0.35 rad.
FURUTA_ACTION_SCALE = 0.25 * ACTUATOR_E6416.effort_limit / ACTUATOR_E6416.stiffness


def get_spec() -> mujoco.MjSpec:
  return mujoco.MjSpec.from_file(str(FURUTA_XML))


def get_furuta_cfg(pendulum_angle: float = 0.0) -> EntityCfg:
  """Furuta pendulum entity, starting with the pendulum at ``pendulum_angle`` (0 = up)."""
  return EntityCfg(
    spec_fn=get_spec,
    articulation=EntityArticulationInfoCfg(actuators=(FURUTA_ARM_ACTUATOR,)),
    init_state=EntityCfg.InitialStateCfg(
      pos=(0.0, 0.0, 0.0),
      joint_pos={"arm": 0.0, "pendulum": pendulum_angle},
      joint_vel={".*": 0.0},
    ),
  )
