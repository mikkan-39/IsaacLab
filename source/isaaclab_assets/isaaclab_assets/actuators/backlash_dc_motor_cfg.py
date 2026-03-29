from isaaclab.actuators.actuator_pd_cfg import DCMotorCfg
from isaaclab.utils import configclass

from . import backlash_dc_motor


@configclass
class BacklashDCMotorCfg(DCMotorCfg):
    """Configuration for a DC motor with command delay, gear backlash, and noise.

    See :class:`BacklashDCMotor` for the actuator model description.
    """

    class_type: type = backlash_dc_motor.BacklashDCMotor

    min_delay: int = 1
    """Minimum number of physics time-steps of command delay. Defaults to 1."""

    max_delay: int = 4
    """Maximum number of physics time-steps of command delay. Defaults to 4."""

    backlash_rad: float = 0.028
    """Total gear backlash dead-zone width in radians (~1.6 deg). Defaults to 0.028."""

    noise_std: float = 0.009
    """Standard deviation of additive Gaussian noise on the effective position target (~0.5 deg). Defaults to 0.009."""
