"""Custom MiniGrid environments.

Import this module to register all custom envs with gymnasium:
    from vspg import envs  # noqa: F401
"""
from .keydoor_env import KeyDoorEnv  # noqa: F401
from .register import register_all

register_all()
