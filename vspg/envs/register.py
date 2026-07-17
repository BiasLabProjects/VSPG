"""Register custom environments with gymnasium."""
import gymnasium as gym


def register_all() -> None:
    ids = {e for e in gym.envs.registry}

    specs = [
        ("HDPolicy-KeyDoor-10x10-v0", "vspg.envs.keydoor_env:KeyDoorEnv",
         {"size": 10}),
        ("HDPolicy-KeyDoor-8x8-v0",  "vspg.envs.keydoor_env:KeyDoorEnv",
         {"size": 8}),
    ]

    for env_id, entry, kwargs in specs:
        if env_id not in ids:
            gym.register(id=env_id, entry_point=entry, kwargs=kwargs)
