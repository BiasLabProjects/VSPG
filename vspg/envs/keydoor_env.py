"""KeyDoorEnv — a single-room environment with a key and locked door.

Layout (size=10):
  - Vertical wall separates left and right halves at x = wall_col
  - Locked door in the wall
  - Key placed in the left room
  - Goal in the bottom-right corner of the right room
  - Agent starts in the left room

The agent must: find the key → pick it up → unlock the door → reach the goal.
Significantly harder than the empty environment.
"""
from __future__ import annotations

import gymnasium as gym
from minigrid.core.constants import COLOR_NAMES
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door, Goal, Key, Wall
from minigrid.minigrid_env import MiniGridEnv


class KeyDoorEnv(MiniGridEnv):
    """One locked door, one key, one goal.

    Parameters
    ----------
    size : int
        Grid size (width = height = size).
    wall_col : int | None
        Column of the separating wall.  Defaults to size // 2.
    """

    def __init__(
        self,
        size: int = 10,
        wall_col: int | None = None,
        max_steps: int | None = None,
        **kwargs,
    ):
        self.size = size
        self.wall_col = wall_col if wall_col is not None else size // 2

        if max_steps is None:
            max_steps = 4 * size ** 2

        mission_space = MissionSpace(mission_func=self._gen_mission)
        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            max_steps=max_steps,
            **kwargs,
        )

    @staticmethod
    def _gen_mission() -> str:
        return "use the key to open the door and then get to the goal"

    def _gen_grid(self, width: int, height: int) -> None:
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        wc = self.wall_col
        for row in range(height):
            self.grid.set(wc, row, Wall())

        # Door and key — use first colour for both so they match
        color = COLOR_NAMES[0]
        door_row = height - 3
        self.grid.set(wc, door_row, Door(color, is_locked=True))
        self.grid.set(wc - 2, door_row, Key(color))

        # Goal in bottom-right of right room
        self.put_obj(Goal(), width - 2, height - 2)

        # Agent placed randomly in the left room
        self.place_agent(top=(1, 1), size=(wc - 1, height - 2))

        self.mission = self._gen_mission()
