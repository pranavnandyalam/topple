"""Gymnasium registration: ``gym.make("topple.envs:topple/Balance-v1")``.

Kept out of ``topple/__init__`` so importing the contract stays lightweight.
"""

import gymnasium as gym

from topple.envs.balance import BalanceEnv, RewardConfig
from topple.envs.waypoint import WaypointEnv, WaypointRewardConfig
from topple.ids import gym_id

BALANCE_V1 = gym_id("balance-v1")
WAYPOINT_V1 = gym_id("waypoint-v1")

if BALANCE_V1 not in gym.registry:
    gym.register(id=BALANCE_V1, entry_point="topple.envs.balance:BalanceEnv")
if WAYPOINT_V1 not in gym.registry:
    gym.register(id=WAYPOINT_V1, entry_point="topple.envs.waypoint:WaypointEnv")

__all__ = [
    "BALANCE_V1",
    "WAYPOINT_V1",
    "BalanceEnv",
    "RewardConfig",
    "WaypointEnv",
    "WaypointRewardConfig",
]
