import gymnasium as gym

from . import agents
from .turtlebot3_env_cfg import TurtleBot3EnvCfg

gym.register(
    id="Isaac-TurtleBot3-Straight-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": TurtleBot3EnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:TurtleBot3PPORunnerCfg",
    },
)