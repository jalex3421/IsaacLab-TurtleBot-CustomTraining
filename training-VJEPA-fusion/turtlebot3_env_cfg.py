from __future__ import annotations

import torch

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import RayCaster, RayCasterCfg, TiledCameraCfg, patterns
from isaaclab.terrains import TerrainGeneratorCfg, TerrainImporterCfg
from isaaclab.utils import configclass

from .turtlebot3_asset_cfg import TURTLEBOT3_ASSET_CFG
from .turtlebot3_maze_terrain import MazeTerrainCfg

WHEEL_JOINT_NAMES = ["a__namespace_wheel_left_joint", "a__namespace_wheel_right_joint"]

# Carves a perfect (single-solution) maze into each environment's sub-terrain cell instead of scattering
# random boxes: one maze per environment (8x8 = 64, matching the default num_envs), combined into a single
# static mesh, since RayCasterCfg only supports raycasting against one shared mesh prim. The robot spawns
# in the maze's center cell, which a perfect maze guarantees can reach every corridor.
MAZE_TERRAIN_CFG = TerrainGeneratorCfg(
    size=(4.0, 4.0),
    num_rows=8,
    num_cols=8,
    border_width=0.0,
    sub_terrains={
        "maze": MazeTerrainCfg(
            proportion=1.0,
            num_cells_range=(3, 5),
            wall_height=0.5,
            wall_thickness=0.05,
        )
    },
)


def track_forward_vel(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Reward forward velocity in the robot's body frame."""
    asset: Articulation = env.scene[asset_cfg.name]
    # Clamped to a generous multiple of the robot's nominal top speed: obstacle collisions can otherwise
    # inject brief, unrealistic velocity spikes that dominate the PPO value estimate and destabilize training.
    return torch.clamp(asset.data.root_lin_vel_b[:, 0], -2.0, 2.0)


def penalize_spinning(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize angular velocity about the vertical axis."""
    asset: Articulation = env.scene[asset_cfg.name]
    # Same reasoning as track_forward_vel: clamp before squaring so a collision-induced angular velocity
    # spike can't blow up this term (and the value function along with it).
    ang_vel = torch.clamp(asset.data.root_ang_vel_b[:, 2], -10.0, 10.0)
    return torch.square(ang_vel)


def lidar_scan(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg = SceneEntityCfg("lidar")) -> torch.Tensor:
    """Per-beam distance from the lidar to the nearest hit, clamped to the sensor's max range."""
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    distances = torch.linalg.norm(sensor.data.ray_hits_w - sensor.data.pos_w.unsqueeze(1), dim=-1)
    return torch.nan_to_num(distances, nan=sensor.cfg.max_distance, posinf=sensor.cfg.max_distance)


def min_lidar_distance(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg = SceneEntityCfg("lidar")) -> torch.Tensor:
    """Distance from the lidar to the nearest obstacle, rewarding the robot for keeping its distance."""
    return torch.amin(lidar_scan(env, sensor_cfg), dim=-1)


def collision(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg = SceneEntityCfg("lidar"), threshold: float = 0.15
) -> torch.Tensor:
    """Terminate when the lidar detects an obstacle closer than the robot's footprint radius plus margin."""
    return torch.any(lidar_scan(env, sensor_cfg) < threshold, dim=-1)


@configclass
class TurtleBot3SceneCfg(InteractiveSceneCfg):
    """Configuration for the turtlebot3 scene."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=MAZE_TERRAIN_CFG,
        collision_group=-1,
        visual_material=None,
        debug_vis=False,
    )

    robot: Articulation = TURTLEBOT3_ASSET_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    lidar = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/turtlebot3_burger/a__namespace_base_scan",
        ray_alignment="yaw",
        pattern_cfg=patterns.LidarPatternCfg(
            channels=1,
            vertical_fov_range=(0.0, 0.0),
            horizontal_fov_range=(-180.0, 180.0),
            horizontal_res=10.0,
        ),
        mesh_prim_paths=["/World/ground"],
        # Bounded by half the 4.0m terrain cell size: sub-terrain cells have no walls between them, so a
        # longer range risks picking up a neighboring environment's maze.
        max_distance=2.0,
        debug_vis=False,
    )

    # Forward-facing RGB camera feeding the V-JEPA-style vision backbone. Mounted on the base link (the
    # burger USD has no dedicated camera frame) pointing along the robot's local +X (forward) with +Z up,
    # which is exactly the "world" offset convention's axis mapping.
    front_camera = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/turtlebot3_burger/a__namespace_base_link/front_cam",
        offset=TiledCameraCfg.OffsetCfg(pos=(0.05, 0.0, 0.09), rot=(1.0, 0.0, 0.0, 0.0), convention="world"),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=8.0, focus_distance=2.0, horizontal_aperture=16.0, clipping_range=(0.05, 4.0)
        ),
        # Kept small: this feeds a lightweight per-step vision encoder running on 64 parallel envs, not an
        # offline vision model, so resolution is traded for real-time throughput.
        width=64,
        height=64,
    )

    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    wheel_velocities = mdp.JointVelocityActionCfg(asset_name="robot", joint_names=WHEEL_JOINT_NAMES, scale=5.0)


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        lidar_scan = ObsTerm(func=lidar_scan, params={"sensor_cfg": SceneEntityCfg("lidar")})

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class ImageCfg(ObsGroup):
        """Raw camera frames, kept as a separate group so the CNN/ViT policy branch never gets flattened
        together with the proprioceptive vector."""

        camera_rgb = ObsTerm(
            func=mdp.image, params={"sensor_cfg": SceneEntityCfg("front_camera"), "data_type": "rgb"}
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    image: ImageCfg = ImageCfg()


@configclass
class EventsCfg:
    """Event terms for the MDP."""

    # Required whenever the scene uses a TerrainImporterCfg: per-env spawn origins come from the terrain's
    # sub-cell grid (env.scene.env_origins), not the default env-spacing grid, and only an explicit reset
    # event applies that offset to the robot's root state.
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.2, 0.2), "y": (-0.2, 0.2), "yaw": (-3.14, 3.14)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    track_forward_vel = RewTerm(func=track_forward_vel, weight=1.0)
    penalize_spinning = RewTerm(func=penalize_spinning, weight=-0.1)
    keep_clearance = RewTerm(func=min_lidar_distance, weight=0.1)
    collision_penalty = RewTerm(func=mdp.is_terminated_term, weight=-10.0, params={"term_keys": "collision"})


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    collision = DoneTerm(func=collision, params={"sensor_cfg": SceneEntityCfg("lidar"), "threshold": 0.15})


@configclass
class TurtleBot3EnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the turtlebot3 maze-navigation environment."""

    scene: TurtleBot3SceneCfg = TurtleBot3SceneCfg(num_envs=64, env_spacing=4.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventsCfg = EventsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self) -> None:
        self.decimation = 2
        self.episode_length_s = 10.0
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation
        self.scene.lidar.update_period = self.decimation * self.sim.dt
        self.scene.front_camera.update_period = self.decimation * self.sim.dt