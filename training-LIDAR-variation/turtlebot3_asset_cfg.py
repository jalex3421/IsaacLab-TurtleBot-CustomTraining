from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
import isaaclab.sim as sim_utils

TURTLEBOT3_ASSET_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="/home/owner/isaac/Myprojects/turtlebot.usd",
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0), # Default spawn height
        joint_pos={".*": 0.0},
    ),
    actuators={
        "wheels": ImplicitActuatorCfg(
            # Exact joint names from the robotis-git TurtleBot3 URDF
            joint_names_expr=["a__namespace_wheel_left_joint", "a__namespace_wheel_right_joint"],
            effort_limit=10.0,
            velocity_limit=5.0,
            stiffness=0.0,
            damping=100.0,   # High damping stabilizes PhysX velocity targets
        ),
    },
)