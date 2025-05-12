import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
import os
import numpy as np
# 1.17, -np.pi/2, np.pi/2, -np.pi/2, -np.pi/2, np.pi/2
##
# Configuration
##

script_dir = os.path.dirname(os.path.abspath(__file__))
ur5e_usd_path = os.path.join(script_dir, "asset/ur_square_fingers.usd")

UR5e_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        # usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/UniversalRobots/UR10/ur10_instanceable.usd",
        usd_path=ur5e_usd_path,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        activate_contact_sensors=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "shoulder_pan_joint": 1.17,
            "shoulder_lift_joint": -np.pi/2,
            "elbow_joint": np.pi/2,
            "wrist_1_joint": -np.pi/2,
            "wrist_2_joint": -np.pi/2,
            "wrist_3_joint": np.pi/2,
            "Slider_1": 0.025,
            "Slider_2": 0.025,
        },
    ),
    # "Slider_*": 0.02,
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            velocity_limit_sim=20.0,
            effort_limit_sim=87.0,
            stiffness=800.0,
            damping=400.0,
        ),
    },
)
"""Configuration of UR-10 arm using implicit actuator models."""
