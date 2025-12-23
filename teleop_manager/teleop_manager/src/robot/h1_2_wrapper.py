import os

from rclpy.logging import get_logger

# pinocchio
import pinocchio as pin
from pinocchio.utils import *
from pinocchio import RobotWrapper

import numpy as np

from ament_index_python.packages import get_package_share_directory

class state():
    def __init__(self):
        self.q: np.array
        self.v: np.array
        # self.a: np.array
        # self.q_des: np.array
        # self.v_des: np.array
        # self.a_des: np.array
        # self.q_ref: np.array
        # self.v_ref: np.array

        # self.acc: np.array
        # self.tau: np.array
        # self.torque: np.array
        # self.v_input: np.array

        self.nq: np.array
        self.nv: np.array
        # self.na: np.array

        # self.id: np.array
        # self.G: np.array
        self.l_J: np.array
        self.r_J: np.array
        self.l_oMi: pin.SE3
        self.r_oMi: pin.SE3

class H12Wrapper(RobotWrapper):
    def __init__(self):
        self.logger = get_logger("H12Wrapper")
        self.logger.info("H12Wrapper has been started.")

        # URDF file
        descriptions_path = get_package_share_directory('h1_2_description')
        h12_urdf_path = os.path.join(descriptions_path, 'h1_2.urdf')

        self.__robot = RobotWrapper.BuildFromURDF(h12_urdf_path, package_dirs=[descriptions_path])

        # EEF
        self.l_eef = "left_wrist_yaw_joint"
        self.r_eef = "right_wrist_yaw_joint"

        # 팔(Arm)을 제외한 모든 고정 대상 관절 리스트
        self.joint2lock = [
                            # 하체 - 왼쪽 다리 (Left Leg)
                            "left_hip_yaw_joint",
                            "left_hip_pitch_joint",
                            "left_hip_roll_joint",
                            "left_knee_joint",
                            "left_ankle_pitch_joint",
                            "left_ankle_roll_joint",

                            # 하체 - 오른쪽 다리 (Right Leg)
                            "right_hip_yaw_joint",
                            "right_hip_pitch_joint",
                            "right_hip_roll_joint",
                            "right_knee_joint",
                            "right_ankle_pitch_joint",
                            "right_ankle_roll_joint",

                            # 허리 (Waist/Torso)
                            "torso_joint",

                            # 왼쪽 손가락 (Left Fingers/Thumb)
                            "L_thumb_proximal_yaw_joint",
                            "L_thumb_proximal_pitch_joint",
                            "L_thumb_intermediate_joint",
                            "L_thumb_distal_joint",
                            "L_index_proximal_joint",
                            "L_index_intermediate_joint",
                            "L_middle_proximal_joint",
                            "L_middle_intermediate_joint",
                            "L_ring_proximal_joint",
                            "L_ring_intermediate_joint",
                            "L_pinky_proximal_joint",
                            "L_pinky_intermediate_joint",

                            # 오른쪽 손가락 (Right Fingers/Thumb)
                            "R_thumb_proximal_yaw_joint",
                            "R_thumb_proximal_pitch_joint",
                            "R_thumb_intermediate_joint",
                            "R_thumb_distal_joint",
                            "R_index_proximal_joint",
                            "R_index_intermediate_joint",
                            "R_middle_proximal_joint",
                            "R_middle_intermediate_joint",
                            "R_ring_proximal_joint",
                            "R_ring_intermediate_joint",
                            "R_pinky_proximal_joint",
                            "R_pinky_intermediate_joint"
                        ]

        # 모든 값이 0이 아닌, Pinocchio가 권장하는 기본 자세(Quaternion 1 포함)를 가져옵니다.
        q_neutral = pin.neutral(self.__robot.model)
        # self.logger.info(f'Robot neutral configuration: {q_neutral}')
        # self.logger.info(f'{np.array([0.0] * self.__robot.model.nq)}')

        # 이 값을 reference_configuration에 전달합니다.
        self.robot_reduced = self.__robot.buildReducedRobot(
            list_of_joints_to_lock=self.joint2lock,
            reference_configuration=q_neutral,
        )

        # self.robot_reduced = self.__robot.buildReducedRobot(
        #     list_of_joints_to_lock=self.joint2lock,
        #     reference_configuration=np.array([0.0] * self.__robot.model.nq),
        # )
        
        self.model = self.robot_reduced.model
        self.data, self.__collision_data, self.__visual_data = \
            pin.createDatas(self.robot_reduced.model, self.robot_reduced.collision_model, self.robot_reduced.visual_model)

        self.state = state()
        self.state.nq = self.robot_reduced.nq
        self.state.nv = self.robot_reduced.nv
        self.state.q = zero(self.state.nq) # 14
        self.state.v = zero(self.state.nv)
        self.state.l_oMi = pin.SE3()
        self.state.r_oMi = pin.SE3()

    def computeAllTerms(self):
        pin.computeAllTerms(self.model, self.data, self.state.q, self.state.v)

        self.state.l_J = self.getJointJacobian(self.index(self.l_eef), pin.LOCAL)[:,:int(self.state.nq/2)]
        self.state.r_J = self.getJointJacobian(self.index(self.r_eef), pin.LOCAL)[:,int(self.state.nq/2):]

        self.state.l_oMi = self.data.oMi[self.index(self.l_eef)]
        self.state.r_oMi = self.data.oMi[self.index(self.r_eef)]


# if __name__ == "__main__":
#     robot = H12Wrapper()
