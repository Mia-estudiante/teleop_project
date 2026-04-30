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

        # self.G: np.array
        self.l_J: np.array
        self.r_J: np.array
        
        self.head_oMi: pin.SE3
        self.l_oMi: pin.SE3
        self.r_oMi: pin.SE3

class IgrisCWrapper(RobotWrapper):
    def __init__(self):
        self.logger = get_logger("IgrisCWrapper")
        self.logger.info("IgrisCWrapper has been started.")

        # URDF file
        descriptions_path = get_package_share_directory('igris_c_description')
        igris_c_urdf_path = os.path.join(descriptions_path, 'urdf/igris_c_v2_hand.urdf')

        self.__robot = RobotWrapper.BuildFromURDF(igris_c_urdf_path, package_dirs=[descriptions_path])

        # EEF
        self.l_eef = "21_Joint_Wrist_Pitch_Left"
        self.r_eef = "28_Joint_Wrist_Pitch_Right"

        # 팔(Arm) 조인트를 제외한 모든 조인트를 고정하는 리스트
        self.joint2lock = [
            # --- 허리 (Waist) ---
            "0_Joint_Waist_Yaw",
            "1_Joint_Waist_Roll",
            "2_Joint_Waist_Pitch",

            # --- 왼쪽 다리 (Left Leg) ---
            "3_Joint_Hip_Pitch_Left",
            "4_Joint_Hip_Roll_Left",
            "5_Joint_Hip_Yaw_Left",
            "6_Joint_Knee_Pitch_Left",
            "7_Joint_Ankle_Pitch_Left",
            "8_Joint_Ankle_Roll_Left",

            # --- 오른쪽 다리 (Right Leg) ---
            "9_Joint_Hip_Pitch_Right",
            "10_Joint_Hip_Roll_Right",
            "11_Joint_Hip_Yaw_Right",
            "12_Joint_Knee_Pitch_Right",
            "13_Joint_Ankle_Pitch_Right",
            "14_Joint_Ankle_Roll_Right",

            # --- 머리 (Head) ---
            "29_Joint_Neck_Yaw",
            "30_Joint_Neck_Pitch",

            # --- 왼쪽 손가락 (Left Hand Fingers) ---
            "Left_0_Joint_Thumb_Proximal",
            "Left_1_Joint_Thumb_Middle",
            "Left_2_Joint_Thumb_Distal",
            "Left_3_Joint_Index_Middle",
            "Left_4_Joint_Index_Distal",
            "Left_5_Joint_Middle_Middle",
            "Left_6_Joint_Middle_Distal",
            "Left_7_Joint_Ring_Middle",
            "Left_8_Joint_Ring_Distal",
            "Left_9_Joint_Little_Middle",
            "Left_10_Joint_Little_Distal",

            # --- 오른쪽 손가락 (Right Hand Fingers) ---
            "Right_0_Joint_Thumb_Proximal",
            "Right_1_Joint_Thumb_Middle",
            "Right_2_Joint_Thumb_Distal",
            "Right_3_Joint_Index_Middle",
            "Right_4_Joint_Index_Distal",
            "Right_5_Joint_Middle_Middle",
            "Right_6_Joint_Middle_Distal",
            "Right_7_Joint_Ring_Middle",
            "Right_8_Joint_Ring_Distal",
            "Right_9_Joint_Little_Middle",
            "Right_10_Joint_Little_Distal"
        ]

        # 모든 값이 0이 아닌, Pinocchio가 권장하는 기본 자세(Quaternion 1 포함)를 가져옵니다.
        q_neutral = pin.neutral(self.__robot.model)
        self.logger.info(f'Robot neutral configuration: {q_neutral}')

        # 이 값을 reference_configuration에 전달합니다.
        self.robot_reduced = self.__robot.buildReducedRobot(
            list_of_joints_to_lock=self.joint2lock,
            reference_configuration=q_neutral, # np.array([0.0] * self.__robot.model.nq)
        )
        
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

        self.all_joint_names = [self.__robot.model.names[i] for i in range(self.__robot.model.njoints)]
        print(self.all_joint_names)
        self.joint_names = [self.model.names[i] for i in range(self.model.njoints) if self.model.names[i] not in self.joint2lock]
        # # 전체 관절 이름과 부모 ID를 리스트로 출력
        # for i, (name, parent) in enumerate(zip(self.model.names, self.model.parents)):
        #     print(f"ID: {i:2} | Joint: {name:20} | Parent ID: {parent}")
        for i in range(1, self.model.njoints):
            name = self.model.names[i]
            joint = self.model.joints[i]
            
            start_idx = joint.idx_q
            num_q = joint.nq
            
            # 해당 조인트가 q 벡터에서 차지하는 범위를 출력
            print(f"{name:<30} | {start_idx:<10} | {num_q}")

    def computeAllTerms(self):
        pin.computeAllTerms(self.model, self.data, self.state.q, self.state.v)

        self.state.l_J = self.getJointJacobian(self.index(self.l_eef), pin.LOCAL)[:,:int(self.state.nq/2)]
        self.state.r_J = self.getJointJacobian(self.index(self.r_eef), pin.LOCAL)[:,int(self.state.nq/2):]

        self.state.l_oMi = self.data.oMi[self.index(self.l_eef)]
        self.state.r_oMi = self.data.oMi[self.index(self.r_eef)]
        self.state.head_oMi = pin.SE3(np.eye(3), np.array([0, 0, 0.357])) # pelvis 기준(fixed) - pelvisMhead