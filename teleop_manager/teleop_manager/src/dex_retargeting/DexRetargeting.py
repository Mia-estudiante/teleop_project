import os
import numpy as np

# dex_retargeting
from dex_retargeting.retargeting_config import RetargetingConfig

from ament_index_python.packages import get_package_share_directory

class DexRetargeting:
    def __init__(self):
        # URDF file
        descriptions_path = get_package_share_directory('h1_2_description')
        RetargetingConfig.set_default_urdf_dir(str(descriptions_path))
        left_yml_path = os.path.join(descriptions_path, 'inspire_hand/inspire_hand_left.yml')
        right_yml_path = os.path.join(descriptions_path, 'inspire_hand/inspire_hand_right.yml')

        self.left_retargeting = RetargetingConfig.load_from_file(left_yml_path).build()
        self.right_retargeting = RetargetingConfig.load_from_file(right_yml_path).build()

        self.left_retargeting_joint_names = self.left_retargeting.joint_names
        self.right_retargeting_joint_names = self.right_retargeting.joint_names
        
        self.left_indices = self.left_retargeting.optimizer.target_link_human_indices
        self.right_indices = self.right_retargeting.optimizer.target_link_human_indices

        # 12개
        # ['index_proximal_joint', 'index_intermediate_joint', 'middle_proximal_joint', 'middle_intermediate_joint', 'pinky_proximal_joint', 'pinky_intermediate_joint', 
        # 'ring_proximal_joint', 'ring_intermediate_joint', 'thumb_proximal_yaw_joint', 'thumb_proximal_pitch_joint', 'thumb_intermediate_joint', 'thumb_distal_joint']
        # print("Left retargeting joint names:", self.left_retargeting_joint_names)
        # print("Right retargeting joint names:", self.right_retargeting_joint_names)
        # print("Left retargeting indices:", self.left_indices)
        # print("Right retargeting indices:", self.right_indices)

    def retarget(self, retargeting, indices, joint_pos: np.ndarray) -> np.ndarray:
        origin_indices = indices[0, :]  # [ 0, 0, 0, 0, 0 ],
        task_indices = indices[1, :]    # [ 4, 8, 12, 16, 20 ] 
        ref_value = joint_pos[task_indices, :] - joint_pos[origin_indices, :]
        qpos = retargeting.retarget(ref_value)
        return qpos
    
if __name__ == '__main__':
    dex_retargeting = DexRetargeting()