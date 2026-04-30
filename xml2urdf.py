import mujoco
import os
from ament_index_python.packages import get_package_share_directory

urdf_path = os.path.join(get_package_share_directory('igris_c_description'), 'urdf', 'igris_c_v2_hand_.urdf')
print(f"urdf_path: {urdf_path}")

# urdf_path = '/home/home/mujoco_ws/src/igris_c_description/urdf/igris_c_v2_hand_.urdf'
model = mujoco.MjModel.from_xml_path(urdf_path)
mujoco.mj_saveLastXML('hand_converted.xml', model)

# from dm_control import mjcf

# # URDF 로드 후 MJCF로 변환
# hand_model = mjcf.from_path(urdf_path)
# hand_model.save('hand_converted.xml')