import os
from launch import LaunchDescription
from launch_ros.actions import Node

from ament_index_python.packages import get_package_share_directory

'''
TF 확인용
'''

def generate_launch_description():
    # 1. URDF 파일 경로 설정
    urdf_path = os.path.join(get_package_share_directory('h1_2_description'), 'h1_2.urdf')

    # 2. URDF 파일 읽기
    with open(urdf_path, 'r') as infp:
        robot_desc = infp.read()

    return LaunchDescription([
        # Robot State Publisher: URDF를 기반으로 /tf 메시지 발행
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            output='screen',
            parameters=[{'robot_description': robot_desc}]
        ),

        # Joint State Publisher GUI: 관절 조절 슬라이더 UI
        Node(
            package='joint_state_publisher_gui',
            executable='joint_state_publisher_gui',
            name='joint_state_publisher_gui',
            output='screen'
        ),

        # RViz2: 3D 시각화 도구
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='screen',
            # 필요한 경우 -d 옵션으로 rviz 설정 파일을 지정할 수 있습니다.
            # arguments=['-d', rviz_config_path] 
        ),
    ])