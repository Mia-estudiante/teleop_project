from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='simulation',
            executable='simulation_node',
            name='sim',
            output='screen'
        ),
        Node(
            package='teleop_manager',
            executable='teleop_arm_node_test',
            name='teleop_arm_test',
            output='screen'
        ),
        Node(
            package='user_manager',
            executable='user_node',
            name='user',
            output='screen'
        )
    ])