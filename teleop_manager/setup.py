import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'teleop_manager'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py'))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='home',
    maintainer_email='home@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'teleop_arm_data_node = teleop_manager.teleop_arm_data_node:main',
            'teleop_arm_node = teleop_manager.teleop_arm_node:main',
            'teleop_upper_node = teleop_manager.teleop_upper_node:main',
            'teleop_upper_igrisc_node = teleop_manager.teleop_upper_igrisc_node:main',
            'teleop_upper_igrisc_test_node = teleop_manager.teleop_upper_igrisc_test_node:main',
            'teleop_hand_igrisc_node = teleop_manager.teleop_hand_igrisc_node:main',
            'test_node = teleop_manager.test_node:main',
            'teleop_arm_igrisc_real_node = teleop_manager.teleop_arm_igrisc_real_node:main',
            'teleop_hand_igrisc_real_node = teleop_manager.teleop_hand_igrisc_real_node:main',
        ],
    },
)
