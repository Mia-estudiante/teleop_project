import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'user_manager'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'data'),
            glob('user_manager/src/data/*.json'))
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
            'user_data_node = user_manager.user_data_node:main',
            'avp_hand_node = user_manager.avp_hand_node:main',
            'avp_igrisc_node = user_manager.avp_igrisc_node:main',
            'avp_h12_node = user_manager.avp_h12_node:main',
            'avp_node_ = user_manager.avp_node_:main',
            'hand_replay_node = user_manager.hand_replay_node:main',
            'hand_vis_node = user_manager.hand_vis_node:main',
        ],
    },
)
