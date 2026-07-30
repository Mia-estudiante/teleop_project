from setuptools import find_packages, setup

package_name = 'simulation'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='home',
    maintainer_email='home@todo.todo',
    description='TODO: Package description',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'simulation_node = simulation.simulation_node:main',
            'simulation_hand_node = simulation.simulation_hand_node:main',
            'simulation_upper_node = simulation.simulation_upper_node:main',
            'simulation_igrisc_node = simulation.simulation_igrisc_node:main',
            'simulation_igrisc_test_node = simulation.simulation_igrisc_test_node:main',
            'simulation_igrisc_urdf_node = simulation.simulation_igrisc_urdf_node:main',
            'simulation_igrisc_cube_node = simulation.simulation_igrisc_cube_node:main',
            'simulation_igrisc_hand_node = simulation.simulation_igrisc_hand_node:main',
            'simulation_test = simulation.simulation_test:main',
            'simulation_test_igris = simulation.simulation_test_igris:main'
        ],
    },
)
