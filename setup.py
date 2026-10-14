from setuptools import find_packages, setup
from glob import glob
import os

package_name = 'vision_pkg'

setup(
    name=package_name,
    version='0.0.0',

    packages=find_packages(exclude=['test']),

    data_files=[
        (
            'share/ament_index/resource_index/packages',
            ['resource/' + package_name]
        ),
        (
            'share/' + package_name,
            ['package.xml']
        ),
        (
            os.path.join('share', package_name, 'urdf'),
            glob('urdf/*.urdf')
        ),
        (
            os.path.join('share', package_name, 'meshes'),
            glob('meshes/*') + glob('mashes/*')
        ),
        (
            os.path.join('share', package_name, 'pt'),
            glob('pt/*.pt')
        ),
        (
            os.path.join('share', package_name, 'launch'),
            glob('launch/*.launch.py')
        ),
    ],

    install_requires=['setuptools'],
    zip_safe=True,

    maintainer='wonseok',
    maintainer_email='wonseok@todo.todo',

    description='Vision and dual-arm robot control package',
    license='TODO: License declaration',

    extras_require={
        'test': ['pytest'],
    },

    entry_points={
        'console_scripts': [
            'B_arm_v2_tolerance = vision_pkg.B_arm_v2_tolerance:main',
            'base_top = vision_pkg.base_top:main',
            'centric_gripper_5axis = vision_pkg.centric_gripper_5axis:main',
            'dxl_bridge_suction_5axis = vision_pkg.dxl_bridge_suction_5axis:main',
            'experiment = vision_pkg.experiment:main',
            'kinematics_ee_5axis = vision_pkg.kinematics_ee_5axis:main',
            'path_collision_guard_ee_5axis = vision_pkg.path_collision_guard_ee_5axis:main',
            'process_sequence_5axis = vision_pkg.process_sequence_5axis:main',
            'rail_serial_5axis = vision_pkg.rail_serial_5axis:main',
            'trajectory_5axis = vision_pkg.trajectory_5axis:main',
            'vertical_path_5axis = vision_pkg.vertical_path_5axis:main',
            'baseplate_hole_wonseok = vision_pkg.baseplate_hole_wonseok:main',
            'Carpart_top = vision_pkg.Carpart_top:main',
            'Carpart_side = vision_pkg.Carpart_side:main',
            'rail_serial_centric_5axis = vision_pkg.rail_serial_centric_5axis:main',
            'centric_sequence_5axis = vision_pkg.centric_sequence_5axis:main',
            'Baseplate_hole_wonseok = vision_pkg.Baseplate_hole_wonseok:main',
            'dxl_bridge_centric_5axis= vision_pkg.dxl_bridge_centric_5axis:main',
            'a_arm_dot_experiment_5axis= vision_pkg.a_arm_dot_experiment_5axis:main',
            'Seperate = vision_pkg.Seperate:main',
            'car_process = vision_pkg.car_process:main',
            'base_process = vision_pkg.base_process:main',
            'Baseplate_side = vision_pkg.Baseplate_side:main',
            'jabara = vision_pkg.jabara:main',
            'top = vision_pkg.top:main',
            'side = vision_pkg.side:main',
            'cell = vision_pkg.cell:main',
            'cell_cli = vision_pkg.cell_cli:main',
            'assembly_debug = vision_pkg.assembly_debug:main',
            'lid_debug = vision_pkg.lid_debug:main',
            'lid_vision = vision_pkg.Baseplate_top:main',
            'lid_holes = vision_pkg.lid_holes:main',
            'holes = vision_pkg.holes:main',
        ],
    },
)
