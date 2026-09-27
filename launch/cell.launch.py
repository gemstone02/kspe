"""Two cameras and cell controller; A/centric and B have separate U2D2 ports."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    def flag(name):
        return ParameterValue(LaunchConfiguration(name), value_type=bool)

    args = [
        DeclareLaunchArgument('top_device', default_value='/dev/video0'),
        DeclareLaunchArgument('arm_device', default_value='/dev/video2'),
        DeclareLaunchArgument('dxl_port', default_value='/dev/ttyUSB0'),
        DeclareLaunchArgument('b_dxl_port', default_value='/dev/ttyUSB1'),
        DeclareLaunchArgument('rail_port', default_value='/dev/ttyACM0'),
        DeclareLaunchArgument('use_robot', default_value='false'),
        DeclareLaunchArgument('use_b_robot', default_value='false'),
        DeclareLaunchArgument('use_rail', default_value='false'),
        DeclareLaunchArgument('use_simulation', default_value='true'),
        DeclareLaunchArgument('check_collision', default_value='true'),
        DeclareLaunchArgument('simulate_inspection', default_value='false'),
        DeclareLaunchArgument('lid_probe_load_pct', default_value='30.0'),
        # Default centric output travel for this cell. Auto grasp still stops
        # earlier when sustained contact load is detected.
        DeclareLaunchArgument('centric_grasp_limit_deg', default_value='180.0'),
    ]
    return LaunchDescription(args + [
        Node(package='v4l2_camera', executable='v4l2_camera_node',
             name='top_camera', namespace='top_camera',
             parameters=[{'video_device': LaunchConfiguration('top_device')}],
             output='screen'),
        Node(package='v4l2_camera', executable='v4l2_camera_node',
             name='arm_camera', namespace='arm_camera',
             parameters=[{'video_device': LaunchConfiguration('arm_device')}],
             output='screen'),
        Node(package='vision_pkg', executable='top', name='top',
             parameters=[{'image_topic': '/top_camera/image_raw'}], output='screen'),
        Node(package='vision_pkg', executable='base_top',
             name='baseplate_top',
             parameters=[{'image_topic': '/top_camera/image_raw'}], output='screen'),
        Node(package='vision_pkg', executable='side', name='side',
             parameters=[{'image_topic': '/arm_camera/image_raw'}], output='screen'),
        Node(package='vision_pkg', executable='holes', name='holes',
             parameters=[{'image_topic': '/top_camera/image_raw'}], output='screen'),
        Node(package='vision_pkg', executable='cell', name='cell',
             parameters=[{
                 'use_robot': flag('use_robot'),
                 'use_b_robot': flag('use_b_robot'),
                 'use_rail': flag('use_rail'),
                 'use_simulation': flag('use_simulation'),
                 'check_collision': flag('check_collision'),
                 'simulate_inspection': flag('simulate_inspection'),
                 'dxl_port': LaunchConfiguration('dxl_port'),
                 'b_dxl_port': LaunchConfiguration('b_dxl_port'),
                 'rail_port': LaunchConfiguration('rail_port'),
                 'dxl_baud': 2000000,
                 'b_dxl_baud': 2000000,
                 'require_reference_home': False,
                 'rail_baud': 115200,
                 'rail_auto_zero': True,
                 'lid_probe_load_pct': ParameterValue(
                     LaunchConfiguration('lid_probe_load_pct'), value_type=float),
                 'centric_planet_max_angle_deg': ParameterValue(
                     LaunchConfiguration('centric_grasp_limit_deg'), value_type=float),
                 'centric_full_close_angle_deg': ParameterValue(
                     LaunchConfiguration('centric_grasp_limit_deg'), value_type=float),
             }], output='screen'),
    ])
