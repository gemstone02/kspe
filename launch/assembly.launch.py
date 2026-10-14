"""A-arm assembly trials without B arm or inspection cameras."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    def flag(name):
        return ParameterValue(LaunchConfiguration(name), value_type=bool)

    return LaunchDescription([
        DeclareLaunchArgument('dxl_port', default_value='/dev/ttyUSB0'),
        DeclareLaunchArgument('rail_port', default_value='/dev/ttyACM0'),
        DeclareLaunchArgument('use_robot', default_value='false'),
        DeclareLaunchArgument('use_rail', default_value='false'),
        DeclareLaunchArgument('use_simulation', default_value='true'),
        DeclareLaunchArgument('check_collision', default_value='true'),
        DeclareLaunchArgument('use_camera', default_value='false'),
        DeclareLaunchArgument('top_device', default_value='/dev/video0'),
        DeclareLaunchArgument('centric_grasp_limit_deg', default_value='180.0'),
        DeclareLaunchArgument('assembly_reference_mode', default_value='current'),
        DeclareLaunchArgument('insert_approach_clearance_mm', default_value='25.0'),
        DeclareLaunchArgument('insert_q5_rotation_deg', default_value='15.0'),
        DeclareLaunchArgument('joint_path_speed_deg_s', default_value='25.0'),
        DeclareLaunchArgument('vertical_speed_mps', default_value='0.02'),
        DeclareLaunchArgument('motor_trace', default_value='true'),
        DeclareLaunchArgument('motor_trace_period_sec', default_value='0.5'),
        DeclareLaunchArgument('motor_trace_ids', default_value='1,2,3,4,5,6,7'),
        DeclareLaunchArgument('motor_trace_dir',
                              default_value='/media/wonseok/새 볼륨/project/a_motors_log'),
        Node(package='v4l2_camera', executable='v4l2_camera_node',
             name='top_camera', namespace='top_camera',
             condition=IfCondition(LaunchConfiguration('use_camera')),
             parameters=[{'video_device': LaunchConfiguration('top_device')}],
             output='screen'),
        Node(package='vision_pkg', executable='holes', name='holes',
             condition=IfCondition(LaunchConfiguration('use_camera')),
             parameters=[{'image_topic': '/top_camera/image_raw'}], output='screen'),
        Node(package='vision_pkg', executable='assembly_debug', name='assembly_debug',
             parameters=[{
                 'assembly_debug_mode': True,
                 'use_robot': flag('use_robot'),
                 'use_rail': flag('use_rail'),
                 'use_b_robot': False,
                 'use_simulation': flag('use_simulation'),
                 'check_collision': flag('check_collision'),
                 'auto_start': False,
                 'auto_start_on_vision': False,
                 'dxl_port': LaunchConfiguration('dxl_port'),
                 'dxl_baud': 2000000,
                 'rail_port': LaunchConfiguration('rail_port'),
                 'rail_baud': 115200,
                 'rail_auto_zero': True,
                 'require_reference_home': False,
                 'assembly_reference_mode': LaunchConfiguration('assembly_reference_mode'),
                 'insert_approach_clearance_mm': ParameterValue(
                     LaunchConfiguration('insert_approach_clearance_mm'), value_type=float),
                 'insert_q5_rotation_deg': ParameterValue(
                     LaunchConfiguration('insert_q5_rotation_deg'), value_type=float),
                 'joint_path_speed_deg_s': ParameterValue(
                     LaunchConfiguration('joint_path_speed_deg_s'), value_type=float),
                 'vertical_speed_mps': ParameterValue(
                     LaunchConfiguration('vertical_speed_mps'), value_type=float),
                 'motor_trace': flag('motor_trace'),
                 'motor_trace_period_sec': ParameterValue(
                     LaunchConfiguration('motor_trace_period_sec'), value_type=float),
                 'motor_trace_ids': ParameterValue(
                     LaunchConfiguration('motor_trace_ids'), value_type=str),
                 'motor_trace_dir': ParameterValue(
                     LaunchConfiguration('motor_trace_dir'), value_type=str),
                 'centric_planet_max_angle_deg': ParameterValue(
                     LaunchConfiguration('centric_grasp_limit_deg'), value_type=float),
                 'centric_full_close_angle_deg': ParameterValue(
                     LaunchConfiguration('centric_grasp_limit_deg'), value_type=float),
             }], output='screen'),
    ])
