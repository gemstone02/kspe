"""Interactive command prompt for a cell started with ros2 launch.

Run in a second sourced terminal: python3 -m vision_pkg.cell_cli
"""

import time

import rclpy
from std_msgs.msg import String


def main():
    rclpy.init()
    node = rclpy.create_node('cell_cli')
    publisher = node.create_publisher(String, '/centric_sequence/command', 10)
    print('CELL> aj 0 0 0 0 0 | bj 0 0 0 0 0 0 | r 500 | stop | quit')
    try:
        while rclpy.ok():
            try:
                command = input('CELL> ').strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not command:
                continue
            if command.lower() in ('quit', 'exit', 'q'):
                break
            deadline = time.monotonic() + 2.0
            while publisher.get_subscription_count() == 0 and time.monotonic() < deadline:
                rclpy.spin_once(node, timeout_sec=0.1)
            if publisher.get_subscription_count() == 0:
                print('셀 노드가 없습니다. 먼저 ros2 launch를 실행하세요.')
                continue
            publisher.publish(String(data=command))
            rclpy.spin_once(node, timeout_sec=0.2)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
