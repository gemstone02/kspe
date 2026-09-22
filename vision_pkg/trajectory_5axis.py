"""
robot_description / trajectory_5axis.py
---------------------------------------
관절공간 부드러운 경로 생성.
 - 초기 관절값 -> 목표 관절값 사이에 중간점을 경유점으로 자동 생성
 - S-커브(5차 다항식) 시간 프로파일로 시작/끝 속도·가속도 0 (매끄러운 가감속)
 - 이동 시간은 관절 최대 이동량(거리)에 비례해서 자동 결정
"""

import numpy as np


def scurve(s):
    """s in [0,1] -> 0..1 로 부드럽게. 시작·끝에서 속도/가속도 = 0."""
    s = np.clip(s, 0.0, 1.0)
    return 6*s**5 - 15*s**4 + 10*s**3


def plan_joint_path(q_start, q_goal, hz=240.0,
                    speed_deg_per_s=60.0, t_min=0.6, t_max=4.0):
    """
    q_start, q_goal : 관절값 (길이 N)
    hz              : 명령 갱신 주기 (PyBullet 스텝과 동일하게)
    speed_deg_per_s : 대략적인 관절 속도. 이동시간 = 최대이동각 / 이 값
    반환 : (traj[K,N], T[s], K)
      - 중간점을 경유점으로 두고 전체를 하나의 S-커브 s 로 훑어
        경유점에서 멈추지 않고 부드럽게 통과한다.
    """
    q_start = np.asarray(q_start, float).ravel()
    q_goal  = np.asarray(q_goal,  float).ravel()
    q_via   = 0.5 * (q_start + q_goal)          # 관절공간 중간 경유점

    max_move_deg = np.degrees(np.abs(q_goal - q_start)).max()
    T = float(np.clip(max_move_deg / speed_deg_per_s, t_min, t_max))
    n = max(int(round(T * hz)), 2)

    traj = np.zeros((n, q_start.size))
    for i in range(n):
        s = scurve(i / (n - 1))                 # 0..1 부드럽게
        if s <= 0.5:
            traj[i] = q_start + (q_via - q_start) * (s / 0.5)
        else:
            traj[i] = q_via + (q_goal - q_via) * ((s - 0.5) / 0.5)
    return traj, T, n



def main(args=None):
    """
    ros2 run vision_pkg trajectory_5axis

    관절 경로 생성 함수 자체는 process_sequence_5axis가 import해서 사용한다.
    이 entry point는 ROS2 설치/함수 동작 확인용 진단 실행이다.

    Parameters
      q_start_deg       기본 [0,0,0,0]
      q_goal_deg        기본 [0,20,-30,-80]
      hz                기본 240
      speed_deg_per_s   기본 60
    """
    import rclpy
    from rclpy.node import Node

    rclpy.init(args=args)
    node = Node("trajectory_5axis")
    try:
        node.declare_parameter("q_start_deg", [0.0, 0.0, 0.0, 0.0])
        node.declare_parameter("q_goal_deg", [0.0, 20.0, -30.0, -80.0])
        node.declare_parameter("hz", 240.0)
        node.declare_parameter("speed_deg_per_s", 60.0)

        q0 = np.radians(
            np.asarray(node.get_parameter("q_start_deg").value, float)
        )
        q1 = np.radians(
            np.asarray(node.get_parameter("q_goal_deg").value, float)
        )
        hz = float(node.get_parameter("hz").value)
        speed = float(node.get_parameter("speed_deg_per_s").value)

        path, T, n = plan_joint_path(
            q0,
            q1,
            hz=hz,
            speed_deg_per_s=speed,
        )

        node.get_logger().info(
            "trajectory generated: samples=%d, T=%.3f s" % (n, T)
        )
        node.get_logger().info(
            "start(deg)=%s" % np.degrees(path[0]).round(3).tolist()
        )
        node.get_logger().info(
            "goal(deg)=%s" % np.degrees(path[-1]).round(3).tolist()
        )
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
