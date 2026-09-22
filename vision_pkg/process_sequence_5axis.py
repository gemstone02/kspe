#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Carpart_top.py 좌표 공정과 A/B 로봇 수동·동시 제어를 담당하는 공정 노드.

입력
  /detected_object (std_msgs/String)
  dot,x_mm,y_mm,0.0,0,OK

동작
  dot 상부 접근 -> 엔드이펙터 수직 유지 직선 하강
  -> 솔레노이드 흡착 ON -> 0.5초 유지 -> 같은 경로 상승

상부 NG: 레일 원점(0 mm)에서 불량 배출 -> A/B HOME
상부 OK: BOTTOM -> BACK -> FRONT -> LEFT -> RIGHT 검사
         -> 중간 NG 즉시 불량 배출
         -> 전부 OK이면 정상 위치(rail 50 mm) 배출 -> 초기 위치 복귀

Carpart_side.py의 OK/NG를 받아 B_HOME 복귀와 A-arm 분류 배출도 수행한다.
센트릭 그리퍼는 사용하지 않는다.
관절각은 모터각이 아니라 로봇 관절각(deg)이다.
"""

import json
import os
import re
import sys
import threading
import time
import xml.etree.ElementTree as ET
from collections import deque
from queue import Empty, Queue

import numpy as np
import pybullet as p
import pybullet_data
import rclpy
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from std_msgs.msg import Float64, String

try:
    from .kinematics_ee_5axis import N_JOINTS, fk_pos, ik_fix_candidates, wrap_to_pi
    from .path_collision_guard_ee_5axis import CollisionChecker
    from .rail_serial_5axis import RAIL_MAX_M, RAIL_MIN_M, RailSerial
    from .trajectory_5axis import plan_joint_path
    from .vertical_path_5axis import (
        joint_speed_limits,
        level_error,
        plan_level_descent_retimed,
        retime_path,
    )
except ImportError:
    from kinematics_ee_5axis import N_JOINTS, fk_pos, ik_fix_candidates, wrap_to_pi
    from path_collision_guard_ee_5axis import CollisionChecker
    from rail_serial_5axis import RAIL_MAX_M, RAIL_MIN_M, RailSerial
    from trajectory_5axis import plan_joint_path
    from vertical_path_5axis import (
        joint_speed_limits,
        level_error,
        plan_level_descent_retimed,
        retime_path,
    )


try:
    PACKAGE_SHARE = get_package_share_directory("vision_pkg")
except Exception:
    PACKAGE_SHARE = os.path.dirname(os.path.abspath(__file__))

DEFAULT_MESH_DIR = next(
    (x for x in (os.path.join(PACKAGE_SHARE, "meshes"),
                 os.path.join(PACKAGE_SHARE, "mashes")) if os.path.isdir(x)),
    os.path.join(PACKAGE_SHARE, "meshes"),
)

URDF_NAME = "robot_arm_5axis_rail_suction.urdf"
ARM_URDF_NAME = "robot_arm_5axis_suction.urdf"
RAIL_JOINT_NAME = "joint_rail"

# Carpart_top.py의 A3 좌표(mm) -> 기존 실측 로봇 좌표 변환값.
A3_W_MM, A3_H_MM = 420.0, 297.0
A3_LT_ROBOT_X_MM = 482.0
A3_LT_ROBOT_Y_MM = 214.55
USER_X_SIGN, USER_Y_SIGN, USER_Z_SIGN = -1.0, +1.0, +1.0

A_POSES = {
    "A_HOME": [0.0, 0.0, 0.0, 0.0, 0.0],
    "A_BOTTOM_VIEW": [0.0, 5.0, -10.0, -88.0, 0.0],
    "A_BACK_VIEW": [0.0, 30.0, -120.0, 60.0, 0.0],
    "A_FRONT_VIEW": [0.0, -30.0, -21.0, -103.0, 0.0],
    "A_LEFT_VIEW": [-40.0, 35.0, -100.0, -38.0, 0.0],
    "A_RIGHT_VIEW": [40.0, 35.0, -100.0, -38.0, 0.0],
}

B_POSES = {
    "B_HOME": [90.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "B_BOTTOM_VIEW": [180.0, 0.0, -10.0, 0.0, 84.0, 0.0],
    "B_BACK_VIEW": [180.0, 0.0, -100.0, 0.0, -39.0, 0.0],
    "B_FRONT_VIEW": [180.0, 0.0, -15.0, 0.0, 110.0, 0.0],
    "B_LEFT_VIEW": [140.0, 58.0, 0.0, -5.0, 100.0, -60.0],
    "B_RIGHT_VIEW": [220.0, -58.0, 0.0, 15.0, 100.0, 60.0],
}


RAIL_TRANSPORT_A_POSE = [0.0, 50.0, -50.0, -95.0, 0.0]
SORT_A_POSE = [180.0, -25.0, -50.0, -65.0, 0.0]
A_BOTTOM_BACK_TRANSIT = [0.0, 45.0, -45.0, -80.0, 0.0]
B_TRANSIT_SAFE = [180.0, 0.0, 0.0, 0.0, 0.0, 0.0]
B_BACK_FRONT_TRANSIT = [270.0, 0.0, -30.0, 0.0, -10.0, 0.0]
B_JOINT_NAMES = ("q1", "q2", "q3", "q4", "q5", "q6")
B_LIMITS_DEG = (
    (-360.0, 360.0), (-90.0, 90.0), (-120.0, 120.0),
    (-360.0, 360.0), (-120.0, 120.0), (-360.0, 360.0),
)

VIEW_KEYS = ("HOME", "BOTTOM", "BACK", "FRONT", "LEFT", "RIGHT")


def pose_key(arm, view):
    """단축 자세명(HOME/FRONT 등)을 실제 pose dictionary key로 변환한다."""
    view = str(view).upper()
    suffix = "HOME" if view == "HOME" else "%s_VIEW" % view
    return "%s_%s" % (str(arm).upper(), suffix)


def user_mm_to_arm_m(x_mm, y_mm, z_mm):
    return np.array([
        USER_X_SIGN * float(x_mm) / 1000.0,
        USER_Y_SIGN * float(y_mm) / 1000.0,
        USER_Z_SIGN * float(z_mm) / 1000.0,
    ], dtype=float)


class ProcessNode(Node):
    def __init__(self):
        super().__init__("process_sequence_5axis")
        d = self.declare_parameter
        d("pkg_dir", PACKAGE_SHARE)
        d("mesh_dir", DEFAULT_MESH_DIR)
        d("urdf_name", URDF_NAME)
        d("arm_urdf_name", ARM_URDF_NAME)
        d("use_robot", False)
        d("dxl_port", "/dev/ttyUSB1")
        d("use_b_robot", False)
        d("b_dxl_port", "/dev/ttyUSB0")
        d("b_dxl_baud", 2000000)
        d("b_auto_zero", True)
        d("use_rail", False)
        d("rail_port", "/dev/ttyACM0")
        d("rail_baud", 115200)
        d("rail_auto_zero", True)
        d("check_collision", True)
        d("sim_hz", 240.0)
        d("render_hz", 60.0)
        d("robot_cmd_hz", 50.0)
        d("joint_speed_deg_s", 60.0)
        d("vertical_speed_mps", 0.03)
        d("limit_joint_speed", True)
        d("speed_margin", 0.90)
        d("target_topic", "/detected_object")
        d("inspection_result_topic", "/b_inspection_result")
        d("inspection_command_topic", "/b_inspection_cmd")
        d("target_z_mm", 37.0)
        d("approach_mm", 80.0)
        d("vision_rail_ref_m", 0.0)
        d("suction_dwell_sec", 0.5)
        d("suction_ack_timeout_sec", 1.0)
        d("q5_hold_deg", 0.0)
        # 이전 실행 명령과의 호환성을 위해 파라미터 이름은 유지하지만,
        # 이 버전은 카메라 좌표와 모든 시퀀스를 항상 자동 실행한다.
        d("auto_run", True)
        d("manual_mode", False)
        d("direct_real_motion", True)
        d("settle_tol_deg", 1.5)
        d("settle_timeout_sec", 25.0)
        d("duplicate_radius_mm", 2.0)
        d("duplicate_hold_sec", 3.0)
        d("busy_flag_path", "/tmp/robot_auto_busy.flag")
        d("manual_collision_samples", 12)
        d("normal_rail_mm", 850.0)
        d("defect_rail_mm", 550.0)
        d("top_defect_rail_mm", 0.0)
        d("inspection_duration_sec", 3.0)
        d("inspection_result_timeout_sec", 5.0)
        d("rail_arrival_tol_mm", 2.0)
        d("rail_arrival_timeout_sec", 60.0)

        g = self.get_parameter
        self.pkg_dir = str(g("pkg_dir").value)
        self.mesh_dir = str(g("mesh_dir").value)
        self.urdf_name = str(g("urdf_name").value)
        self.arm_urdf_name = str(g("arm_urdf_name").value)
        self.use_robot = bool(g("use_robot").value)
        self.use_b_robot = bool(g("use_b_robot").value)
        self.use_rail = bool(g("use_rail").value)
        self.rail_auto_zero = bool(g("rail_auto_zero").value)
        self.sim_hz = max(20.0, float(g("sim_hz").value))
        self.render_hz = max(1.0, float(g("render_hz").value))
        self.robot_cmd_hz = max(1.0, float(g("robot_cmd_hz").value))
        self.joint_speed = max(1.0, float(g("joint_speed_deg_s").value))
        self.vertical_speed = max(0.003, float(g("vertical_speed_mps").value))
        self.limit_joint_speed = bool(g("limit_joint_speed").value)
        self.speed_margin = float(np.clip(float(g("speed_margin").value), 0.1, 1.0))
        self.target_z_mm = float(g("target_z_mm").value)
        self.approach_mm = float(g("approach_mm").value)
        self.vision_rail_ref = float(g("vision_rail_ref_m").value)
        self.suction_dwell = max(0.0, float(g("suction_dwell_sec").value))
        self.suction_ack_timeout = max(
            0.1, float(g("suction_ack_timeout_sec").value))
        self.q5 = np.radians(float(g("q5_hold_deg").value))
        self.auto_run = True
        self.manual_mode = False
        self.direct_real_motion = bool(g("direct_real_motion").value)
        self.settle_tol = np.radians(float(g("settle_tol_deg").value))
        self.settle_timeout = max(0.5, float(g("settle_timeout_sec").value))
        self.duplicate_radius_mm = max(0.0, float(g("duplicate_radius_mm").value))
        self.duplicate_hold_sec = max(0.0, float(g("duplicate_hold_sec").value))
        self.busy_flag_path = str(g("busy_flag_path").value)
        self.manual_collision_samples = max(0, int(g("manual_collision_samples").value))
        self.normal_rail_mm = float(g("normal_rail_mm").value)
        self.defect_rail_mm = float(g("defect_rail_mm").value)
        self.top_defect_rail_mm = float(g("top_defect_rail_mm").value)
        self.inspection_duration = max(
            0.1, float(g("inspection_duration_sec").value))
        self.inspection_result_timeout = max(
            0.5, float(g("inspection_result_timeout_sec").value))
        self.rail_arrival_tol_mm = max(
            0.1, float(g("rail_arrival_tol_mm").value))
        self.rail_arrival_timeout = max(
            1.0, float(g("rail_arrival_timeout_sec").value))
        if self.approach_mm <= 0:
            raise ValueError("approach_mm must be > 0")

        self.rail_origin = self._read_rail_origin()
        self._setup_simulation()
        self._setup_hardware(g)

        self.q_cmd = np.zeros(N_JOINTS)
        self.traj = None
        self.traj_index = 0
        self.step = None
        self.queue = deque()
        self.busy = False
        self.phase = "IDLE"
        self.cycle_id = ""
        self.current_target = None
        self.last_target = None
        self.last_target_time = -1e9
        self._await_approve = False
        self._play_t0 = None
        self._last_cmd_t = 0.0
        self.a_goal5 = np.concatenate([self.q_cmd, [self.q5]])
        self._settle_t0 = None
        self._settle_tick = 0
        self._render_tick = 0
        self._terminal_queue = Queue()
        self.b_q_deg = np.zeros(6, dtype=float)
        self.b_goal_deg = self.b_q_deg.copy()
        self.b_last_named_pose = "B_ZERO"
        self._b_settle_t0 = None
        self.comm_paused = False
        self._paused_phase = None
        self._pause_rail_mm = None
        self._pause_suction_on = None
        self._rail_wait_t0 = None
        self.pending_classification = None
        self.classification_armed = True
        self.top_judge = ""
        self.inspection_expected_view = None
        self.inspection_result = None
        self.inspection_result_t0 = None
        self.inspection_ok_count = 0
        self.suction_on = (
            bool(self.rail.suction_on) if self.rail is not None else False)

        self.state_pub = self.create_publisher(String, "/process_state", 10)
        self.rail_pub = self.create_publisher(Float64, "/rail_position", 10)
        self.create_subscription(
            String, str(g("target_topic").value), self.on_detected_object, 10)
        self.create_subscription(
            String, str(g("inspection_result_topic").value),
            self.on_inspection_result, 10)
        self.create_subscription(String, "/process_cmd", self.on_process_cmd, 10)
        self.inspection_cmd_pub = self.create_publisher(
            String, str(g("inspection_command_topic").value), 10)
        self.create_timer(1.0 / self.sim_hz, self.tick)
        self._apply_sim(self.q_cmd)
        self._start_terminal_thread()
        self._set_phase("IDLE")
        self.get_logger().info(
            "Ready: A=%s, B=%s, rail=%s, collision=%s, always_auto=True, "
            "direct_real=%s"
            % (self.bridge is not None, self.b_arm is not None,
               self.rail is not None, self.checker is not None,
               self.direct_real_motion))
        self._print_help()

    # --------------------------------------------------------------- setup
    def _setup_simulation(self):
        p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, 0)
        p.loadURDF("plane.urdf")
        p.resetDebugVisualizerCamera(1.6, 50, -25, [0.0, 0.4, 0.4])
        self.robot = p.loadURDF(
            self._prepare_urdf(self.urdf_name), [0, 0, 0], useFixedBase=True)
        self.rev, self.rail_jid = [], None
        for jid in range(p.getNumJoints(self.robot)):
            info = p.getJointInfo(self.robot, jid)
            name = info[1].decode() if isinstance(info[1], bytes) else str(info[1])
            if info[2] == p.JOINT_REVOLUTE:
                self.rev.append(jid)
            elif info[2] == p.JOINT_PRISMATIC and name == RAIL_JOINT_NAME:
                self.rail_jid = jid
        shape = p.createVisualShape(
            p.GEOM_SPHERE, radius=0.012, rgbaColor=[1, 0, 0, 0.9])
        self.marker = p.createMultiBody(
            0, baseVisualShapeIndex=shape, basePosition=[10, 10, 10])
        self.b_stop = p.addUserDebugParameter("EMERGENCY STOP", 1, 0, 0)
        self.b_home = p.addUserDebugParameter("GO HOME", 1, 0, 0)
        self._button = {"stop": 0, "home": 0}
        self._text = {}
        self.checker = None
        if bool(self.get_parameter("check_collision").value):
            self.checker = CollisionChecker(
                self._find_urdf(self.arm_urdf_name),
                self.pkg_dir, N_JOINTS, self.mesh_dir)

    def _setup_hardware(self, g):
        self.rail, self.rail_m = None, 0.0
        if self.use_rail:
            try:
                self.rail = RailSerial(
                    port=str(g("rail_port").value),
                    baud=int(g("rail_baud").value),
                    logger=self.get_logger())
                self.rail.connect()
                if self.rail_auto_zero and not self.rail.zero(wait=True, timeout=2.0):
                    raise RuntimeError("rail zero acknowledgement timeout")
                if not self.rail.suction(
                        False, wait=True, timeout=self.suction_ack_timeout):
                    raise RuntimeError("startup suction OFF acknowledgement timeout")
                self.get_logger().info("시작 안전상태: 솔레노이드 흡착 OFF 확인")
            except Exception as exc:
                self.get_logger().error("레일 연결/원점 실패: %s" % exc)
                if self.rail is not None:
                    try:
                        self.rail.close()
                    except Exception:
                        pass
                self.rail = None

        self.bridge, self.vmax, self.manual_vmax5 = None, None, None
        if self.use_robot:
            try:
                try:
                    from .dxl_bridge_suction_5axis import (
                        DxlBridge, JOINT_MOTORS, PROFILE_VELOCITY)
                except ImportError:
                    from dxl_bridge_suction_5axis import (
                        DxlBridge, JOINT_MOTORS, PROFILE_VELOCITY)
                self.bridge = DxlBridge(port=str(g("dxl_port").value))
                self.bridge.connect()
                self.bridge.capture_home()
                self.get_logger().warn(
                    "시작 자세를 q1~q5=0도로 저장했습니다. 기준 자세를 확인하세요.")
                if self.limit_joint_speed:
                    gears = [cfg["gear"] for cfg in JOINT_MOTORS[:N_JOINTS]]
                    self.vmax = joint_speed_limits(PROFILE_VELOCITY, gears)
                    cfg5 = JOINT_MOTORS[4]
                    dps5 = (PROFILE_VELOCITY[cfg5["ids"][0]]
                            * 0.229 * 6.0 / cfg5["gear"])
                    self.manual_vmax5 = np.concatenate(
                        [self.vmax, [np.radians(dps5)]])
            except Exception as exc:
                self.get_logger().error("A-arm 연결 실패: %s" % exc)
                if self.bridge is not None:
                    try:
                        self.bridge.close()
                    except Exception:
                        pass
                self.bridge = None

        self.b_arm = None
        if self.use_b_robot:
            try:
                b_port = str(g("b_dxl_port").value)
                if self.use_robot and b_port == str(g("dxl_port").value):
                    raise ValueError("A-arm과 B-arm은 서로 다른 U2D2 포트가 필요합니다.")
                try:
                    from .B_arm_v2_tolerance import (
                        DxlConfig, DynamixelArmB, RobotCalibration)
                except ImportError:
                    from B_arm_v2_tolerance import (
                        DxlConfig, DynamixelArmB, RobotCalibration)
                cfg = DxlConfig(
                    device_name=b_port,
                    baudrate=int(g("b_dxl_baud").value))
                self.b_arm = DynamixelArmB(cfg, RobotCalibration())
                self.b_arm.open()
                self.b_arm.set_extended_position_mode_all()
                self.b_arm.ping_all()
                if bool(g("b_auto_zero").value):
                    self.b_arm.set_current_as_zero()
                    self.get_logger().warn(
                        "B-arm 시작 자세를 q1~q6=0도로 저장했습니다.")
            except Exception as exc:
                self.get_logger().error("B-arm 연결 실패: %s" % exc)
                if self.b_arm is not None:
                    try:
                        self.b_arm.close()
                    except Exception:
                        pass
                self.b_arm = None

    # --------------------------------------------------------------- URDF
    def _find_urdf(self, name):
        candidates = [
            os.path.join(self.pkg_dir, "urdf", name),
            os.path.join(self.pkg_dir, name),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), name),
        ]
        for path in candidates:
            if os.path.isfile(path):
                return path
        raise FileNotFoundError("URDF not found: " + " | ".join(candidates))

    def _rewrite_mesh_paths(self, text):
        mesh_dir = os.path.abspath(self.mesh_dir)
        def replace(match):
            quote, raw = match.group(1), match.group(2)
            if not raw.lower().endswith((".stl", ".dae", ".obj")):
                return match.group(0)
            base = os.path.basename(raw.replace("\\", "/"))
            candidate = os.path.join(mesh_dir, base)
            if not os.path.isfile(candidate):
                for root, _dirs, files in os.walk(mesh_dir):
                    if base in files:
                        candidate = os.path.join(root, base)
                        break
            return ("filename=%s%s%s" % (quote, candidate, quote)
                    if os.path.isfile(candidate) else match.group(0))
        return re.sub(r"filename=([\"'])(.*?)(?:\1)", replace, text)

    def _prepare_urdf(self, name):
        source = self._find_urdf(name)
        with open(source, encoding="utf-8") as stream:
            text = self._rewrite_mesh_paths(stream.read())
        text = text.replace(
            "package://robot_description/", self.pkg_dir.rstrip("/") + "/")
        text = text.replace(
            "package://vision_pkg/", self.pkg_dir.rstrip("/") + "/")
        output = os.path.join("/tmp", "simple_" + os.path.basename(name))
        with open(output, "w", encoding="utf-8") as stream:
            stream.write(text)
        return output

    def _read_rail_origin(self):
        try:
            root = ET.parse(self._find_urdf(self.urdf_name)).getroot()
            for joint in root.findall("joint"):
                if joint.get("name") == RAIL_JOINT_NAME:
                    origin = joint.find("origin")
                    if origin is not None and origin.get("xyz"):
                        return np.array(
                            [float(v) for v in origin.get("xyz").split()])
        except Exception as exc:
            self.get_logger().warn("rail origin 읽기 실패(0 가정): %s" % exc)
        return np.zeros(3)

    # ---------------------------------------------------------- coordinates
    def world_to_arm(self, world_xyz):
        return np.asarray(world_xyz, float) - (
            self.rail_origin + np.array([0.0, self.rail_m, 0.0]))

    def a3_to_world(self, a3_x_mm, a3_y_mm, z_mm):
        robot_x_mm = A3_LT_ROBOT_X_MM - float(a3_y_mm)
        robot_y_mm = float(a3_x_mm) - A3_LT_ROBOT_Y_MM
        arm = user_mm_to_arm_m(robot_x_mm, robot_y_mm, z_mm)
        return arm + self.rail_origin + np.array(
            [0.0, self.vision_rail_ref, 0.0])

    # -------------------------------------------------------------- planner
    @staticmethod
    def _S(kind, **kwargs):
        result = {"kind": kind}
        result.update(kwargs)
        return result

    def _collision_error(self, trajectory, tag):
        if self.checker is None:
            return ""
        ok, index, reason = self.checker.check_path(trajectory)
        return "" if ok else "%s 충돌(index=%d, %s)" % (tag, index, reason)

    def _joint_approach(self, q_goal):
        traj, _duration, _count = plan_joint_path(
            self.q_cmd, q_goal, hz=self.sim_hz,
            speed_deg_per_s=self.joint_speed)
        if self.vmax is not None and self.limit_joint_speed:
            traj, _duration, info = retime_path(
                traj, self.vmax, self.sim_hz, margin=self.speed_margin)
            if traj is None:
                return None, "접근 속도 조정 실패: %s" % info
        error = self._collision_error(traj, "접근")
        return (None, error) if error else (traj, "")

    def _vertical_line(self, q_from, p_from, p_to, tag):
        traj, _duration, info = plan_level_descent_retimed(
            q_from, p_from, p_to,
            hz=self.sim_hz,
            lin_speed=self.vertical_speed,
            vmax=self.vmax if self.limit_joint_speed else None,
            margin=self.speed_margin,
            relaxed=False)
        if traj is None:
            return None, "%s 생성 실패: %s" % (tag, info)
        error = self._collision_error(traj, tag)
        return (None, error) if error else (traj, "")

    def build_dot_sequence(self, a3_x_mm, a3_y_mm):
        world_low = self.a3_to_world(
            a3_x_mm, a3_y_mm, self.target_z_mm)
        world_high = self.a3_to_world(
            a3_x_mm, a3_y_mm, self.target_z_mm + self.approach_mm)
        arm_low = self.world_to_arm(world_low)
        arm_high = self.world_to_arm(world_high)

        candidates = ik_fix_candidates(arm_high, q0=self.q_cmd)
        candidates = [
            q for q, _err in candidates
            if self.checker is None or self.checker.check_config(q)[0]
        ]
        if not candidates:
            return None, None, (
                "접근점에 충돌 없는 정확한 수직 IK 해가 없습니다: %s"
                % arm_high.round(4))
        candidates.sort(
            key=lambda q: float(np.linalg.norm(wrap_to_pi(q - self.q_cmd))))

        last_error = ""
        for q_high in candidates:
            # 이전 공정의 상승점처럼 현재 자세도 수직이면 접근 구간 전체를
            # Cartesian 경로로 계산해 엔드이펙터 수직을 계속 유지한다.
            if level_error(self.q_cmd) <= 1e-3:
                approach, error = self._vertical_line(
                    self.q_cmd, fk_pos(self.q_cmd), arm_high, "수직자세 접근")
            else:
                # q=0 시작처럼 비수직인 경우에는 첫 접근에서 수직 자세로 전환한다.
                approach, error = self._joint_approach(q_high)
            if approach is None:
                last_error = error
                continue

            q_high_actual = np.asarray(approach[-1], float)[:N_JOINTS]
            if level_error(q_high_actual) > 1e-4:
                last_error = "접근 끝 자세가 정확한 수직이 아닙니다."
                continue
            descent, error = self._vertical_line(
                q_high_actual, arm_high, arm_low, "수직 하강")
            if descent is None:
                last_error = error
                continue
            ascent = np.asarray(descent[::-1], float).copy()
            error = self._collision_error(ascent, "수직 상승")
            if error:
                last_error = error
                continue

            return [
                self._S("traj", traj=approach, name="dot 상부 접근"),
                self._S("settle", name="접근점 도달 확인"),
                self._S("traj", traj=descent, name="수직 하강"),
                self._S("settle", name="하강점 도달 확인", auto_next=True),
                self._S("suction", on=True, name="솔레노이드 흡착 ON",
                        auto_next=True),
                self._S("wait", sec=self.suction_dwell,
                        name="흡착 안정화 %.2f초" % self.suction_dwell,
                        auto_next=True),
                self._S("traj", traj=ascent, name="수직 상승"),
                self._S("settle", name="상승점 도달 확인"),
            ], world_low, ""
        return None, None, last_error or "수직 경로를 만들 수 없습니다."

    # ------------------------------------------------------------- callback
    def on_detected_object(self, msg):
        if self.busy:
            self.get_logger().warn("[DOT] 현재 이동 중 -> 새 좌표 무시")
            return
        fields = [x.strip() for x in str(msg.data).split(",")]
        if len(fields) < 6:
            self.get_logger().error(
                "[DOT] 형식 오류: dot,x_mm,y_mm,0.0,0,OK 필요")
            return
        if fields[0].lower() != "dot":
            return
        top_judge = fields[5].upper()
        if top_judge not in ("OK", "NG"):
            self.get_logger().info(
                "[DOT] 유효하지 않은 상태=%s -> 움직이지 않음" % fields[5])
            return
        if top_judge == "NG":
            self.get_logger().warn(
                "[DOT] 상부 임시판정 NG이지만 좌표는 사용합니다. "
                "파지 후 즉시 상부 불량 배출로 전환합니다.")
        try:
            x_mm, y_mm = float(fields[1]), float(fields[2])
        except ValueError:
            self.get_logger().error("[DOT] 좌표 숫자 변환 실패: %s" % msg.data)
            return
        self.start_dot_target(x_mm, y_mm, "Carpart_top", top_judge=top_judge)

    def on_inspection_result(self, msg):
        if self.comm_paused:
            self.get_logger().warn("통신 일시정지 중 검사 결과 무시: 재개 전 검사 상태를 확인하세요.")
            return
        try:
            data = json.loads(str(msg.data))
        except (TypeError, ValueError) as exc:
            self.get_logger().warn("검사 결과 JSON 해석 실패: %s" % exc)
            return
        result = str(data.get("result", data.get("judge", ""))).upper()
        if result not in ("OK", "NG", "NO_OBJECT"):
            return
        expected = self.inspection_expected_view
        if expected is None:
            self.get_logger().info(
                "검사 대기 중이 아닌 결과는 무시합니다: %s" % result)
            return
        received_view = str(data.get("view", "")).upper()
        if received_view and received_view != expected.upper():
            self.get_logger().warn(
                "이전/다른 자세 결과 무시: expected=%s, received=%s"
                % (expected, received_view))
            return
        self.get_logger().warn(
            "[INSPECTION] %s 수신(view=%s)" %
            (result, expected))
        if result == "NG":
            self.inspection_expected_view = None
            self.inspection_result = None
            self.get_logger().error(
                "다면검사 NG: 남은 검사를 중단하고 불량 배출로 전환합니다.")
            self._start_sort_sequence("NG", top_defect=False)
        else:
            # 이전 버전 카메라의 NO_OBJECT도 결함 미검출로 받아 다음 자세로
            # 진행한다. 새 측면 카메라는 자세별 OK/NG만 발행한다.
            self.inspection_result = "OK"
            if result == "NO_OBJECT":
                self.get_logger().warn(
                    "이전 버전의 NO_OBJECT 수신(%s): 불량 미검출로 처리합니다."
                    % expected)

    def _build_multiview_steps(self):
        steps = []

        def add_inspection(view_name):
            steps.extend([
                self._S("inspection_start", view=view_name,
                        name="%s 검사 START" % view_name, auto_next=True),
                self._S("wait", sec=self.inspection_duration,
                        name="%s %.1f초 검사" %
                             (view_name, self.inspection_duration),
                        auto_next=True),
                self._S("inspection_end", view=view_name,
                        name="%s 검사 END" % view_name, auto_next=True),
                self._S("inspection_result_wait", view=view_name,
                        name="%s OK/NG 결과 대기" % view_name,
                        auto_next=True),
            ])

        def add_view(a_goal, b_goal, view_name):
            steps.extend([
                self._S("dual_pose", a_goal=a_goal, b_goal=b_goal,
                        name="%s A/B 동시 이동" % view_name, auto_next=True),
                self._S("settle", name="%s A-arm 도달 확인" % view_name,
                        auto_next=True),
                self._S("b_settle", name="%s B-arm 도달 확인" % view_name,
                        auto_next=True),
            ])
            add_inspection(view_name)

        for view in ("BOTTOM", "BACK", "FRONT", "LEFT"):
            if view == "BACK":
                # A는 BOTTOM에서 지정된 중간 자세에 도달한 뒤 BACK으로 간다.
                # B는 A가 중간 자세에 도달할 때까지 BOTTOM에 머문다.
                steps.extend([
                    self._S("a_pose", goal=A_BOTTOM_BACK_TRANSIT,
                            name="A BOTTOM->BACK 중간자세", auto_next=True),
                    self._S("settle", name="A 중간자세 도달 확인",
                            auto_next=True),
                ])
                add_view(A_POSES["A_BACK_VIEW"],
                         B_POSES["B_BACK_VIEW"], "B_BACK_VIEW")
                continue
            if view == "FRONT":
                # A는 FRONT로 직행하고 B는 지정된 중간자세 도달 후 FRONT로 이동.
                steps.extend([
                    self._S("dual_pose", a_goal=A_POSES["A_FRONT_VIEW"],
                            b_goal=B_BACK_FRONT_TRANSIT,
                            name="A FRONT 직행 + B BACK->FRONT 중간자세",
                            auto_next=True),
                    self._S("b_settle", name="B BACK->FRONT 중간자세 도달 확인",
                            auto_next=True),
                    self._S("b_command", goal=B_POSES["B_FRONT_VIEW"],
                            name="B FRONT 이동", auto_next=True),
                    self._S("settle", name="B_FRONT_VIEW A-arm 도달 확인",
                            auto_next=True),
                    self._S("b_settle", name="B_FRONT_VIEW B-arm 도달 확인",
                            auto_next=True),
                ])
                add_inspection("B_FRONT_VIEW")
                continue
            add_view(
                A_POSES["A_%s_VIEW" % view],
                B_POSES["B_%s_VIEW" % view],
                "B_%s_VIEW" % view)

        # LEFT -> RIGHT: A는 RIGHT로 직행하고 B만 안전자세를 경유한다.
        steps.extend([
            self._S("dual_pose", a_goal=A_POSES["A_RIGHT_VIEW"],
                    b_goal=B_TRANSIT_SAFE,
                    name="A RIGHT 직행 + B 중간 안전자세",
                    auto_next=True),
            self._S("b_settle", name="B 중간 안전자세 도달 확인",
                    auto_next=True),
            self._S("b_command", goal=B_POSES["B_RIGHT_VIEW"],
                    name="B RIGHT 이동", auto_next=True),
            self._S("settle", name="B_RIGHT_VIEW A-arm 도달 확인",
                    auto_next=True),
            self._S("b_settle", name="B_RIGHT_VIEW B-arm 도달 확인",
                    auto_next=True),
        ])
        add_inspection("B_RIGHT_VIEW")

        steps.append(self._S(
            "route_sort", result="OK", name="5방향 검사 정상 배출",
            auto_next=True))
        return steps

    def _start_multiview_sequence(self):
        self.inspection_expected_view = None
        self.inspection_result = None
        self.inspection_ok_count = 0
        self.get_logger().warn(
            "상부 OK -> 5방향 다면검사 시작"
            "(BOTTOM -> BACK -> FRONT -> LEFT -> RIGHT)")
        self._start(self._build_multiview_steps(), "MULTIVIEW")

    def _start_sort_sequence(self, result, top_defect=False):
        result = str(result).upper()
        if result == "OK":
            rail_mm = self.normal_rail_mm
            phase = "SORT_OK"
        else:
            rail_mm = (self.top_defect_rail_mm
                       if top_defect else self.defect_rail_mm)
            phase = "SORT_TOP_NG" if top_defect else "SORT_NG"
        if not RAIL_MIN_M <= rail_mm / 1000.0 <= RAIL_MAX_M:
            self.get_logger().error("판정 레일 목표가 범위 밖입니다: %.1f mm" % rail_mm)
            return False
        self.inspection_expected_view = None
        self.inspection_result = None
        self.cycle_id = "%s_%s" % (phase, time.strftime("%Y%m%d_%H%M%S"))
        b_home_steps = []
        if self._b_is_at_right_view():
            b_home_steps.extend([
                self._S("b_command", goal=B_TRANSIT_SAFE,
                        name="B RIGHT->HOME 중간 안전자세",
                        auto_next=True),
                self._S("b_settle",
                        name="B 중간 안전자세 도달 확인",
                        auto_next=True),
            ])
        b_home_steps.append(
            self._S("b_command", goal=B_POSES["B_HOME"],
                    name="B-arm HOME 이동", auto_next=True))
        steps = b_home_steps + [
            self._S("b_settle", name="B-arm HOME 도달 확인", auto_next=True),
            self._S("a_pose", goal=RAIL_TRANSPORT_A_POSE,
                    name="A-arm 레일 이송 자세 이동", auto_next=True),
            self._S("settle", name="A-arm 레일 이송 자세 도달 확인",
                    auto_next=True),
            self._S("rail_abs", mm=rail_mm,
                    name="레일 %.1f mm 이동" % rail_mm, auto_next=True),
            self._S("rail_wait", mm=rail_mm,
                    name="레일 도착 확인", auto_next=True),
            self._S("a_pose", goal=SORT_A_POSE, name="A-arm 흡착 해제 자세 이동",
                    auto_next=True),
            self._S("settle", name="A-arm 흡착 해제 자세 도달 확인",
                    auto_next=True),
            self._S("suction", on=False, name="흡착 해제",
                    auto_next=True),
            self._S("a_pose", goal=RAIL_TRANSPORT_A_POSE,
                    name="A-arm 레일 복귀 자세 이동", auto_next=True),
            self._S("settle", name="A-arm 레일 복귀 자세 도달 확인",
                    auto_next=True),
            self._S("rail_abs", mm=0.0, name="레일 원점 복귀",
                    auto_next=True),
            self._S("rail_wait", mm=0.0, name="레일 원점 도달 확인",
                    auto_next=True),
            self._S("dual_pose", a_goal=A_POSES["A_HOME"],
                    b_goal=B_POSES["B_HOME"], name="A/B HOME 복귀",
                    auto_next=True),
            self._S("settle", name="A-arm HOME 도달 확인",
                    auto_next=True),
            self._S("b_settle", name="B-arm HOME 도달 확인",
                    auto_next=True),
        ]
        self.get_logger().warn(
            "[%s] B_HOME -> A 이송 자세 -> rail %.1f mm -> "
            "A 흡착 해제 자세 -> suction OFF -> A 이송 자세 -> rail 0 mm -> ALL HOME"
            % (phase, rail_mm))
        self._start(steps, phase)
        return True

    def start_dot_target(self, x_mm, y_mm, source="manual", top_judge="OK"):
        if self.busy:
            self.get_logger().warn("[DOT] 현재 이동 중입니다.")
            return False
        if not np.isfinite([x_mm, y_mm]).all():
            self.get_logger().error("[DOT] 유한하지 않은 좌표입니다.")
            return False
        if not (0 <= x_mm <= A3_W_MM and 0 <= y_mm <= A3_H_MM):
            self.get_logger().error(
                "[DOT] A3 범위 밖: (%.1f, %.1f)" % (x_mm, y_mm))
            return False
        now = time.time()
        target = np.array([x_mm, y_mm])
        if self.last_target is not None:
            distance = float(np.linalg.norm(target - self.last_target))
            if (distance <= self.duplicate_radius_mm
                    and now - self.last_target_time <= self.duplicate_hold_sec):
                self.get_logger().info("[DOT] 중복 좌표 무시(%.1f mm)" % distance)
                return False

        steps, world_low, error = self.build_dot_sequence(x_mm, y_mm)
        if steps is None:
            self.get_logger().error("[DOT] 경로 거부: %s" % error)
            return False
        self.top_judge = str(top_judge).upper()
        steps = list(steps)
        b_home_steps = []
        if self._b_is_at_right_view():
            b_home_steps.extend([
                self._S("b_command", goal=B_TRANSIT_SAFE,
                        name="B RIGHT->HOME 중간 안전자세", auto_next=True),
                self._S("b_settle", name="B 중간 안전자세 도달 확인",
                        auto_next=True),
            ])
        # B의 HOME 목표를 먼저 보내고 A의 흡착 접근을 시작한다.
        # B 도달 확인은 A 수직 상승 뒤에 수행하여 두 팔이 함께 움직이게 한다.
        steps = b_home_steps + [
            self._S("b_command", goal=B_POSES["B_HOME"],
                    name="A 흡착 접근과 함께 B-arm HOME 이동", auto_next=True),
        ] + steps + [
            self._S("b_settle", name="B-arm HOME 도달 확인", auto_next=True),
            self._S("top_route", judge=self.top_judge,
                    name="상부 판정 분기", auto_next=True)]
        self.cycle_id = time.strftime("%Y%m%d_%H%M%S")
        self.current_target = target
        self.last_target = target.copy()
        self.last_target_time = now
        p.resetBasePositionAndOrientation(
            self.marker, world_low.tolist(), [0, 0, 0, 1])
        self.get_logger().info(
            "[DOT] %s A3=(%.1f, %.1f), 하강 Z=%.1f mm, 접근=%.1f mm"
            % (source, x_mm, y_mm, self.target_z_mm, self.approach_mm))
        self._start(steps, "DOT")
        return True

    def on_process_cmd(self, msg):
        raw = str(msg.data).strip()
        if raw:
            self._handle_command_safely(raw, "ROS /process_cmd")

    # ---------------------------------------------------------- step engine
    def _start(self, steps, phase):
        if self.comm_paused:
            self.get_logger().error("통신 일시정지 중에는 새 공정을 시작할 수 없습니다.")
            return
        self.queue = deque(steps)
        self.step = None
        self.traj = None
        self.busy = True
        self._write_busy_flag()
        self._set_phase(phase)
        self._await_approve = False

    def _set_phase(self, phase):
        self.phase = phase
        msg = String()
        msg.data = "%s|%s" % (phase, self.cycle_id)
        self.state_pub.publish(msg)
        self.get_logger().info("--- PHASE: %s ---" % phase)

    def _activate_a_trajectory(self, trajectory):
        self.traj = np.asarray(trajectory, float)
        self.traj_index = 0
        self._play_t0 = None
        final = np.asarray(self.traj[-1], float).ravel()
        if final.size < 5:
            final = np.concatenate([final[:N_JOINTS], [self.q5]])
        self.a_goal5 = final[:5].copy()
        if self.bridge is not None and self.direct_real_motion:
            try:
                # 경로는 이미 충돌검사를 마쳤다. 실물에서는 중간 목표를
                # 하나씩 보내지 않고 최종 목표를 즉시 보내 DXL profile로 움직인다.
                self.bridge.command_joints(self.a_goal5.tolist())
                self.traj = None
                self.step["_done"] = True
            except Exception as exc:
                self._abort("A-arm 직접 목표 전송 실패: %s" % exc)
                return False
        return True

    def _next_step(self):
        if not self.queue:
            completed_phase = self.phase
            self.busy = False
            self.step = None
            self.current_target = None
            self._remove_busy_flag()
            self._set_phase("DONE")
            if completed_phase.startswith("SORT_"):
                self.classification_armed = True
                self.get_logger().info("다음 제품 판정을 다시 받을 수 있습니다.")
            return
        self.step = self.queue.popleft()
        self.step["_t0"] = None
        self.step["_done"] = False
        self.get_logger().info(
            "[%s] %s" % (self.step["kind"], self.step.get("name", "")))
        if self.step["kind"] == "traj":
            self._activate_a_trajectory(self.step["traj"])
        elif self.step["kind"] in ("a_pose", "dual_pose"):
            traj, error = self._make_a_trajectory(self.step[
                "a_goal" if self.step["kind"] == "dual_pose" else "goal"])
            if traj is None:
                self._abort("A-arm 자세 경로 거부: %s" % error)
                return
            # A 경로 계산/충돌검사를 먼저 끝낸 뒤 두 팔의 목표를 연달아 보낸다.
            if self.step["kind"] == "dual_pose":
                if not self._command_b(self.step["b_goal"]):
                    self._abort("B-arm 동시 자세 명령 실패")
                    return
            self.step["kind"] = "traj"
            self._activate_a_trajectory(traj)
        elif self.step["kind"] == "settle":
            self._settle_t0 = None
            self._settle_tick = 0
        elif self.step["kind"] == "b_settle":
            self._b_settle_t0 = None
        elif self.step["kind"] == "rail_wait":
            self._rail_wait_t0 = None
        elif self.step["kind"] == "b_command":
            if self._command_b(self.step["goal"]):
                self.step["_done"] = True
            else:
                self._abort("B-arm 명령 실패")
        elif self.step["kind"] == "rail_abs":
            if self._command_rail_abs(float(self.step["mm"])):
                self.step["_done"] = True
            else:
                self._abort("레일 절대이동 명령 실패")
        elif self.step["kind"] == "inspection_start":
            self.inspection_expected_view = str(self.step["view"])
            self.inspection_result = None
            self.inspection_result_t0 = None
            self._send_inspection_command(
                ["start", self.inspection_expected_view])
            self.step["_done"] = True
        elif self.step["kind"] == "inspection_end":
            self._send_inspection_command(["end"])
            self.step["_done"] = True
        elif self.step["kind"] == "inspection_result_wait":
            self.inspection_result_t0 = None
        elif self.step["kind"] == "top_route":
            if str(self.step["judge"]).upper() == "NG":
                self._start_sort_sequence("NG", top_defect=True)
            else:
                self._start_multiview_sequence()
            return
        elif self.step["kind"] == "route_sort":
            self._start_sort_sequence(str(self.step["result"]), top_defect=False)
            return
        elif self.step["kind"] == "suction":
            desired = bool(self.step["on"])
            if self.rail is None:
                if self.use_robot:
                    self._abort(
                        "실로봇 흡착용 레일 Arduino가 연결되어 있지 않습니다.")
                    return
                self.suction_on = desired
                self.get_logger().info(
                    "[SIM] 흡착 %s" % ("ON" if desired else "OFF"))
            else:
                ok = self.rail.suction(
                    desired, wait=True, timeout=self.suction_ack_timeout)
                if not ok:
                    self._abort(
                        "Arduino 흡착 %s 응답을 확인하지 못했습니다."
                        % ("ON" if desired else "OFF"))
                    return
                self.suction_on = desired
                self.get_logger().info(
                    "Arduino 흡착 %s 확인" % ("ON" if desired else "OFF"))
            self.step["_done"] = True

    def _play(self, now):
        if self._play_t0 is None:
            self._play_t0 = now
        last = len(self.traj) - 1
        wanted = int(round((now - self._play_t0) * self.sim_hz))
        self.traj_index = min(max(self.traj_index, wanted), last)
        q_now = np.asarray(self.traj[self.traj_index], float).ravel()
        self.q_cmd = q_now[:N_JOINTS].copy()
        if q_now.size > N_JOINTS:
            self.q5 = float(q_now[N_JOINTS])
        if (self.bridge is not None
                and now - self._last_cmd_t >= 1.0 / self.robot_cmd_hz):
            self._last_cmd_t = now
            self.bridge.command_joints(list(self.q_cmd) + [self.q5])
        if self.traj_index >= last:
            if self.bridge is not None:
                self.bridge.command_joints(list(self.q_cmd) + [self.q5])
            self.traj = None
            self._play_t0 = None
            self.step["_done"] = True

    def _settled(self, now):
        if self.bridge is None:
            return True
        if self._settle_t0 is None:
            self._settle_t0, self._settle_tick = now, 0
        self._settle_tick += 1
        if self._settle_tick < max(1, int(self.sim_hz * 0.10)):
            return False
        self._settle_tick = 0
        try:
            actual = np.asarray(self.bridge.read_joints(), float).ravel()
            if actual.size >= 5:
                # PyBullet은 별도의 사전 애니메이션이 아니라 실제 관절을 따라간다.
                self.q_cmd = actual[:N_JOINTS].copy()
                self.q5 = float(actual[4])
            goal = self.a_goal5
            n = min(actual.size, goal.size)
            error = float(np.max(np.abs(wrap_to_pi(actual[:n] - goal[:n]))))
            if error <= self.settle_tol:
                self._settle_t0 = None
                return True
            if now - self._settle_t0 > self.settle_timeout:
                self._abort("도달 시간초과(오차 %.2f도)" % np.degrees(error))
        except Exception as exc:
            self._abort("관절 도달 확인 실패: %s" % exc)
        return False

    @staticmethod
    def _b_dict(values):
        return {name: float(value) for name, value in zip(B_JOINT_NAMES, values)}

    def _command_b(self, values):
        goal = np.asarray(values, dtype=float).ravel()
        if goal.size != 6 or not np.isfinite(goal).all():
            self.get_logger().error("B-arm 목표는 유한한 관절각 6개여야 합니다.")
            return False
        for name, value, (lo, hi) in zip(B_JOINT_NAMES, goal, B_LIMITS_DEG):
            if not lo <= value <= hi:
                self.get_logger().error(
                    "B-arm %s=%.1f도는 제한 [%.1f, %.1f] 밖입니다."
                    % (name, value, lo, hi))
                return False
        try:
            goal_dict = self._b_dict(goal)
            if self.b_arm is not None:
                self.b_arm._validate_joint_limits(goal_dict)
                self.b_goal_tick = self.b_arm.mapper.joint_deg_to_goal_tick(
                    goal_dict)
                self.b_arm.move_joint_deg(goal_dict, wait=False)
            elif self.use_b_robot:
                self.get_logger().error("B-arm 실물 연결이 없습니다.")
                return False
            self.b_goal_deg = goal.copy()
            self.b_q_deg = goal.copy()
            self.b_last_named_pose = ""
            if np.allclose(goal, B_TRANSIT_SAFE, atol=1e-6):
                self.b_last_named_pose = "B_TRANSIT_SAFE"
            else:
                for pose_name, pose_values in B_POSES.items():
                    if np.allclose(goal, pose_values, atol=1e-6):
                        self.b_last_named_pose = pose_name
                        break
            if self.b_arm is None:
                self.get_logger().info("[SIM] B-arm -> %s" % goal.tolist())
            return True
        except Exception as exc:
            self.get_logger().error("B-arm 이동 명령 오류: %s" % exc)
            return False

    def _b_settled(self, now):
        if self.b_arm is None:
            return True
        if self._b_settle_t0 is None:
            self._b_settle_t0 = now
        try:
            present = self.b_arm.read_present_ticks()
            done = all(
                abs(present[dxl_id] - self.b_goal_tick[dxl_id])
                <= self.b_arm.cfg.position_tolerance_tick_by_id.get(
                    dxl_id, self.b_arm.cfg.position_tolerance_tick)
                for dxl_id in self.b_arm.cfg.ids)
            if done:
                self.b_q_deg = np.array([
                    self.b_arm.mapper.present_tick_to_joint_deg(present)[name]
                    for name in B_JOINT_NAMES], dtype=float)
                self._b_settle_t0 = None
                return True
            if now - self._b_settle_t0 > self.b_arm.cfg.move_timeout_sec:
                self._abort("B-arm 도달 시간초과")
        except Exception as exc:
            self._pause_for_b_comm("B-arm 도달 확인 실패: %s" % exc)
        return False

    def _pause_for_b_comm(self, reason):
        """B 도달 확인 통신 오류: 현재 단계/대기열을 보존하고 구동을 정지한다."""
        if self.comm_paused:
            return
        self.comm_paused = True
        self._paused_phase = self.phase
        self._pause_rail_mm = (float(self.rail.position_mm)
                               if self.rail is not None else None)
        self._pause_suction_on = bool(self.suction_on)
        self.get_logger().error("통신 일시정지: %s" % reason)
        if self.bridge is not None:
            try:
                self.bridge.emergency_stop()
            except Exception as exc:
                self.get_logger().error("A-arm 정지 확인 실패: %s" % exc)
        if self.rail is not None:
            try:
                self.rail.estop()
            except Exception as exc:
                self.get_logger().error("레일 정지 확인 실패: %s" % exc)
        if self.b_arm is not None:
            try:
                # 첫 실패 직후 무리한 다중 재시도를 하지 않는다. 모든 ID를
                # 읽을 수 있을 때에만 현재 위치 유지 명령을 시도한다.
                present = self.b_arm.read_present_ticks(
                    retry=1, use_fallback=False)
                self.b_arm.write_goal_ticks(present)
                self.get_logger().warn("B-arm 현재 위치 유지 명령 전송(실제 정지 확인 필요)")
            except Exception as exc:
                self.get_logger().error(
                    "B-arm 정지 명령을 확인하지 못했습니다: %s. "
                    "하드웨어 비상정지를 사용하고 현장을 확인하세요." % exc)
        self._set_phase("COMM_PAUSED")
        self.get_logger().warn(
            "공정은 자동 재개되지 않습니다. 현장 안전 확인 후 resume; "
            "복구할 수 없으면 stop으로 공정을 취소하세요.")

    def _resume_b_comm(self):
        """운전자 승인 후에만 통신·정지·다른 축의 위치를 검증하고 B 단계를 재개한다."""
        if not self.comm_paused or self.step is None or self.step["kind"] != "b_settle":
            self.get_logger().warn("재개 가능한 B-arm 통신 일시정지 단계가 없습니다.")
            return False
        if self.b_arm is None:
            self.get_logger().error("B-arm이 연결되지 않아 재개할 수 없습니다.")
            return False
        try:
            from .B_arm_v2_tolerance import (ADDR_TORQUE_ENABLE,
                                              ADDR_HARDWARE_ERROR_STATUS)
        except ImportError:
            from B_arm_v2_tolerance import (ADDR_TORQUE_ENABLE,
                                            ADDR_HARDWARE_ERROR_STATUS)
        try:
            first = self.b_arm.read_present_ticks(retry=2, use_fallback=False)
            time.sleep(0.15)
            second = self.b_arm.read_present_ticks(retry=2, use_fallback=False)
            for motor_id in self.b_arm.cfg.ids:
                if abs(second[motor_id] - first[motor_id]) > 15:
                    raise RuntimeError("B ID %d가 아직 움직이고 있습니다." % motor_id)
                torque = self.b_arm.read1(motor_id, ADDR_TORQUE_ENABLE)
                hwerr = self.b_arm.read1(motor_id, ADDR_HARDWARE_ERROR_STATUS)
                if torque != 1 or hwerr != 0:
                    raise RuntimeError(
                        "B ID %d torque=%d, hardware_error=%d" %
                        (motor_id, torque, hwerr))
            if self.bridge is not None:
                actual_a = np.asarray(self.bridge.read_joints(), float).ravel()
                if actual_a.size < 5 or not np.isfinite(actual_a[:5]).all():
                    raise RuntimeError("A-arm 실제 관절각을 확인할 수 없습니다.")
                error = float(np.max(np.abs(wrap_to_pi(
                    actual_a[:5] - self.a_goal5))))
                if error > self.settle_tol:
                    raise RuntimeError(
                        "A-arm이 이전 목표에서 %.2f도 벗어났습니다. "
                        "stop 후 수동 재정렬이 필요합니다." % np.degrees(error))
                self.q_cmd = actual_a[:N_JOINTS].copy()
                self.q5 = float(actual_a[4])
            if self.rail is not None:
                requested_at = time.time()
                if not self.rail.query():
                    raise RuntimeError("레일 상태 요청 전송 실패")
                deadline = time.time() + 1.0
                while self.rail.last_rx < requested_at and time.time() < deadline:
                    time.sleep(0.01)
                if self.rail.last_rx < requested_at or self.rail.stale(sec=3.0):
                    raise RuntimeError("레일 상태 응답이 없거나 오래되었습니다.")
                first_rail_mm = float(self.rail.position_mm)
                time.sleep(0.15)
                requested_at = time.time()
                if not self.rail.query():
                    raise RuntimeError("레일 두 번째 상태 요청 실패")
                deadline = time.time() + 1.0
                while self.rail.last_rx < requested_at and time.time() < deadline:
                    time.sleep(0.01)
                if self.rail.last_rx < requested_at:
                    raise RuntimeError("레일 두 번째 상태 응답 없음")
                if (not self.rail.zero_set or self.rail.moving or
                    abs(float(self.rail.position_mm) - first_rail_mm)
                        > self.rail_arrival_tol_mm or
                    abs(float(self.rail.position_mm) - self._pause_rail_mm)
                        > self.rail_arrival_tol_mm):
                    raise RuntimeError("레일 원점/정지/위치가 일시정지 전과 다릅니다.")
                if bool(self.rail.suction_on) != self._pause_suction_on:
                    raise RuntimeError("흡착 상태가 일시정지 전과 다릅니다.")
            # 이전 B 목표를 명시적으로 다시 보내되 다른 공정 단계는 재실행하지 않는다.
            if not self._command_b(self.b_goal_deg.copy()):
                raise RuntimeError("B-arm 이전 목표 재전송 실패")
        except Exception as exc:
            self.get_logger().error(
                "재개 거부: %s. 계속 일시정지 중입니다. 현장을 확인하세요." % exc)
            return False
        self._b_settle_t0 = None
        self.comm_paused = False
        previous_phase = self._paused_phase
        self._paused_phase = None
        self._pause_rail_mm = None
        self._pause_suction_on = None
        self._set_phase(previous_phase or "RUNNING")
        self.get_logger().warn("승인된 이전 B-arm 도달 확인 단계부터 공정을 재개합니다.")
        return True

    def _command_rail_abs(self, target_mm):
        target_m = float(target_mm) / 1000.0
        if not RAIL_MIN_M <= target_m <= RAIL_MAX_M:
            self.get_logger().error("레일 범위 밖: %.1f mm" % target_mm)
            return False
        if self.rail is None:
            if self.use_rail or self.use_robot:
                self.get_logger().error("실물 공정에 필요한 레일 연결이 없습니다.")
                return False
            self.rail_m = target_m
            return True
        if not self.rail.zero_set:
            self.get_logger().error("레일 원점이 설정되지 않았습니다.")
            return False
        return bool(self.rail.move_abs_mm(target_mm))

    def _rail_arrived(self, now, target_mm):
        if self.rail is None:
            return abs(self.rail_m * 1000.0 - target_mm) <= self.rail_arrival_tol_mm
        if self._rail_wait_t0 is None:
            self._rail_wait_t0 = now
        error = abs(float(self.rail.position_mm) - float(target_mm))
        if not self.rail.moving and error <= self.rail_arrival_tol_mm:
            self._rail_wait_t0 = None
            return True
        if now - self._rail_wait_t0 > self.rail_arrival_timeout:
            self._abort("레일 도달 시간초과(오차 %.1f mm)" % error)
        return False

    def _inspection_result_ready(self, now):
        if self.inspection_result_t0 is None:
            self.inspection_result_t0 = now
        if self.inspection_result == "OK":
            self.inspection_ok_count += 1
            self.get_logger().info(
                "검사 OK 누적: %d/5 (%s)" %
                (self.inspection_ok_count, self.inspection_expected_view))
            self.inspection_expected_view = None
            self.inspection_result = None
            self.inspection_result_t0 = None
            return True
        if now - self.inspection_result_t0 > self.inspection_result_timeout:
            self._abort(
                "검사 결과 대기 시간초과(view=%s)" %
                self.inspection_expected_view)
        return False

    def tick(self):
        now = time.time()
        self._poll_terminal()
        if self.rail is not None:
            self.rail_m = float(np.clip(
                self.rail.position_m, RAIL_MIN_M, RAIL_MAX_M))
            self.suction_on = bool(self.rail.suction_on)
        rail_msg = Float64()
        rail_msg.data = self.rail_m
        self.rail_pub.publish(rail_msg)

        if self._clicked(self.b_stop, "stop"):
            self._abort("EMERGENCY STOP")
        if self._clicked(self.b_home, "home") and not self.busy:
            self._manual_dual_joint(A_POSES["A_HOME"], B_POSES["B_HOME"])

        if self.comm_paused:
            self._render()
            return

        if self.busy and not self._await_approve and self.step is None:
            self._next_step()
        current = self.step
        if current is not None and not self._await_approve:
            kind = current["kind"]
            if kind == "traj" and not current.get("_done"):
                self._play(now)
            elif kind == "wait":
                if current["_t0"] is None:
                    current["_t0"] = now
                if now - current["_t0"] >= float(current["sec"]):
                    current["_done"] = True
            elif kind == "settle" and self._settled(now):
                current["_done"] = True
            elif kind == "b_settle" and self._b_settled(now):
                current["_done"] = True
            elif kind == "rail_wait" and self._rail_arrived(
                    now, float(current["mm"])):
                current["_done"] = True
            elif kind == "inspection_result_wait" and \
                    self._inspection_result_ready(now):
                current["_done"] = True
            if current.get("_done"):
                self.step = None
                if (not self.auto_run and self.queue
                        and not current.get("auto_next", False)):
                    self._await_approve = True
        self._render()

    # ------------------------------------------------------------ terminal
    def _start_terminal_thread(self):
        if not sys.stdin or not sys.stdin.isatty():
            return
        threading.Thread(target=self._terminal_loop, daemon=True).start()

    def _terminal_loop(self):
        while rclpy.ok():
            try:
                raw = input("AB> ").strip()
            except (EOFError, KeyboardInterrupt):
                return
            if raw:
                self._terminal_queue.put(raw)

    def _print_help(self):
        print("\n================ A + B CONTROL ================")
        print(" aj q1 q2 q3 q4 q5 : 관절각 직접 이동 [deg]")
        print(" bj q1..q6           : B 관절각 직접 이동 [deg]")
        print(" dualj A1..A5 B1..B6: A/B 거의 동시에 이동")
        print(" AFRONT / BFRONT     : FRONT 자세로 A만 / B만 이동")
        print(" FRONT               : A/B FRONT 자세 동시 이동")
        print(" HOME,BOTTOM,BACK,FRONT,LEFT,RIGHT 모두 같은 단축형 지원")
        print(" poses               : 저장 자세 목록")
        print(" p                   : A/B 관절각, 레일, 흡착 상태")
        print(" resume              : B 통신 일시정지 후 현장 확인·검증하여 재개")
        print(" target x y          : A3 dot 좌표 수동 시험 [mm]")
        print(" von / voff          : 솔레노이드 흡착 ON / OFF")
        print(" home                : A_HOME + B_HOME 동시 이동")
        print(" inspect start VIEW  : Carpart_side 검사 시작")
        print(" inspect end|reset   : 검사 종료/초기화")
        print(" classify ok|ng|topng: 정상/다면불량/상부불량 수동시험")
        print(" 카메라 dot 좌표는 별도 전환 없이 항상 자동 실행")
        print(" rz / r? / r mm      : 레일 원점/상태/절대이동")
        print(" stop                : 즉시 정지")
        print(" help                : 도움말")
        print("=============================================\n")

    @staticmethod
    def _smooth(u):
        u = np.clip(np.asarray(u, float), 0.0, 1.0)
        return 6*u**5 - 15*u**4 + 10*u**3

    def _current_joint5(self):
        if self.bridge is not None:
            try:
                value = np.asarray(self.bridge.read_joints(), float).ravel()
                if value.size >= 5:
                    return value[:5]
            except Exception as exc:
                self.get_logger().warn("관절 읽기 실패: %s" % exc)
        return np.concatenate([self.q_cmd, [self.q5]])

    def _current_b_deg(self):
        if self.b_arm is not None:
            try:
                joints = self.b_arm.mapper.present_tick_to_joint_deg(
                    self.b_arm.read_present_ticks())
                self.b_q_deg = np.array(
                    [joints[name] for name in B_JOINT_NAMES], dtype=float)
            except Exception as exc:
                self.get_logger().warn("B-arm 관절 읽기 실패: %s" % exc)
        return self.b_q_deg.copy()

    def _make_a_trajectory(self, values):
        if len(values) != 5:
            return None, "A-arm 목표는 관절각 5개여야 합니다."
        try:
            goal = np.radians(np.array([float(v) for v in values], dtype=float))
        except (TypeError, ValueError):
            return None, "A-arm 관절각은 숫자여야 합니다."
        if not np.isfinite(goal).all():
            return None, "A-arm 관절각은 유한한 값이어야 합니다."
        start = self._current_joint5()
        delta = goal - start
        if self.manual_vmax5 is not None:
            limit = np.maximum(self.manual_vmax5 * self.speed_margin, 1e-6)
            duration = max(0.5, float(np.max(1.875*np.abs(delta)/limit)))
        else:
            duration = max(
                0.6, float(np.max(np.degrees(np.abs(delta)))) / self.joint_speed)
        count = max(2, int(np.ceil(min(duration, 120.0)*self.sim_hz)) + 1)
        traj = start[None, :] + self._smooth(
            np.linspace(0, 1, count))[:, None] * delta[None, :]
        if self.checker is not None and self.manual_collision_samples > 0:
            ok, index, reason = self.checker.check_path(
                traj[:, :N_JOINTS], max_samples=self.manual_collision_samples)
            if not ok:
                return None, "경로 충돌(index=%d, %s)" % (index, reason)
        return traj, ""

    def _manual_joint(self, values):
        if self.busy:
            self.get_logger().warn("현재 이동 중입니다.")
            return False
        traj, error = self._make_a_trajectory(values)
        if traj is None:
            self.get_logger().error("A-arm 이동 거부: %s" % error)
            return False
        self._start([
            self._S("traj", traj=traj, name="수동 관절 이동", auto_next=True),
            self._S("settle", name="A-arm 도달 확인")],
            "MANUAL_JOINT")
        return True

    def _manual_b_joint(self, values):
        if self.busy:
            self.get_logger().warn("현재 이동 중입니다.")
            return False
        if len(values) != 6:
            self.get_logger().error("형식: bj q1 q2 q3 q4 q5 q6")
            return False
        try:
            goal = [float(v) for v in values]
        except ValueError:
            self.get_logger().error("B-arm 관절각은 숫자여야 합니다.")
            return False
        self._start([
            self._S("b_command", goal=goal, name="B-arm 관절 이동",
                    auto_next=True),
            self._S("b_settle", name="B-arm 도달 확인")],
            "MANUAL_B_JOINT")
        self._await_approve = False
        return True

    def _manual_dual_joint(self, a_values, b_values):
        if self.busy:
            self.get_logger().warn("현재 이동 중입니다.")
            return False
        traj, error = self._make_a_trajectory(a_values)
        if traj is None:
            self.get_logger().error("A-arm 이동 거부: %s" % error)
            return False
        # B 목표를 먼저 기록/전송하고 같은 tick에서 A trajectory를 시작한다.
        if not self._command_b(b_values):
            return False
        self._start([
            self._S("traj", traj=traj, name="A/B 동시: A-arm 이동",
                    auto_next=True),
            self._S("settle", name="A-arm 도달 확인", auto_next=True),
            self._S("b_settle", name="B-arm 도달 확인")],
            "MANUAL_DUAL")
        self._await_approve = False
        return True

    def _b_is_at_left_view(self):
        if self.b_last_named_pose == "B_LEFT_VIEW":
            return True
        actual = self._current_b_deg()
        return bool(np.max(np.abs(
            actual - np.asarray(B_POSES["B_LEFT_VIEW"], dtype=float))) <= 5.0)

    def _a_is_at_bottom_view(self):
        actual = np.asarray(self._current_joint5(), dtype=float).ravel()
        goal = np.radians(A_POSES["A_BOTTOM_VIEW"])
        return bool(actual.size >= 5 and np.isfinite(actual[:5]).all()
                    and np.max(np.abs(wrap_to_pi(actual[:5] - goal)))
                    <= np.radians(5.0))

    def _manual_a_bottom_to_back(self):
        if self.busy:
            self.get_logger().warn("현재 이동 중입니다.")
            return False
        self._start([
            self._S("a_pose", goal=A_BOTTOM_BACK_TRANSIT,
                    name="A BOTTOM->BACK 중간자세", auto_next=True),
            self._S("settle", name="A 중간자세 도달 확인", auto_next=True),
            self._S("a_pose", goal=A_POSES["A_BACK_VIEW"],
                    name="A BACK 이동", auto_next=True),
            self._S("settle", name="A BACK 도달 확인")],
            "A_BOTTOM_TO_BACK")
        return True

    def _manual_dual_bottom_to_back(self):
        if self.busy:
            self.get_logger().warn("현재 이동 중입니다.")
            return False
        self._start([
            self._S("a_pose", goal=A_BOTTOM_BACK_TRANSIT,
                    name="A BOTTOM->BACK 중간자세", auto_next=True),
            self._S("settle", name="A 중간자세 도달 확인", auto_next=True),
            self._S("dual_pose", a_goal=A_POSES["A_BACK_VIEW"],
                    b_goal=B_POSES["B_BACK_VIEW"],
                    name="A/B BACK 동시 이동", auto_next=True),
            self._S("settle", name="A BACK 도달 확인", auto_next=True),
            self._S("b_settle", name="B BACK 도달 확인")],
            "DUAL_BOTTOM_TO_BACK")
        return True

    def _b_is_at_back_view(self):
        if self.b_last_named_pose == "B_BACK_VIEW":
            return True
        actual = self._current_b_deg()
        return bool(np.max(np.abs(
            actual - np.asarray(B_POSES["B_BACK_VIEW"], dtype=float))) <= 5.0)

    def _b_is_at_right_view(self):
        if self.b_last_named_pose == "B_RIGHT_VIEW":
            return True
        actual = self._current_b_deg()
        return bool(np.max(np.abs(
            actual - np.asarray(B_POSES["B_RIGHT_VIEW"], dtype=float))) <= 5.0)

    def _manual_b_left_to_right(self):
        if self.busy:
            self.get_logger().warn("현재 이동 중입니다.")
            return False
        self._start([
            self._S("b_command", goal=B_TRANSIT_SAFE,
                    name="B LEFT->RIGHT 중간 안전자세", auto_next=True),
            self._S("b_settle", name="B 중간 안전자세 도달 확인",
                    auto_next=True),
            self._S("b_command", goal=B_POSES["B_RIGHT_VIEW"],
                    name="B RIGHT 이동", auto_next=True),
            self._S("b_settle", name="B RIGHT 도달 확인")],
            "B_LEFT_TO_RIGHT")
        return True

    def _manual_b_back_to_front(self):
        if self.busy:
            self.get_logger().warn("현재 이동 중입니다.")
            return False
        self._start([
            self._S("b_command", goal=B_BACK_FRONT_TRANSIT,
                    name="B BACK->FRONT 중간자세", auto_next=True),
            self._S("b_settle", name="B 중간자세 도달 확인", auto_next=True),
            self._S("b_command", goal=B_POSES["B_FRONT_VIEW"],
                    name="B FRONT 이동", auto_next=True),
            self._S("b_settle", name="B FRONT 도달 확인")],
            "B_BACK_TO_FRONT")
        return True

    def _manual_dual_back_to_front(self):
        if self.busy:
            self.get_logger().warn("현재 이동 중입니다.")
            return False
        self._start([
            self._S("dual_pose", a_goal=A_POSES["A_FRONT_VIEW"],
                    b_goal=B_BACK_FRONT_TRANSIT,
                    name="A FRONT 직행 + B BACK->FRONT 중간자세",
                    auto_next=True),
            self._S("b_settle", name="B 중간자세 도달 확인", auto_next=True),
            self._S("b_command", goal=B_POSES["B_FRONT_VIEW"],
                    name="B FRONT 이동", auto_next=True),
            self._S("settle", name="A FRONT 도달 확인", auto_next=True),
            self._S("b_settle", name="B FRONT 도달 확인")],
            "DUAL_BACK_TO_FRONT")
        return True

    def _manual_dual_left_to_right(self):
        if self.busy:
            self.get_logger().warn("현재 이동 중입니다.")
            return False
        self._start([
            self._S("dual_pose", a_goal=A_POSES["A_RIGHT_VIEW"],
                    b_goal=B_TRANSIT_SAFE,
                    name="A RIGHT 직행 + B 중간 안전자세", auto_next=True),
            self._S("b_settle", name="B 중간 안전자세 도달 확인",
                    auto_next=True),
            self._S("b_command", goal=B_POSES["B_RIGHT_VIEW"],
                    name="B RIGHT 이동", auto_next=True),
            self._S("settle", name="A RIGHT 도달 확인", auto_next=True),
            self._S("b_settle", name="B RIGHT 도달 확인")],
            "DUAL_LEFT_TO_RIGHT")
        return True

    def _handle_pose_shortcut(self, token):
        key = str(token).upper().replace("FORNT", "FRONT")
        if key in VIEW_KEYS:
            if key == "BACK" and not self.busy and self._a_is_at_bottom_view():
                return self._manual_dual_bottom_to_back()
            if key == "FRONT" and self._b_is_at_back_view():
                return self._manual_dual_back_to_front()
            if key == "RIGHT" and self._b_is_at_left_view():
                return self._manual_dual_left_to_right()
            return self._manual_dual_joint(
                A_POSES[pose_key("A", key)], B_POSES[pose_key("B", key)])
        if key.startswith("A") and key[1:] in VIEW_KEYS:
            if key == "ABACK" and not self.busy and self._a_is_at_bottom_view():
                return self._manual_a_bottom_to_back()
            return self._manual_joint(A_POSES[pose_key("A", key[1:])])
        if key.startswith("B") and key[1:] in VIEW_KEYS:
            if key == "BFRONT" and self._b_is_at_back_view():
                return self._manual_b_back_to_front()
            if key == "BRIGHT" and self._b_is_at_left_view():
                return self._manual_b_left_to_right()
            return self._manual_b_joint(B_POSES[pose_key("B", key[1:])])
        return None

    def _send_inspection_command(self, parts):
        if not parts:
            self.get_logger().error("형식: inspect start VIEW | end | reset")
            return
        action = parts[0].lower()
        if action == "start" and len(parts) == 2:
            payload = "START,%s" % parts[1].upper()
        elif action in ("end", "reset") and len(parts) == 1:
            payload = action.upper()
        else:
            self.get_logger().error("형식: inspect start VIEW | end | reset")
            return
        msg = String()
        msg.data = payload
        self.inspection_cmd_pub.publish(msg)
        self.get_logger().info("검사 명령 발행: %s" % payload)

    def _target_command(self, values):
        if len(values) != 2:
            self.get_logger().error("형식: target <x_mm> <y_mm>")
            return
        try:
            x_mm, y_mm = map(float, values)
        except ValueError:
            self.get_logger().error("target 좌표는 숫자여야 합니다.")
            return
        self.start_dot_target(x_mm, y_mm, "terminal")

    def _handle_terminal(self, raw):
        parts = raw.replace(",", " ").split()
        if not parts:
            return
        cmd = parts[0].lower()
        if self.comm_paused and cmd not in (
                "resume", "stop", "s", "reset",
                "p", "status", "voff", "help", "?"):
            self.get_logger().warn(
                "통신 일시정지 중에는 resume/stop/voff/상태 조회만 가능합니다.")
            return
        if cmd == "resume":
            self._resume_b_comm()
            return
        if len(parts) == 1 and self._handle_pose_shortcut(parts[0]) is not None:
            return
        if cmd in ("aj", "j"):
            self._manual_joint(parts[1:])
        elif cmd == "bj":
            self._manual_b_joint(parts[1:])
        elif cmd == "dualj":
            if len(parts) != 12:
                self.get_logger().error("형식: dualj A1 A2 A3 A4 A5 B1 B2 B3 B4 B5 B6")
            else:
                self._manual_dual_joint(parts[1:6], parts[6:12])
        elif cmd == "poses":
            print("\n[A POSES]")
            for name, values in A_POSES.items():
                print("  %-16s %s" % (name, values))
            print("[B POSES]")
            for name, values in B_POSES.items():
                print("  %-16s %s" % (name, values))
            print()
        elif cmd in ("p", "status"):
            a_deg = np.degrees(self._current_joint5())
            b_deg = (self.b_q_deg.copy() if self.comm_paused
                     else self._current_b_deg())
            print("\n[A CURRENT] q1={:.2f} q2={:.2f} q3={:.2f} "
                  "q4={:.2f} q5={:.2f} deg"
                  .format(*a_deg.tolist()))
            print("[B CURRENT] q1={:.2f} q2={:.2f} q3={:.2f} "
                  "q4={:.2f} q5={:.2f} q6={:.2f} deg"
                  .format(*b_deg.tolist()))
            print("[SYSTEM] rail={:.1f} mm | suction={} | phase={}\n"
                  .format(self.rail_m*1000.0,
                          "ON" if self.suction_on else "OFF", self.phase))
            if self.comm_paused:
                print("[WARN] B CURRENT는 마지막 기록값이며 실제 위치가 아닐 수 있습니다.\n")
        elif cmd == "target":
            self._target_command(parts[1:])
        elif cmd == "von":
            self._manual_suction(True)
        elif cmd == "voff":
            self._manual_suction(False)
        elif cmd == "h":
            self._manual_dual_joint(A_POSES["A_HOME"], B_POSES["B_HOME"])
        elif cmd == "inspect":
            self._send_inspection_command(parts[1:])
        elif cmd == "classify" and len(parts) == 2:
            value = parts[1].lower()
            if self.busy:
                self.get_logger().warn("현재 공정 중에는 판정 시험을 시작할 수 없습니다.")
            elif value == "ok":
                self._start_sort_sequence("OK", top_defect=False)
            elif value == "ng":
                self._start_sort_sequence("NG", top_defect=False)
            elif value == "topng":
                self._start_sort_sequence("NG", top_defect=True)
            else:
                self.get_logger().error("형식: classify ok | ng | topng")
        elif cmd in ("auto", "manual"):
            self.get_logger().info(
                "이 버전은 항상 자동 모드입니다. 별도 on/off 명령이 필요 없습니다.")
        elif cmd in ("stop", "s"):
            self._abort("터미널 STOP")
        elif cmd == "reset":
            self._reset()
        elif cmd in ("rz", "railzero"):
            self._rail_zero()
        elif cmd == "r?":
            self._rail_status()
        elif cmd == "r":
            self._rail_move(parts[1:])
        elif cmd in ("help", "?"):
            self._print_help()
        else:
            self.get_logger().warn("알 수 없는 명령: %s" % raw)

    def _manual_suction(self, on):
        desired = bool(on)
        if self.rail is None:
            if self.use_robot:
                self.get_logger().error(
                    "레일 Arduino가 연결되지 않아 실물 흡착 명령을 보낼 수 없습니다.")
                return False
            self.suction_on = desired
            self.get_logger().info(
                "[SIM] 흡착 %s" % ("ON" if desired else "OFF"))
            return True
        ok = self.rail.suction(
            desired, wait=True, timeout=self.suction_ack_timeout)
        if ok:
            self.suction_on = desired
            self.get_logger().info(
                "흡착 %s 확인" % ("ON" if desired else "OFF"))
        else:
            self.get_logger().error(
                "흡착 %s 응답 시간초과" % ("ON" if desired else "OFF"))
        return ok

    def _handle_command_safely(self, raw, source):
        """잘못된 명령 하나가 ROS executor와 전체 하드웨어를 종료하지 않게 한다."""
        try:
            if raw.strip().lower().split()[:1] == ["resume"] and source != "terminal":
                self.get_logger().error("resume은 로봇 옆에서 확인한 뒤 터미널에서만 입력하세요.")
                return
            self._handle_terminal(raw)
        except Exception as exc:
            self.get_logger().error(
                "[%s] 명령 처리 오류: %s. 안전 정지 후 명령 대기를 계속합니다."
                % (source, exc))
            self._abort("명령 처리 오류")

    def _poll_terminal(self):
        for _ in range(4):
            try:
                raw = self._terminal_queue.get_nowait()
            except Empty:
                return
            self._handle_command_safely(raw, "terminal")

    def _rail_zero(self):
        if self.busy:
            self.get_logger().warn("이동 중에는 레일 원점을 바꿀 수 없습니다.")
        elif self.rail is None:
            self.rail_m = 0.0
        elif self.rail.zero(wait=True, timeout=2.0):
            self.rail_m = 0.0
        else:
            self.get_logger().error("레일 원점 응답 시간초과")

    def _rail_status(self):
        if self.rail is None:
            self.get_logger().info("rail=%.1f mm(sim)" % (self.rail_m*1000.0))
        else:
            self.rail.query()
            self.get_logger().info(
                "rail=%.1f mm, zero=%s, moving=%s"
                % (self.rail.position_mm, self.rail.zero_set, self.rail.moving))

    def _rail_move(self, values):
        if len(values) != 1:
            self.get_logger().error("형식: r <mm>")
            return
        if self.busy:
            self.get_logger().warn("팔 이동 중에는 레일을 움직일 수 없습니다.")
            return
        try:
            target_mm = float(values[0])
        except ValueError:
            self.get_logger().error("레일 위치는 숫자여야 합니다.")
            return
        target_m = target_mm / 1000.0
        if not RAIL_MIN_M <= target_m <= RAIL_MAX_M:
            self.get_logger().error("레일 범위 밖: %.1f mm" % target_mm)
            return
        if self.rail is None:
            self.rail_m = target_m
        elif not self.rail.zero_set:
            self.get_logger().error("먼저 rz로 레일 원점을 설정하세요.")
            return
        else:
            self.rail.move_abs_mm(target_mm)

    # --------------------------------------------------------------- safety
    def _abort(self, reason):
        self.get_logger().error("중단: %s" % reason)
        was_comm_paused = self.comm_paused
        self.comm_paused = False
        self._paused_phase = None
        self._pause_rail_mm = None
        self._pause_suction_on = None
        inspection_was_active = self.inspection_expected_view is not None
        self.queue.clear()
        self.step = self.traj = None
        self.busy = False
        self._await_approve = False
        self.pending_classification = None
        self.classification_armed = True
        self.inspection_expected_view = None
        self.inspection_result = None
        self.inspection_result_t0 = None
        self.inspection_ok_count = 0
        if inspection_was_active and hasattr(self, "inspection_cmd_pub"):
            self._send_inspection_command(["reset"])
        if self.bridge is not None:
            try:
                self.bridge.emergency_stop()
            except Exception:
                pass
        # B 재시도에 오래 걸리더라도 레일은 먼저 정지시킨다.
        if self.rail is not None:
            try:
                self.rail.estop()
            except Exception:
                pass
        if self.b_arm is not None:
            try:
                present = (self.b_arm.read_present_ticks(
                    retry=1, use_fallback=False) if was_comm_paused
                    else self.b_arm.read_present_ticks())
                self.b_arm.write_goal_ticks(present)
                joints = self.b_arm.mapper.present_tick_to_joint_deg(present)
                self.b_q_deg = np.array(
                    [joints[name] for name in B_JOINT_NAMES], dtype=float)
                self.b_goal_deg = self.b_q_deg.copy()
                self.b_goal_tick = dict(present)
            except Exception as exc:
                self.get_logger().error(
                    "B-arm 비상정지 확인 실패: %s. 하드웨어 비상정지와 "
                    "현장 확인이 필요합니다." % exc)
        self._remove_busy_flag()
        self._set_phase("ABORT")

    def _reset(self):
        if self.busy:
            self._abort("reset")
        self.queue.clear()
        self.step = self.traj = None
        self.busy = False
        self.current_target = None
        self.pending_classification = None
        self.classification_armed = True
        self.inspection_expected_view = None
        self.inspection_result = None
        self.inspection_result_t0 = None
        self.inspection_ok_count = 0
        self._remove_busy_flag()
        self._set_phase("IDLE")

    def _write_busy_flag(self):
        try:
            with open(self.busy_flag_path, "w", encoding="utf-8") as stream:
                stream.write(self.cycle_id + "\n")
        except Exception as exc:
            self.get_logger().warn("busy flag 생성 실패: %s" % exc)

    def _remove_busy_flag(self):
        try:
            if self.busy_flag_path and os.path.exists(self.busy_flag_path):
                os.unlink(self.busy_flag_path)
        except Exception as exc:
            self.get_logger().warn("busy flag 삭제 실패: %s" % exc)

    # --------------------------------------------------------------- render
    def _clicked(self, button, key):
        try:
            value = p.readUserDebugParameter(button)
        except Exception as exc:
            if not getattr(self, "_button_read_warned", False):
                self.get_logger().warn(
                    "PyBullet 버튼 읽기 실패(노드는 계속 실행): %s" % exc)
                self._button_read_warned = True
            return False
        if value > self._button[key]:
            self._button[key] = value
            return True
        return False

    def _apply_sim(self, q):
        if self.rail_jid is not None:
            p.resetJointState(self.robot, self.rail_jid, float(self.rail_m))
        full = list(np.asarray(q, float)[:N_JOINTS]) + [self.q5]
        full += [0.0] * max(0, len(self.rev)-len(full))
        for index, jid in enumerate(self.rev):
            p.resetJointState(self.robot, jid, float(full[index]))

    def _render(self):
        self._render_tick += 1
        if self._render_tick < max(1, int(round(self.sim_hz/self.render_hz))):
            return
        self._render_tick = 0
        self._apply_sim(self.q_cmd)
        p.stepSimulation()
        target = ("-" if self.current_target is None else
                  "%.1f, %.1f mm" % tuple(self.current_target))
        lines = [
            ("phase", "PHASE: %s [ALWAYS AUTO]" % self.phase,
             [1, 1, .3], 1.35),
            ("target", "DOT A3: %s | rail %.1f mm | suction %s" %
             (target, self.rail_m*1000.0,
              "ON" if self.suction_on else "OFF"), [.3, 1, .4], 1.20),
            ("step", "STEP: %s | remaining %d" %
             ((self.step or {}).get("name", "-"), len(self.queue)),
             [.8, .8, 1], 1.05),
        ]
        for key, text, color, height in lines:
            if self._text.get(key) is not None:
                p.removeUserDebugItem(self._text[key])
            self._text[key] = p.addUserDebugText(
                text, [0, 0, height], textColorRGB=color, textSize=1.1)

    def shutdown(self):
        self._remove_busy_flag()
        if self.bridge is not None:
            try:
                self.bridge.emergency_stop()
                self.bridge.close()
            except Exception:
                pass
        if self.b_arm is not None:
            try:
                present = self.b_arm.read_present_ticks()
                self.b_arm.write_goal_ticks(present)
                self.b_arm.close()
            except Exception:
                pass
        if self.rail is not None:
            try:
                self.rail.suction(False, wait=False)
                self.rail.estop()
                self.rail.close()
            except Exception:
                pass
        if self.checker is not None:
            try:
                self.checker.close()
            except Exception:
                pass
        try:
            p.disconnect()
        except Exception:
            pass


def main(args=None):
    rclpy.init(args=args)
    node = ProcessNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

