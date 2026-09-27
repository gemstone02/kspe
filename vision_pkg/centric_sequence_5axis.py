#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Baseplate shaft/bearing assembly for A-arm, rail and centric gripper.

This single sequence file receives ``/detected_holes`` coordinates and runs
six pick/place cycles: shaft 1 at 663 mm to MID, shaft 2 at 718 mm to SIDE,
bearings at 842/913 mm to MID/SIDE, then gears at 663/718 mm to MID/SIDE. Placement is at
Z=25 mm for shafts/gears and Z=20 mm for bearings by default. The process starts from the PyBullet START button by default.
EMERGENCY STOP holds the current positions without terminating the ROS 2 node,
so another command can follow.
"""

import math
import multiprocessing as mp
import os
import queue
import re
import sys
import tempfile
import threading
import time
import xml.etree.ElementTree as ET

import numpy as np

import rclpy
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, String

try:
    from .centric_gripper_5axis import CentricGripper, GraspResult
    from .dxl_bridge_centric_5axis import (
        DxlBridge,
        JOINT_MOTORS,
        PROFILE_VELOCITY,
    )
    from .kinematics_ee_5axis import (
        N_JOINTS,
        fk_pos,
        ik_fix_candidates,
        wrap_to_pi,
    )
    from .rail_serial_centric_5axis import RAIL_MAX_M, RAIL_MIN_M, RailSerial
    from .trajectory_5axis import plan_joint_path
    from .vertical_path_5axis import (
        joint_speed_limits,
        level_error,
        plan_level_descent_retimed,
        retime_path,
    )
except ImportError:
    from centric_gripper_5axis import CentricGripper, GraspResult
    from dxl_bridge_centric_5axis import (
        DxlBridge,
        JOINT_MOTORS,
        PROFILE_VELOCITY,
    )
    from kinematics_ee_5axis import (
        N_JOINTS,
        fk_pos,
        ik_fix_candidates,
        wrap_to_pi,
    )
    from rail_serial_centric_5axis import RAIL_MAX_M, RAIL_MIN_M, RailSerial
    from trajectory_5axis import plan_joint_path
    from vertical_path_5axis import (
        joint_speed_limits,
        level_error,
        plan_level_descent_retimed,
        retime_path,
    )


def _deg_to_rad(values):
    return [math.radians(float(value)) for value in values]


def _wrapped_error(a, b):
    return (float(a) - float(b) + math.pi) % (2.0 * math.pi) - math.pi


try:
    PACKAGE_SHARE = get_package_share_directory("vision_pkg")
except Exception:
    PACKAGE_SHARE = os.path.dirname(os.path.abspath(__file__))

DEFAULT_MESH_DIR = next(
    (
        path
        for path in (
            os.path.join(PACKAGE_SHARE, "meshes"),
            os.path.join(PACKAGE_SHARE, "mashes"),
        )
        if os.path.isdir(path)
    ),
    os.path.join(PACKAGE_SHARE, "meshes"),
)

URDF_NAME = "robot_arm_5axis_rail_suction.urdf"
RAIL_JOINT_NAME = "joint_rail"



def _queue_put_latest(q, value):
    """Best-effort non-blocking queue update.

    The real-robot process must never wait for PyBullet.  If the viewer queue
    is full, the stale frame is discarded and replaced when possible.
    """
    try:
        q.put_nowait(value)
        return True
    except queue.Full:
        try:
            q.get_nowait()
        except (queue.Empty, OSError, EOFError):
            pass
        try:
            q.put_nowait(value)
            return True
        except (queue.Full, OSError, EOFError):
            return False
    except (OSError, EOFError):
        return False


def _pybullet_viewer_worker(config, state_queue, command_queue, stop_event):
    """PyBullet GUI process.

    IMPORTANT:
    - This function runs in a separate OS process.
    - It never opens or accesses the Dynamixel/U2D2 port.
    - GUI/rendering delays therefore cannot block real motor communication.
    """
    try:
        # Lower the GUI process priority so it cannot compete aggressively with
        # the real robot controller on Linux.
        try:
            os.nice(10)
        except Exception:
            pass

        import pybullet as pb
        import pybullet_data as pb_data
    except Exception as exc:
        try:
            command_queue.put_nowait(("viewer_error", "PyBullet import failed: %s" % exc))
        except Exception:
            pass
        return

    client = None
    try:
        client = pb.connect(pb.GUI)
        if client < 0:
            raise RuntimeError("PyBullet GUI connection failed")

        pb.setAdditionalSearchPath(pb_data.getDataPath())
        pb.setGravity(0, 0, 0)
        pb.loadURDF("plane.urdf")
        pb.resetDebugVisualizerCamera(1.6, 50, -25, [0.0, 0.4, 0.4])

        robot = pb.loadURDF(
            str(config["urdf_path"]),
            [0, 0, 0],
            useFixedBase=True,
        )

        revolute = []
        rail_jid = None
        for joint_id in range(pb.getNumJoints(robot)):
            info = pb.getJointInfo(robot, joint_id)
            name = info[1].decode() if isinstance(info[1], bytes) else str(info[1])
            if info[2] == pb.JOINT_REVOLUTE:
                revolute.append(joint_id)
            elif info[2] == pb.JOINT_PRISMATIC and name == RAIL_JOINT_NAME:
                rail_jid = joint_id

        labels = {
            "stop": "!!! EMERGENCY STOP !!!   [X]",
            "resume_cycle": ">>> RESUME PAUSED PROCESS <<<   [R]",
            "abort_cycle": ">>> ABORT PAUSED PROCESS <<<",
            "tool_done": ">>> TOOL CHANGE COMPLETE <<<   [T]",
            "b_home": ">>> B ARM HOME <<<   [B]",
            "home": ">>> A ARM HOME [0,0,0,0,0] <<<   [H]",
            "rail_home": ">>> RAIL HOME (A+RAIL) <<<",
            "all_initial": ">>> ALL INITIAL (A + RAIL) <<<   [I]",
            "open": ">>> OPEN GRIPPER <<<   [O]",
            "grasp": ">>> CENTRIC GRASP ONLY <<<   [G]",
            "motors": ">>> READ MOTOR ID 1~8 STATUS <<<   [M]",
            "pick_pose_preclose": ">>> A PICK POSE + GRIPPER PRECLOSE <<<",
            "test_mid_vertical": ">>> TEST MID VERTICAL APPROACH Z30 <<<",
            "test_side_vertical": ">>> TEST SIDE VERTICAL APPROACH Z30 <<<",
            "sample": "LOAD SAMPLE HOLE COORDS",
        }

        buttons = {}
        button_values = {}
        for key, label in labels.items():
            buttons[key] = pb.addUserDebugParameter(label, 1, 0, 0)
            button_values[key] = 0.0

        marker_ids = {}
        for key, color in {
            "mid": [0.1, 0.9, 0.2, 0.95],
            "side": [0.1, 0.5, 1.0, 0.95],
        }.items():
            shape = pb.createVisualShape(
                pb.GEOM_SPHERE,
                radius=0.012,
                rgbaColor=color,
            )
            marker_ids[key] = pb.createMultiBody(
                0,
                baseVisualShapeIndex=shape,
                basePosition=[10, 10, 10],
            )

        debug_text = {}
        holes_text_id = None
        last_text_update = 0.0

        latest = {
            "q_deg": [0.0] * 5,
            "rail_mm": 0.0,
            "gripper_state": "OPEN",
            "phase": "STARTING",
            "busy": False,
            "markers": {},
            "holes_text": "",
        }

        hz = max(5.0, float(config.get("viewer_hz", 20.0)))
        period = 1.0 / hz
        next_tick = time.monotonic()

        key_map = {
            "x": "stop",
            "r": "resume_cycle",
            "t": "tool_done",
            "b": "b_home",
            "h": "home",
            "i": "all_initial",
            "o": "open",
            "g": "grasp",
            "m": "motors",
        }

        while not stop_event.is_set():
            # Drain state queue and use only the newest frame.
            while True:
                try:
                    latest = state_queue.get_nowait()
                except queue.Empty:
                    break
                except (OSError, EOFError):
                    return

            q_deg = list(latest.get("q_deg", [0.0] * 5))
            rail_mm = float(latest.get("rail_mm", 0.0))
            gripper_state = str(latest.get("gripper_state", "OPEN"))
            phase = str(latest.get("phase", ""))
            busy = bool(latest.get("busy", False))

            if rail_jid is not None:
                pb.resetJointState(robot, rail_jid, rail_mm / 1000.0)

            q_rad = [math.radians(float(v)) for v in q_deg]
            q_rad += [0.0] * max(0, len(revolute) - len(q_rad))
            for index, joint_id in enumerate(revolute):
                pb.resetJointState(robot, joint_id, q_rad[index])

            for key, pos in dict(latest.get("markers", {})).items():
                if key in marker_ids and pos is not None:
                    pb.resetBasePositionAndOrientation(
                        marker_ids[key],
                        [float(v) for v in pos],
                        [0, 0, 0, 1],
                    )

            pb.stepSimulation()

            # Buttons
            for key, button_id in buttons.items():
                try:
                    value = pb.readUserDebugParameter(button_id)
                except Exception:
                    continue
                if value > button_values[key]:
                    button_values[key] = value
                    try:
                        command_queue.put_nowait(("command", key))
                    except queue.Full:
                        pass

            # Keyboard
            try:
                keyboard = pb.getKeyboardEvents()
            except Exception:
                keyboard = {}
            for char, command in key_map.items():
                if keyboard.get(ord(char), 0) & pb.KEY_WAS_TRIGGERED:
                    try:
                        command_queue.put_nowait(("command", command))
                    except queue.Full:
                        pass

            now = time.monotonic()
            if now - last_text_update >= 0.15:
                last_text_update = now
                lines = {
                    "phase": (
                        "PHASE: %s | busy=%s" % (phase, busy),
                        [1.0, 1.0, 0.2],
                        1.35,
                    ),
                    "pose": (
                        "A(deg): %s" % [round(float(v), 1) for v in q_deg],
                        [0.4, 0.9, 1.0],
                        1.20,
                    ),
                    "rail": (
                        "RAIL: %.1f mm | CENTRIC: %s" % (rail_mm, gripper_state),
                        [0.4, 1.0, 0.4],
                        1.05,
                    ),
                }
                for key, (line, color, height) in lines.items():
                    previous = debug_text.get(key)
                    if previous is not None:
                        try:
                            pb.removeUserDebugItem(previous)
                        except Exception:
                            pass
                    debug_text[key] = pb.addUserDebugText(
                        line,
                        [0, 0, height],
                        textColorRGB=color,
                        textSize=1.1,
                    )

                holes_text = str(latest.get("holes_text", ""))
                if holes_text:
                    if holes_text_id is not None:
                        try:
                            pb.removeUserDebugItem(holes_text_id)
                        except Exception:
                            pass
                    holes_text_id = pb.addUserDebugText(
                        holes_text,
                        [0.0, 0.0, 0.9],
                        textColorRGB=[1.0, 1.0, 1.0],
                        textSize=1.0,
                    )

            next_tick += period
            sleep_time = next_tick - time.monotonic()
            if sleep_time > 0.0:
                time.sleep(sleep_time)
            elif sleep_time < -period:
                next_tick = time.monotonic()

    except Exception as exc:
        try:
            command_queue.put_nowait(("viewer_error", "%s" % exc))
        except Exception:
            pass
    finally:
        if client is not None:
            try:
                pb.disconnect(client)
            except Exception:
                pass


def _collision_worker(config, request_queue, response_queue, stop_event):
    """Collision checking process, isolated from the Dynamixel process."""
    checker = None
    try:
        try:
            os.nice(5)
        except Exception:
            pass

        try:
            from .path_collision_guard_ee_5axis import (
                CollisionChecker as WorkerCollisionChecker,
            )
        except ImportError:
            from path_collision_guard_ee_5axis import (
                CollisionChecker as WorkerCollisionChecker,
            )

        checker = WorkerCollisionChecker(
            str(config["arm_urdf_path"]),
            str(config["pkg_dir"]),
            int(config["n_joints"]),
            str(config["mesh_dir"]),
        )

        while not stop_event.is_set():
            try:
                request = request_queue.get(timeout=0.10)
            except queue.Empty:
                continue
            except (OSError, EOFError):
                break

            if request is None:
                break

            request_id, kind, payload = request
            try:
                arr = np.asarray(payload, dtype=float)
                if kind == "config":
                    result = checker.check_config(arr)
                elif kind == "path":
                    result = checker.check_path(arr)
                else:
                    raise ValueError("unknown collision request: %s" % kind)
                response_queue.put((request_id, True, result))
            except Exception as exc:
                response_queue.put((request_id, False, "%s" % exc))
    finally:
        if checker is not None:
            try:
                checker.close()
            except Exception:
                pass


class SequenceStopped(RuntimeError):
    pass


class CentricSequenceNode(Node):
    NODE_NAME = "centric_sequence_5axis"
    STATE_TOPIC = "/centric_sequence/state"
    COMMAND_TOPIC = "/centric_sequence/command"

    def __init__(self):
        super().__init__(self.NODE_NAME)

        d = self.declare_parameter
        d("use_robot", False)
        d("use_rail", False)
        d("use_centric_gripper", False)
        d("use_simulation", True)

        d("pkg_dir", PACKAGE_SHARE)
        d("mesh_dir", DEFAULT_MESH_DIR)
        d("urdf_name", URDF_NAME)
        d("sim_hz", 60.0)
        d("sim_joint_speed_deg_s", 25.0)
        d("sim_rail_speed_mm_s", 100.0)

        d("dxl_port", "/dev/ttyUSB0")
        d("dxl_baud", 2_000_000)
        d("centric_id", 8)
        d("centric_capture_open_on_start", True)
        d("centric_configured_open_position", 506)
        d("centric_planet_max_angle_deg", 170.0)
        d("centric_auto_load_threshold_percent", 20.0)
        d("centric_auto_load_count_limit", 8)
        d("centric_auto_grasp_timeout_sec", 10.0)
        d("centric_emergency_load_percent", 35.0)
        d("centric_communication_retry_count", 5)

        d("rail_port", "/dev/ttyACM0")
        d("rail_baud", 115200)
        d("rail_auto_zero", True)
        d("rail_target_mm", 200.0)
        d("rail_arrival_tol_mm", 2.0)
        d("rail_arrival_timeout_sec", 60.0)

        d("grasp_pose_deg", [0.0, -12.0, -132.0, -35.0, 0.0])
        d("home_pose_deg", [0.0, 0.0, 0.0, 0.0, 0.0])

        # ============================================================
        # A-arm mechanical-reference homing
        # ============================================================
        # 실물 로봇을 아래 기계적 기준 자세에 수동으로 위치시킨 뒤
        # `refhome` 명령을 실행한다.
        #
        # J1/J5는 360도 회전이라 기계적 끝점이 없으므로 반드시
        # 조립/마킹 기준으로 0도에 맞춘 상태여야 한다.
        #
        # J2 = +56.7312 deg  (기계적 +방향 끝점)
        # J3 = -134.3335 deg (기계적 -방향 끝점)
        # J4 = +108.6579 deg (기계적 +방향 끝점)
        d(
            "reference_pose_deg",
            [0.0, 56.7312, -134.3335, 108.6579, 0.0],
        )
        d("require_reference_home", True)
        d("reference_home_speed_deg_s", 8.0)

        d("joint_arrival_tol_deg", 1.5)
        d("joint_arrival_timeout_sec", 25.0)
        d("auto_start", False)

        g = self.get_parameter
        self.use_robot = bool(g("use_robot").value)
        self.use_rail = bool(g("use_rail").value)
        self.use_gripper = bool(g("use_centric_gripper").value)
        self.use_simulation = bool(g("use_simulation").value)
        self.pkg_dir = str(g("pkg_dir").value)
        self.mesh_dir = str(g("mesh_dir").value)
        self.urdf_name = str(g("urdf_name").value)
        self.sim_hz = max(10.0, float(g("sim_hz").value))
        self.sim_joint_speed = max(
            1.0, float(g("sim_joint_speed_deg_s").value)
        )
        self.sim_rail_speed = max(
            1.0, float(g("sim_rail_speed_mm_s").value)
        )
        self.rail_target_mm = float(g("rail_target_mm").value)
        self.rail_tol_mm = max(0.1, float(g("rail_arrival_tol_mm").value))
        self.rail_timeout = max(1.0, float(g("rail_arrival_timeout_sec").value))
        self.joint_tol_rad = math.radians(
            max(0.1, float(g("joint_arrival_tol_deg").value))
        )
        self.joint_timeout = max(
            1.0, float(g("joint_arrival_timeout_sec").value)
        )
        self.grasp_pose_deg = self._pose_parameter("grasp_pose_deg")
        self.home_pose_deg = self._pose_parameter("home_pose_deg")

        self.reference_pose_deg = self._pose_parameter("reference_pose_deg")
        self.require_reference_home = bool(
            g("require_reference_home").value
        )
        self.reference_home_speed = max(
            0.5,
            float(g("reference_home_speed_deg_s").value),
        )

        # simulation에는 별도 기준점 캡처가 필요 없다.
        # 실물 로봇은 refcapture/refhome 실행 전까지 모든 A-arm 공정을 막는다.
        self.reference_ready = not self.use_robot

        min_mm = RAIL_MIN_M * 1000.0
        max_mm = RAIL_MAX_M * 1000.0
        if not min_mm <= self.rail_target_mm <= max_mm:
            raise ValueError(
                "rail_target_mm %.1f is outside [%.1f, %.1f]"
                % (self.rail_target_mm, min_mm, max_mm)
            )
        self.bridge = None
        self.rail = None
        self.gripper = None
        self.busy = False
        self.phase = "INITIALIZING"
        self._state_lock = threading.Lock()
        self._stop_event = threading.Event()
        self.sim_q_deg = list(self.home_pose_deg)
        self.sim_rail_mm = 0.0
        self.sim_gripper_state = "OPEN"
        # PyBullet is intentionally isolated in another OS process.
        # The real-robot process never calls PyBullet GUI APIs.
        self._sim_client = None
        self.robot = None
        self.rev = []
        self.rail_jid = None
        self._button = {}
        self._button_value = {}
        self._debug_text = {}
        self._last_render_text = 0.0

        self._mp_ctx = mp.get_context("spawn")
        self._viewer_process = None
        self._viewer_state_queue = None
        self._viewer_command_queue = None
        self._viewer_stop_event = None
        self._viewer_pending_commands = set()
        self._viewer_markers = {}
        self._viewer_holes_text = ""
        self._last_viewer_send = 0.0
        self._viewer_send_period = 1.0 / 15.0

        self._setup_simulation()
        self._setup_hardware(g)
        self.state_pub = self.create_publisher(
            String, self.STATE_TOPIC, 10
        )
        self.create_subscription(
            String, self.COMMAND_TOPIC, self._on_command, 10
        )
        self.create_timer(0.05, self._tick)
        self._start_terminal_thread()
        self._set_phase("READY")
        self._print_help()

        if bool(g("auto_start").value):
            self.create_timer(1.0, self._auto_start_once)
        self._auto_started = False

    def _pose_parameter(self, name):
        values = [float(v) for v in self.get_parameter(name).value]
        if len(values) != 5 or not all(math.isfinite(v) for v in values):
            raise ValueError("%s must contain five finite joint angles" % name)
        return values

    def _find_urdf(self, name):
        candidates = [
            os.path.join(self.pkg_dir, "urdf", name),
            os.path.join(self.pkg_dir, name),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), name),
        ]
        for path in candidates:
            if os.path.isfile(path):
                return path
        raise FileNotFoundError(
            "URDF not found: " + " | ".join(candidates)
        )

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

            if os.path.isfile(candidate):
                return "filename=%s%s%s" % (
                    quote,
                    candidate,
                    quote,
                )
            return match.group(0)

        return re.sub(
            r"filename=([\"'])(.*?)(?:\1)",
            replace,
            text,
        )

    def _prepare_urdf(self, name):
        source = self._find_urdf(name)

        with open(source, encoding="utf-8") as stream:
            text = self._rewrite_mesh_paths(stream.read())

        text = text.replace(
            "package://robot_description/",
            self.pkg_dir.rstrip("/") + "/",
        )
        text = text.replace(
            "package://vision_pkg/",
            self.pkg_dir.rstrip("/") + "/",
        )

        output = os.path.join(
            tempfile.gettempdir(),
            "centric_sequence_" + os.path.basename(name),
        )

        with open(output, "w", encoding="utf-8") as stream:
            stream.write(text)

        return output

    # ------------------------------------------------------------- simulation
    def _setup_simulation(self):
        """Start PyBullet GUI in a separate OS process.

        No p.connect(), p.stepSimulation(), button polling, or rendering runs in
        the ROS/Dynamixel process.  This is the key isolation boundary.
        """
        if not self.use_simulation:
            return

        # Prepare the URDF here (file I/O only; no PyBullet call).
        urdf_path = self._prepare_urdf(self.urdf_name)

        self._viewer_state_queue = self._mp_ctx.Queue(maxsize=1)
        self._viewer_command_queue = self._mp_ctx.Queue(maxsize=64)
        self._viewer_stop_event = self._mp_ctx.Event()

        config = {
            "urdf_path": urdf_path,
            # Keep GUI smooth enough for monitoring without wasting CPU.
            "viewer_hz": min(30.0, max(10.0, float(self.sim_hz))),
        }

        self._viewer_process = self._mp_ctx.Process(
            target=_pybullet_viewer_worker,
            args=(
                config,
                self._viewer_state_queue,
                self._viewer_command_queue,
                self._viewer_stop_event,
            ),
            name="centric_pybullet_viewer",
            daemon=True,
        )
        self._viewer_process.start()

        self.get_logger().warning(
            "PyBullet GUI isolated in separate process pid=%s; "
            "real Dynamixel process does not call PyBullet APIs."
            % self._viewer_process.pid
        )

        self._send_viewer_state(force=True)

    def _stop_simulation_process(self):
        if self._viewer_stop_event is not None:
            try:
                self._viewer_stop_event.set()
            except Exception:
                pass

        if self._viewer_process is not None:
            try:
                self._viewer_process.join(timeout=2.0)
            except Exception:
                pass

            if self._viewer_process.is_alive():
                try:
                    self._viewer_process.terminate()
                    self._viewer_process.join(timeout=1.0)
                except Exception:
                    pass

        self._viewer_process = None

        for q in (
            self._viewer_state_queue,
            self._viewer_command_queue,
        ):
            if q is not None:
                try:
                    q.close()
                except Exception:
                    pass

        self._viewer_state_queue = None
        self._viewer_command_queue = None
        self._viewer_stop_event = None

    def _send_viewer_state(self, force=False):
        if (
            not self.use_simulation
            or self._viewer_state_queue is None
        ):
            return

        now = time.monotonic()
        if (not force) and (
            now - self._last_viewer_send < self._viewer_send_period
        ):
            return
        self._last_viewer_send = now

        with self._state_lock:
            payload = {
                "q_deg": list(self.sim_q_deg),
                "rail_mm": float(self.sim_rail_mm),
                "gripper_state": str(self.sim_gripper_state),
                "phase": str(self.phase),
                "busy": bool(self.busy),
                "markers": {
                    key: list(value)
                    for key, value in self._viewer_markers.items()
                },
                "holes_text": str(self._viewer_holes_text),
            }

        # Never block the real robot process for visualization.
        _queue_put_latest(self._viewer_state_queue, payload)

    def _poll_viewer_commands(self):
        if (
            not self.use_simulation
            or self._viewer_command_queue is None
        ):
            return

        for _ in range(64):
            try:
                kind, value = self._viewer_command_queue.get_nowait()
            except queue.Empty:
                break
            except (OSError, EOFError):
                break

            if kind == "command":
                self._viewer_pending_commands.add(str(value))
            elif kind == "viewer_error":
                self.get_logger().error(
                    "isolated PyBullet viewer error: %s" % value
                )

    def _button_clicked(self, key):
        # Button/key events arrive through IPC from the isolated GUI process.
        self._poll_viewer_commands()
        key = str(key)
        if key in self._viewer_pending_commands:
            self._viewer_pending_commands.remove(key)
            return True
        return False

    def _set_sim_state(self, q_deg=None, rail_mm=None, gripper=None):
        with self._state_lock:
            if q_deg is not None:
                self.sim_q_deg = [float(value) for value in q_deg]
            if rail_mm is not None:
                self.sim_rail_mm = float(rail_mm)
            if gripper is not None:
                self.sim_gripper_state = str(gripper)
        self._send_viewer_state()

    def _render_sim(self, force_text=False):
        # Rendering happens only in the child process.
        self._send_viewer_state()

    def _tick(self):
        # This timer only polls tiny IPC messages.  It never calls PyBullet.
        self._poll_viewer_commands()

        if self._button_clicked("stop"):
            self._emergency_stop("PyBullet EMERGENCY STOP")
            self._set_phase("STOPPED_READY")
        if self._button_clicked("home"):
            self._start_auxiliary("GOING_HOME", self._home_only)
        if self._button_clicked("rail_home"):
            self._start_auxiliary("RAIL_RETURN_HOME", self._rail_home_only)
        if self._button_clicked("all_initial"):
            self._start_auxiliary("ALL_RETURN_INITIAL", self._all_initial)
        if self._button_clicked("open"):
            self._start_auxiliary("OPENING", self._open_gripper)
        if self._button_clicked("grasp"):
            self._start_auxiliary("GRASPING", self._grasp_only)
        if self._button_clicked("motors"):
            self._start_auxiliary("READING_MOTOR_STATUS", self._log_motor_status)
        self._send_viewer_state()

    @staticmethod
    def _key_triggered(events, key):
        # Keyboard handling is performed inside the isolated viewer process.
        return False

    # --------------------------------------------------------------- hardware
    def _setup_hardware(self, g):
        try:
            if self.use_rail:
                self.rail = RailSerial(
                    port=str(g("rail_port").value),
                    baud=int(g("rail_baud").value),
                    logger=self.get_logger(),
                )
                self.rail.connect()
                if bool(g("rail_auto_zero").value):
                    if not self.rail.zero(wait=True, timeout=2.0):
                        raise RuntimeError("rail zero acknowledgement timeout")
                if not self.rail.zero_set:
                    raise RuntimeError("rail zero is not set")

            if self.use_robot:
                self.bridge = DxlBridge(
                    port=str(g("dxl_port").value),
                    baud=int(g("dxl_baud").value),
                )
                self.bridge.connect()

                # 기존처럼 "프로그램 실행 순간의 자세 = q=0"으로 잡지 않는다.
                # 실물을 기계적 기준 자세에 둔 뒤 refhome/refcapture로
                # zero_tick을 계산한다.
                if self.require_reference_home:
                    self.reference_ready = False
                    self.get_logger().warning(
                        "A-arm zero NOT captured. "
                        "옵션 1: 실물을 reference pose %s deg에 둔 뒤 "
                        "`MECH LIMIT -> STRAIGHT HOME` 버튼. "
                        "옵션 2: 현재 자세가 이미 1자라면 "
                        "`CURRENT POSE = STRAIGHT HOME` 버튼. "
                        "J1/J5는 기계적 끝점이 없으므로 옵션 1에서는 0도 마크에 맞춰야 합니다."
                        % self.reference_pose_deg
                    )
                else:
                    # 필요할 때만 예전 startup capture 방식으로 되돌릴 수 있다.
                    self.bridge.capture_home()
                    self.reference_ready = True
                    self.get_logger().warning(
                        "require_reference_home=false: 현재 자세를 "
                        "[0,0,0,0,0] deg HOME으로 저장했습니다."
                    )

            if self.use_gripper:
                self.gripper = CentricGripper(
                    port=str(g("dxl_port").value),
                    baud=int(g("dxl_baud").value),
                    dxl_id=int(g("centric_id").value),
                    logger=self.get_logger(),
                    planet_max_angle_deg=float(
                        g("centric_planet_max_angle_deg").value
                    ),
                    auto_load_threshold_percent=float(
                        g("centric_auto_load_threshold_percent").value
                    ),
                    auto_load_count_limit=int(
                        g("centric_auto_load_count_limit").value
                    ),
                    auto_grasp_timeout_sec=float(
                        g("centric_auto_grasp_timeout_sec").value
                    ),
                    emergency_load_percent=float(
                        g("centric_emergency_load_percent").value
                    ),
                    communication_retry_count=int(
                        g("centric_communication_retry_count").value
                    ),
                )
                connect_kwargs = dict(
                    capture_open_on_start=bool(
                        g("centric_capture_open_on_start").value
                    ),
                    configured_open_position=int(
                        g("centric_configured_open_position").value
                    ),
                )
                if self.bridge is not None:
                    self.gripper.connect_shared(
                        self.bridge.ph,
                        self.bridge.pk,
                        self.bridge.bus_lock,
                        **connect_kwargs,
                    )
                    self.get_logger().info(
                        "A-arm ID 1~7 and centric ID 8 share one U2D2: %s"
                        % str(g("dxl_port").value)
                    )
                else:
                    # Gripper-only real test: this node owns the same U2D2 port.
                    self.gripper.connect(**connect_kwargs)
                if bool(g("centric_capture_open_on_start").value):
                    self.get_logger().warning(
                        "Centric startup position was captured as 0% OPEN; verify it was fully open"
                    )
        except Exception:
            self._close_hardware()
            self._stop_simulation_process()
            raise

    def _close_hardware(self):
        if self.rail is not None:
            try:
                self.rail.estop()
                self.rail.close()
            except Exception:
                pass
            self.rail = None
        if self.gripper is not None:
            try:
                # Detach ID 8 before the A-arm owner closes the shared port.
                self.gripper.close(torque_off=False)
            except Exception:
                pass
            self.gripper = None
        if self.bridge is not None:
            try:
                # Keep A-arm torque enabled so it does not suddenly go limp.
                self.bridge.close(torque_off=False)
            except Exception:
                pass
            self.bridge = None

    # ------------------------------------------------------ reference homing
    def _assert_reference_ready(self):
        if (
            self.bridge is not None
            and self.require_reference_home
            and not self.reference_ready
        ):
            raise RuntimeError(
                "A-arm reference is not calibrated. "
                "PyBullet의 `MECH LIMIT -> STRAIGHT HOME` 버튼으로 "
                "기계적 기준에서 1자 자세를 만들거나, "
                "`CURRENT POSE = STRAIGHT HOME` 버튼으로 현재 자세를 "
                "[0,0,0,0,0] 기준으로 등록하세요."
            )

    def _mechanical_reference_to_straight(self):
        """Mechanical-limit reference pose -> straight HOME in one button.

        Expected physical reference pose:
          [0, +56.7312, -134.3335, +108.6579, 0] deg

        J1/J5 have no mechanical stop, so they must already be aligned to the
        physical 0-degree marks before this command is used.
        """
        self.get_logger().warning(
            "MECH->STRAIGHT START: 현재 실물이 반드시 %s deg 기계적 기준 자세여야 합니다. "
            "J1/J5는 0도 마크에 맞춰져 있어야 합니다."
            % self.reference_pose_deg
        )
        self._capture_reference_pose()
        self._reference_home()
        self.get_logger().warning(
            "MECH->STRAIGHT COMPLETE: A-arm straight HOME=%s deg"
            % self.home_pose_deg
        )

    def _capture_current_as_straight(self):
        """Set the present physical pose as straight HOME without moving.

        Whatever pose the A-arm is physically holding becomes
        q=[0,0,0,0,0].  IK will be geometrically correct only when the
        physical pose is actually the intended straight pose.
        """
        if self.bridge is None:
            self.reference_ready = True
            self._set_sim_state(q_deg=self.home_pose_deg)
            self.get_logger().warning(
                "[SIM] CURRENT POSE registered as STRAIGHT HOME %s deg"
                % self.home_pose_deg
            )
            return

        self._check_stop()
        self.get_logger().warning(
            "CURRENT->STRAIGHT START: 현재 실물 자세를 움직이지 않고 "
            "[0,0,0,0,0] deg HOME으로 등록합니다."
        )

        self.bridge.capture_home()
        self.reference_ready = True

        actual = self.bridge.read_joints()
        if len(actual) < 5:
            self.reference_ready = False
            raise RuntimeError(
                "CURRENT->STRAIGHT verification failed: "
                "A-arm returned fewer than five joint states"
            )

        actual_deg = [math.degrees(v) for v in actual[:5]]
        max_error = max(abs(v) for v in actual_deg)
        self._set_sim_state(q_deg=self.home_pose_deg)

        self.get_logger().warning(
            "CURRENT->STRAIGHT COMPLETE: current physical pose is now "
            "A=[0,0,0,0,0] deg | verification=%s deg | max %.4f deg"
            % ([round(v, 4) for v in actual_deg], max_error)
        )

        if max_error > 0.25:
            self.reference_ready = False
            raise RuntimeError(
                "CURRENT->STRAIGHT verification failed: max %.4f deg"
                % max_error
            )

    def _capture_reference_pose(self):
        """현재 실물 자세를 known mechanical-reference pose로 등록한다.

        reference_pose_deg의 실제 관절각과 각 Dynamixel Present Position을
        이용하여 q=0에 해당하는 zero_tick을 역산한다.

        zero_tick =
            present_tick
            - dir * pair_sign * ticks(reference_joint_angle)

        J3/J4처럼 모터가 두 개 달린 관절도 각 모터의 현재 위치를 따로
        읽어 각각의 zero_tick을 계산한다.
        """
        if self.bridge is None:
            self.reference_ready = True
            self._set_sim_state(q_deg=self.reference_pose_deg)
            self.get_logger().info(
                "[SIM] reference captured: %s deg" % self.reference_pose_deg
            )
            return

        reference_rad = _deg_to_rad(self.reference_pose_deg)
        present_ticks = {}
        new_zero = {}

        self.get_logger().warning(
            "REFERENCE CAPTURE START: 현재 실물 자세를 %s deg로 간주합니다."
            % self.reference_pose_deg
        )

        # 모든 모터의 현재 위치를 먼저 읽고 zero_tick을 계산한다.
        for joint_index, joint in enumerate(JOINT_MOTORS[:5]):
            ref_q = float(reference_rad[joint_index])
            ref_ticks = float(self.bridge._r2t(ref_q, joint["gear"]))

            for motor_index, motor_id in enumerate(joint["ids"]):
                present = int(self.bridge._present(motor_id))
                sign = float(joint["dir"] * joint["pair_sign"][motor_index])
                zero = int(round(present - sign * ref_ticks))

                present_ticks[motor_id] = present
                new_zero[motor_id] = zero

        # 계산이 모두 끝난 뒤 한 번에 적용한다.
        for motor_id, zero in new_zero.items():
            self.bridge.zero_tick[motor_id] = int(zero)

        self.reference_ready = True

        self.get_logger().info(
            "REFERENCE present ticks: %s" % present_ticks
        )
        self.get_logger().info(
            "CALCULATED q=0 zero ticks: %s" % new_zero
        )

        # 계산 직후 read_joints()가 기준각을 재현하는지 확인한다.
        q_check = [
            math.degrees(v)
            for v in self.bridge.read_joints()[:5]
        ]
        max_err = max(
            abs(a - b)
            for a, b in zip(q_check, self.reference_pose_deg)
        )
        self.get_logger().info(
            "REFERENCE verification [deg] = %s | max error %.4f deg"
            % ([round(v, 4) for v in q_check], max_err)
        )

        if max_err > 0.25:
            self.reference_ready = False
            raise RuntimeError(
                "reference verification failed: max %.4f deg" % max_err
            )

        # 실물 reference가 확정되었으므로 PyBullet 표시도 같은 자세로 맞춘다.
        self._set_sim_state(q_deg=self.reference_pose_deg)

        self.get_logger().warning(
            "REFERENCE CAPTURE COMPLETE. "
            "이제 `refhome`을 입력하면 재캡처 없이 바로 "
            "q=[0,0,0,0,0] HOME으로 이동합니다."
        )

    def _reference_home(self):
        """저장된 reference zero_tick을 이용해 즉시 HOME goal을 전송한다.

        중요:
        - refcapture를 다시 하지 않는다.
        - read_joints()를 먼저 하지 않는다.
        - joint trajectory를 미리 생성하지 않는다.
        - collision path pre-check를 하지 않는다.
        - HOME용 저속 Profile Velocity를 먼저 설정한 뒤
          q=[0,0,0,0,0] Goal Position을 즉시 GroupSyncWrite로 전송한다.
        - Goal을 보낸 뒤에만 Present Position을 읽어 도착 여부를 확인한다.

        따라서 refhome 입력 후 실제 로봇이 움직이기 시작하기 전의
        수십 초 지연을 제거한다.
        """
        if not self.reference_ready:
            raise RuntimeError(
                "reference가 아직 설정되지 않았습니다. "
                "실물을 [0, 56.7312, -134.3335, 108.6579, 0] deg "
                "기계적 기준 자세에 둔 뒤 `refcapture`를 먼저 실행하세요."
            )

        # Simulation only
        if self.bridge is None:
            self.get_logger().warning(
                "[SIM] REFERENCE HOME DIRECT -> %s deg"
                % self.home_pose_deg
            )
            self._animate_sim_arm(self.home_pose_deg)
            self._set_sim_state(q_deg=self.home_pose_deg)
            self.get_logger().warning(
                "[SIM] REFERENCE HOME COMPLETE"
            )
            return

        self._check_stop()

        # --------------------------------------------------------
        # HOME 전용 Profile Velocity 설정
        #
        # Dynamixel X-series:
        # motor speed [deg/s] ~= PV * 0.229[rpm] * 6
        # joint speed = motor speed / gear
        #
        # 따라서:
        # PV ~= desired_joint_deg_s * gear / (0.229 * 6)
        #
        # 기존 PROFILE_VELOCITY보다 빨라지지 않게 제한한다.
        # --------------------------------------------------------
        profile_velocity_addr = 112
        velocity_unit_deg_s = 0.229 * 6.0  # 1.374 deg/s per PV at motor shaft

        home_profile = {}

        for joint in JOINT_MOTORS[:5]:
            requested_pv = int(round(
                self.reference_home_speed
                * float(joint["gear"])
                / velocity_unit_deg_s
            ))
            requested_pv = max(1, requested_pv)

            for motor_id in joint["ids"]:
                normal_pv = int(PROFILE_VELOCITY[motor_id])
                home_profile[motor_id] = min(normal_pv, requested_pv)

        self.get_logger().warning(
            "REFERENCE HOME DIRECT START: "
            "saved zero_tick -> %s deg | requested speed <= %.1f deg/s"
            % (self.home_pose_deg, self.reference_home_speed)
        )

        try:
            # 1) HOME 전용 저속 속도 설정
            for motor_id, pv in home_profile.items():
                self.bridge._w4(
                    motor_id,
                    profile_velocity_addr,
                    int(pv),
                )

            self.get_logger().info(
                "REFERENCE HOME profile velocity: %s"
                % home_profile
            )

            # 2) 핵심: 현재 위치 read / trajectory / collision 검사 없이
            #    즉시 q=0 Goal Position 전송
            goal_rad = _deg_to_rad(self.home_pose_deg)

            send_started = time.monotonic()

            self.bridge.command_joints(goal_rad)

            self.get_logger().warning(
                "REFERENCE HOME GOAL SENT IMMEDIATELY "
                "(%.3f s after direct-home send start)"
                % (time.monotonic() - send_started)
            )

            # PyBullet 목표 상태도 즉시 HOME으로 갱신
            # 실제 상태는 아래 read_joints()에서 계속 동기화한다.
            self._set_sim_state(q_deg=self.home_pose_deg)

            # ----------------------------------------------------
            # 3) Goal 전송 후에만 도착 여부 확인
            #
            # 통신 retry가 있더라도 실제 모터는 이미 움직이고 있으므로
            # 로봇의 이동 시작을 지연시키지 않는다.
            # ----------------------------------------------------
            max_reference_delta = max(
                abs(float(a) - float(b))
                for a, b in zip(
                    self.reference_pose_deg,
                    self.home_pose_deg,
                )
            )
            expected_motion_sec = (
                max_reference_delta / self.reference_home_speed
            )
            monitor_timeout = max(
                float(self.joint_timeout),
                expected_motion_sec + 30.0,
            )

            deadline = time.monotonic() + monitor_timeout
            consecutive_read_failures = 0

            while time.monotonic() < deadline:
                self._check_stop()

                try:
                    actual = self.bridge.read_joints()
                    consecutive_read_failures = 0
                except Exception as exc:
                    consecutive_read_failures += 1

                    # 이미 HOME goal은 전송됐기 때문에 읽기 실패가
                    # 로봇 이동 자체를 막지는 않는다.
                    if consecutive_read_failures == 1 or (
                        consecutive_read_failures % 5 == 0
                    ):
                        self.get_logger().warning(
                            "REFERENCE HOME monitoring read failed "
                            "(robot goal already sent): %s"
                            % exc
                        )

                    time.sleep(0.15)
                    continue

                if len(actual) < 5:
                    time.sleep(0.15)
                    continue

                actual_deg = [
                    math.degrees(value)
                    for value in actual[:5]
                ]
                self._set_sim_state(q_deg=actual_deg)

                max_error = max(
                    abs(_wrapped_error(present, goal))
                    for present, goal in zip(
                        actual[:5],
                        goal_rad,
                    )
                )

                if max_error <= self.joint_tol_rad:
                    self._set_sim_state(q_deg=actual_deg)
                    self.get_logger().warning(
                        "REFERENCE HOME COMPLETE: "
                        "A=%s deg | max error %.3f deg"
                        % (
                            [round(v, 3) for v in actual_deg],
                            math.degrees(max_error),
                        )
                    )
                    return

                time.sleep(0.15)

            raise RuntimeError(
                "REFERENCE HOME arrival timeout. "
                "HOME goal was sent immediately, but arrival confirmation "
                "did not finish within %.1f s."
                % monitor_timeout
            )

        finally:
            # ----------------------------------------------------
            # 4) 기존 공정용 Profile Velocity 복원
            # ----------------------------------------------------
            restore_errors = []

            for motor_id, pv in PROFILE_VELOCITY.items():
                try:
                    self.bridge._w4(
                        motor_id,
                        profile_velocity_addr,
                        int(pv),
                    )
                except Exception as exc:
                    restore_errors.append(
                        "ID%d:%s" % (motor_id, exc)
                    )

            if restore_errors:
                self.get_logger().warning(
                    "PROFILE_VELOCITY restore warnings: %s"
                    % ", ".join(restore_errors)
                )
            else:
                self.get_logger().info(
                    "A-arm normal PROFILE_VELOCITY restored"
                )

    def _log_reference_status(self):
        if self.bridge is None:
            self.get_logger().info(
                "REFERENCE STATUS: simulation | ready=%s | ref=%s"
                % (self.reference_ready, self.reference_pose_deg)
            )
            return

        if not self.reference_ready:
            raw = {}
            for joint in JOINT_MOTORS[:5]:
                for motor_id in joint["ids"]:
                    raw[motor_id] = int(self.bridge._present(motor_id))
            self.get_logger().warning(
                "REFERENCE STATUS: NOT CALIBRATED | raw ticks=%s | "
                "option A: place robot at %s deg then mechhome | "
                "option B: if already straight, run herehome"
                % (raw, self.reference_pose_deg)
            )
            return

        q_deg = [
            math.degrees(v)
            for v in self.bridge.read_joints()[:5]
        ]
        self.get_logger().info(
            "REFERENCE STATUS: READY | A=%s deg | zero_tick=%s"
            % (
                [round(v, 4) for v in q_deg],
                dict(self.bridge.zero_tick),
            )
        )

    # --------------------------------------------------------------- commands
    def _auto_start_once(self):
        if not self._auto_started:
            self._auto_started = True
            self._start_sequence()

    def _on_command(self, msg):
        self._handle_command(msg.data)

    def _handle_command(self, raw):
        parts = str(raw).strip().lower().split()
        if not parts:
            return
        command = parts[0]
        if command in ("mechhome", "mechstraight", "limit2home"):
            self._start_auxiliary(
                "MECH_REFERENCE_TO_STRAIGHT_HOME",
                self._mechanical_reference_to_straight,
            )
        elif command in ("herehome", "currenthome", "currentstraight"):
            self._start_auxiliary(
                "CURRENT_POSE_SET_AS_STRAIGHT_HOME",
                self._capture_current_as_straight,
            )
        elif command in ("refhome", "referencehome", "calhome"):
            self._start_auxiliary(
                "REFERENCE_HOME_DIRECT",
                self._reference_home,
            )
        elif command in ("refcapture", "referencecapture", "calref"):
            self._start_auxiliary(
                "REFERENCE_CAPTURE_ONLY",
                self._capture_reference_pose,
            )
        elif command in ("refstatus", "reference"):
            self._start_auxiliary(
                "REFERENCE_STATUS",
                self._log_reference_status,
            )
        elif command in ("start", "run"):
            self._start_sequence()
        elif command in ("stop", "estop", "0", "s"):
            self._emergency_stop("user command")
            self._set_phase("STOPPED_READY")
        elif command == "status":
            self._log_status()
        elif command in ("motors", "motor"):
            self._start_auxiliary(
                "READING_MOTOR_STATUS", self._log_motor_status
            )
        elif command == "open":
            self._start_auxiliary("OPENING", self._open_gripper)
        elif command == "grasp":
            self._start_auxiliary("GRASPING", self._grasp_only)
        elif command == "home":
            self._start_auxiliary("GOING_HOME", self._home_only)
        elif command in ("railhome", "rhome"):
            self._start_auxiliary("RAIL_RETURN_HOME", self._rail_home_only)
        elif command in ("allinitial", "allhome", "resetpose"):
            self._start_auxiliary("ALL_RETURN_INITIAL", self._all_initial)
        elif command == "rail" and len(parts) == 2:
            try:
                target_mm = float(parts[1])
            except ValueError:
                self.get_logger().error("rail target must be a number in mm")
                return
            self._start_rail_only(target_mm)
        else:
            self.get_logger().warning(
                "commands: mechhome | herehome | refhome | refcapture | "
                "refstatus | start | stop | status | open | grasp | home | "
                "motors | railhome | allinitial | rail <mm>"
            )

    def _start_rail_only(self, target_mm):
        min_mm = RAIL_MIN_M * 1000.0
        max_mm = RAIL_MAX_M * 1000.0
        target_mm = float(target_mm)
        if not min_mm <= target_mm <= max_mm:
            self.get_logger().error(
                "rail-only target %.1f mm is outside [%.1f, %.1f] mm"
                % (target_mm, min_mm, max_mm)
            )
            return
        self._start_auxiliary(
            "RAIL_ONLY_TO_%.0fMM" % target_mm,
            lambda: self._rail_only_to(target_mm),
        )

    def _rail_only_to(self, target_mm):
        self.get_logger().warning(
            "RAIL ONLY: A-arm will not move; check clearance before travel"
        )
        self._move_rail(float(target_mm))

    def _start_sequence(self):
        if (
            self.bridge is not None
            and self.require_reference_home
            and not self.reference_ready
        ):
            self.get_logger().error(
                "START BLOCKED: A-arm reference is not calibrated. "
                "`MECH LIMIT -> STRAIGHT HOME` 또는 "
                "`CURRENT POSE = STRAIGHT HOME` 버튼으로 기준을 먼저 설정하세요."
            )
            return

        with self._state_lock:
            if self.busy:
                self.get_logger().warning("sequence is already busy")
                return
            self.busy = True
            self._stop_event.clear()
        threading.Thread(target=self._run_sequence, daemon=True).start()

    def _start_auxiliary(self, phase, callback):
        with self._state_lock:
            if self.busy:
                self.get_logger().warning("sequence is busy")
                return
            self.busy = True
            self._stop_event.clear()

        def worker():
            try:
                self._set_phase(phase)
                callback()
                self._set_phase("READY")
            except SequenceStopped:
                self.get_logger().warning("motion stopped; node is still active")
                self._set_phase("STOPPED_READY")
            except Exception as exc:
                self._fail(str(exc))
            finally:
                with self._state_lock:
                    self.busy = False

        threading.Thread(target=worker, daemon=True).start()

    # --------------------------------------------------------------- sequence
    def _run_sequence(self):
        try:
            self._set_phase("RAIL_TO_TARGET")
            self._move_rail(self.rail_target_mm)

            self._set_phase("A_TO_GRASP_POSE")
            self._move_arm(self.grasp_pose_deg)

            self._set_phase("CENTRIC_GRASP")
            result = self._auto_grasp()
            if not result.success:
                raise RuntimeError("centric grasp failed: " + result.reason)

            self._set_phase("A_RETURN_HOME")
            self._move_arm(self.home_pose_deg)

            self._set_phase("RAIL_RETURN_HOME")
            self._move_rail(0.0)

            self._set_phase("DONE_HOLDING_OBJECT_AT_HOME")
            self.get_logger().info(
                "sequence complete: A-arm HOME, rail 0 mm, gripper holding"
            )
        except SequenceStopped:
            self.get_logger().warning(
                "process stopped at current position; node remains ready for commands"
            )
            self._set_phase("STOPPED_READY")
        except Exception as exc:
            self._fail(str(exc))
        finally:
            with self._state_lock:
                self.busy = False

    def _move_rail(self, target_mm):
        self._check_stop()
        clock = getattr(self, '_process_time', time.monotonic)
        target_mm = float(target_mm)
        if self.rail is None:
            if self.use_rail:
                raise RuntimeError("rail connection is unavailable")
            self._animate_sim_rail(target_mm)
            self.get_logger().info("[SIM] rail arrived -> %.1f mm" % target_mm)
            return

        # If the rail is already within the target tolerance, do not send a
        # zero-distance move.  Some Arduino firmware variants do not emit a
        # completion event for p 0 while already at 0 mm, which otherwise
        # leaves RailSerial.moving=True until this function times out.
        present_mm = float(self.rail.position_mm)
        error = abs(present_mm - target_mm)
        if not self.rail.moving and error <= self.rail_tol_mm:
            self._set_sim_state(rail_mm=present_mm)
            self.get_logger().info(
                "rail already at target: %.1f mm (error %.1f mm)"
                % (present_mm, error)
            )
            return

        if not self.rail.move_abs_mm(target_mm):
            raise RuntimeError("rail move command rejected")

        deadline = clock() + self.rail_timeout
        next_query = 0.0
        motion_epoch = getattr(self, '_motion_epoch', None)
        while clock() < deadline:
            self._check_stop()
            if motion_epoch != getattr(self, '_motion_epoch', None):
                # A paused rail was stopped by estop; reissue its outstanding
                # destination from the measured current position.
                if not self.rail.move_abs_mm(target_mm):
                    raise RuntimeError('rail resume command rejected')
                motion_epoch = getattr(self, '_motion_epoch', None)
            now = clock()
            if now >= next_query:
                self.rail.query()
                next_query = now + 0.5
            present_mm = float(self.rail.position_mm)
            self._set_sim_state(rail_mm=present_mm)
            error = abs(present_mm - target_mm)
            if not self.rail.moving and error <= self.rail_tol_mm:
                self._set_sim_state(rail_mm=present_mm)
                self.get_logger().info(
                    "rail arrived: %.1f mm (error %.1f mm)"
                    % (present_mm, error)
                )
                return
            time.sleep(0.05)
        raise RuntimeError(
            "rail arrival timeout: present %.1f mm, target %.1f mm"
            % (self.rail.position_mm, target_mm)
        )

    def _move_arm(self, goal_deg):
        self._check_stop()
        self._assert_reference_ready()
        if self.bridge is None:
            if self.use_robot:
                raise RuntimeError("A-arm connection is unavailable")
            self._animate_sim_arm(goal_deg)
            self.get_logger().info("[SIM] A-arm arrived -> %s deg" % list(goal_deg))
            return

        goal_rad = _deg_to_rad(goal_deg)
        self.bridge.command_joints(goal_rad)
        deadline = time.monotonic() + self.joint_timeout
        while time.monotonic() < deadline:
            self._check_stop()
            actual = self.bridge.read_joints()
            if len(actual) < 5:
                raise RuntimeError("A-arm returned fewer than five joint states")
            actual_deg = [math.degrees(value) for value in actual[:5]]
            self._set_sim_state(q_deg=actual_deg)
            max_error = max(
                abs(_wrapped_error(present, goal))
                for present, goal in zip(actual[:5], goal_rad)
            )
            if max_error <= self.joint_tol_rad:
                self._set_sim_state(q_deg=actual_deg)
                self.get_logger().info(
                    "A-arm arrived: %s deg (max error %.2f deg)"
                    % (list(goal_deg), math.degrees(max_error))
                )
                return
            time.sleep(0.10)
        raise RuntimeError("A-arm arrival timeout")

    def _auto_grasp(self):
        self._check_stop()
        self._set_sim_state(gripper="CLOSING")
        if self.gripper is None:
            if self.use_gripper:
                raise RuntimeError("centric gripper connection is unavailable")
            self._interruptible_wait(0.8)
            self._set_sim_state(gripper="HOLDING")
            self.get_logger().info("[SIM] centric auto grasp -> success")
            return GraspResult(True, "simulation")
        result = self.gripper.auto_grasp(self._stop_event.is_set)
        self._set_sim_state(
            gripper="HOLDING" if result.success else "GRASP FAILED"
        )
        return result

    def _open_gripper(self):
        self._check_stop()
        self._set_sim_state(gripper="OPENING")
        if self.gripper is None:
            if self.use_gripper:
                raise RuntimeError("centric gripper connection is unavailable")
            self._interruptible_wait(0.5)
            self._set_sim_state(gripper="OPEN")
            self.get_logger().info("[SIM] centric open")
            return
        if not self.gripper.open(self._stop_event.is_set):
            raise RuntimeError("centric open failed")
        self._set_sim_state(gripper="OPEN")

    def _grasp_only(self):
        result = self._auto_grasp()
        if not result.success:
            raise RuntimeError("centric grasp failed: " + result.reason)

    def _home_only(self):
        self._move_arm(self.home_pose_deg)

    def _rail_home_only(self):
        self._move_rail(0.0)

    def _interruptible_wait(self, seconds):
        deadline = time.monotonic() + max(0.0, float(seconds))
        while time.monotonic() < deadline:
            self._check_stop()
            time.sleep(0.02)

    def _animate_sim_rail(self, target_mm):
        with self._state_lock:
            start_mm = float(self.sim_rail_mm)
        if not self.use_simulation:
            self._set_sim_state(rail_mm=target_mm)
            return
        distance = abs(float(target_mm) - start_mm)
        duration = distance / self.sim_rail_speed
        self._animate_scalar(
            start_mm,
            float(target_mm),
            duration,
            lambda value: self._set_sim_state(rail_mm=value),
        )

    def _animate_sim_arm(self, goal_deg):
        with self._state_lock:
            start_deg = list(self.sim_q_deg)
        goal = [float(value) for value in goal_deg]
        if not self.use_simulation:
            self._set_sim_state(q_deg=goal)
            return
        duration = max(
            (abs(end - start) for start, end in zip(start_deg, goal)),
            default=0.0,
        ) / self.sim_joint_speed
        started = time.monotonic()
        while True:
            self._check_stop()
            elapsed = time.monotonic() - started
            u = 1.0 if duration <= 0.0 else min(1.0, elapsed / duration)
            smooth = u * u * (3.0 - 2.0 * u)
            present = [
                start + (end - start) * smooth
                for start, end in zip(start_deg, goal)
            ]
            self._set_sim_state(q_deg=present)
            if u >= 1.0:
                return
            time.sleep(1.0 / self.sim_hz)

    def _animate_scalar(self, start, end, duration, setter):
        started = time.monotonic()
        while True:
            self._check_stop()
            elapsed = time.monotonic() - started
            u = 1.0 if duration <= 0.0 else min(1.0, elapsed / duration)
            smooth = u * u * (3.0 - 2.0 * u)
            setter(float(start) + (float(end) - float(start)) * smooth)
            if u >= 1.0:
                return
            time.sleep(1.0 / self.sim_hz)

    def _check_stop(self):
        if self._stop_event.is_set():
            raise SequenceStopped("sequence stopped")

    def _fail(self, reason):
        self.get_logger().error("sequence failed: %s" % reason)
        self._emergency_stop(reason)
        self._set_phase("ERROR_HOLDING")

    def _emergency_stop(self, reason):
        self._stop_event.set()
        if self.rail is not None:
            try:
                self.rail.estop()
            except Exception as exc:
                self.get_logger().error("rail stop failed: %s" % exc)
        if self.bridge is not None:
            try:
                self.bridge.emergency_stop()
            except Exception as exc:
                self.get_logger().error("A-arm stop failed: %s" % exc)
        if self.gripper is not None:
            try:
                self.gripper.emergency_stop(reason)
            except Exception as exc:
                self.get_logger().error("centric stop failed: %s" % exc)

    # ---------------------------------------------------------- status / shell
    def _set_phase(self, phase):
        self.phase = str(phase)
        self.get_logger().info("--- PHASE: %s ---" % self.phase)
        self._publish_state()
        self._send_viewer_state(force=True)

    def _publish_state(self):
        msg = String()
        msg.data = "%s|busy=%s" % (self.phase, self.busy)
        self.state_pub.publish(msg)

    def _log_status(self):
        if self.rail is None:
            with self._state_lock:
                rail_text = "%.1f mm (SIM)" % self.sim_rail_mm
        else:
            rail_text = "%.1f mm, moving=%s" % (
                self.rail.position_mm,
                self.rail.moving,
            )
        if self.bridge is None:
            with self._state_lock:
                arm_text = "%s deg (SIM)" % [
                    round(value, 1) for value in self.sim_q_deg
                ]
        else:
            if self.require_reference_home and not self.reference_ready:
                arm_text = "UNCALIBRATED (run refhome)"
            else:
                arm_text = "[" + ", ".join(
                    "%.1f" % math.degrees(v) for v in self.bridge.read_joints()
                ) + "] deg"
        if self.gripper is None:
            gripper_text = "SIM"
        else:
            status = self.gripper.read_status()
            gripper_text = (
                "pos=%d, close=%.1f%%, load=%.1f%%, temp=%dC"
                % (
                    status["position"],
                    status["close_percent"],
                    status["load_percent"],
                    status["temperature_c"],
                )
            )
        self.get_logger().info(
            "phase=%s, busy=%s | rail=%s | A=%s | centric=%s"
            % (self.phase, self.busy, rail_text, arm_text, gripper_text)
        )

    @staticmethod
    def _fmt_status_value(value, fmt):
        if value is None:
            return "N/A"
        try:
            return fmt % value
        except Exception:
            return str(value)

    def _log_motor_status(self):
        """Read ID 1~8 once; intended for READY/diagnostic use."""
        if self.bridge is None and self.gripper is None:
            self.get_logger().info("MOTOR STATUS: simulation only")
            return

        self.get_logger().info("========== MOTOR STATUS ID 1~8 ==========")
        if self.bridge is not None:
            if hasattr(self.bridge, "read_detailed_status"):
                arm_status = self.bridge.read_detailed_status()
                for dxl_id in sorted(arm_status):
                    item = arm_status[dxl_id]
                    self.get_logger().info(
                        "ID%d pos=%s load=%s%% pwm=%s%% vel=%s volt=%sV "
                        "temp=%sC hwerr=%s%s"
                        % (
                            dxl_id,
                            self._fmt_status_value(item.get("position"), "%d"),
                            self._fmt_status_value(
                                item.get("load_percent"), "%.1f"
                            ),
                            self._fmt_status_value(
                                item.get("pwm_percent"), "%.1f"
                            ),
                            self._fmt_status_value(
                                item.get("velocity_raw"), "%d"
                            ),
                            self._fmt_status_value(item.get("voltage_v"), "%.1f"),
                            self._fmt_status_value(
                                item.get("temperature_c"), "%d"
                            ),
                            self._fmt_status_value(
                                item.get("hardware_error"), "0x%02X"
                            ),
                            (
                                " errors=" + "; ".join(item.get("errors", []))
                                if item.get("errors")
                                else ""
                            ),
                        )
                    )
            else:
                self.get_logger().warning(
                    "ID1~7 detailed status unavailable; update "
                    "dxl_bridge_centric_5axis.py"
                )

        if self.gripper is not None:
            try:
                item = self.gripper.read_status()
                self.get_logger().info(
                    "ID%d CENTRIC pos=%d close=%.1f%% load=%.1f%% "
                    "pwm=%.1f%% vel=%d volt=%.1fV temp=%dC hwerr=0x%02X"
                    % (
                        self.gripper.dxl_id,
                        item["position"],
                        item["close_percent"],
                        item["load_percent"],
                        item["pwm_percent"],
                        item["velocity_raw"],
                        item["voltage_v"],
                        item["temperature_c"],
                        item["hardware_error"],
                    )
                )
            except Exception as exc:
                self.get_logger().error("ID8 status failed: %s" % exc)
        self.get_logger().info("==========================================")

    def _start_terminal_thread(self):
        if sys.stdin is not None and sys.stdin.isatty():
            threading.Thread(target=self._terminal_loop, daemon=True).start()

    def _terminal_loop(self):
        while rclpy.ok():
            try:
                self._handle_command(input("CENTRIC_SEQ> "))
            except (EOFError, KeyboardInterrupt):
                return
            except Exception as exc:
                self.get_logger().error("terminal command failed: %s" % exc)

    @staticmethod
    def _print_help():
        print("\nReference startup (choose ONE):")
        print("  A) MECH LIMIT -> STRAIGHT HOME button / mechhome")
        print(
            "     [0, 56.7312, -134.3335, 108.6579, 0] deg 기계적 기준에 둔 뒤"
        )
        print("     버튼 1회 -> zero 계산 -> 자동으로 [0,0,0,0,0] 1자 HOME 이동")
        print("     J1/J5는 360도이므로 반드시 0도 마크에 맞춤")
        print("  B) CURRENT POSE = STRAIGHT HOME button / herehome")
        print("     현재 자세를 움직이지 않고 즉시 [0,0,0,0,0] 1자 HOME으로 등록")
        print("  refcapture/refhome = 기존 2단계 방식도 계속 사용 가능")
        print("  refstatus = 기준 상태 확인")
        print(
            "\nCommands: mechhome | herehome | refhome | refcapture | refstatus | "
            "start | stop(s) | status | open | grasp | home | railhome"
        )
        print(
            "START PROCESS: rail 200 mm -> A grasp pose -> grasp "
            "-> A HOME -> rail 0 mm\n"
        )

    def destroy_node(self):
        self._stop_event.set()
        self._close_hardware()
        self._stop_simulation_process()
        super().destroy_node()


A3_W_MM = 420.0
A3_H_MM = 297.0
A3_LT_ROBOT_X_MM = 482.0
A3_LT_ROBOT_Y_MM = 214.55
USER_X_SIGN = -1.0
USER_Y_SIGN = +1.0
USER_Z_SIGN = +1.0

DEFAULT_RAIL_TRAVEL_POSE_DEG = [0.0, 50.0, -50.0, -95.0, 0.0]
SAMPLE_HOLES = [111.843, 119.738, 123.031, 69.284, -77.50, 51.680]


def _user_mm_to_arm_m(x_mm, y_mm, z_mm):
    return np.array(
        [
            USER_X_SIGN * float(x_mm) / 1000.0,
            USER_Y_SIGN * float(y_mm) / 1000.0,
            USER_Z_SIGN * float(z_mm) / 1000.0,
        ],
        dtype=float,
    )


class _BaseplateCentricSequenceNode(CentricSequenceNode):
    NODE_NAME = "centric_sequence_5axis"
    STATE_TOPIC = "/centric_sequence/state"
    COMMAND_TOPIC = "/centric_sequence/command"

    def __init__(self):
        super().__init__()

        d = self.declare_parameter
        d("holes_topic", "/detected_holes")
        d("auto_start_on_vision", False)
        d("rail_travel_pose_deg", DEFAULT_RAIL_TRAVEL_POSE_DEG)
        d("shaft1_rail_mm", 620.0)
        d("shaft2_rail_mm", 683.0)
        d("bearing1_rail_mm", 805.0)
        d("bearing2_rail_mm", 864.0)
        d("gear1_rail_mm", 663.0)
        d("gear2_rail_mm", 718.0)
        d("gear_grasp_pose_deg", [0.0, -50.0, -65.0, -60.0, 0.0])
        d("bearing_preclose_angle_deg", 100.0)
        d("gear_preclose_angle_deg", 100.0)
        d("centric_full_close_angle_deg", 170.0)
        d("centric_preclose_angle_deg", 100.0)
        d("post_grasp_lift_mm", 30.0)
        d("post_grasp_level_tolerance_deg", 15.0)
        d("release_ratio", 0.50)
        d("full_grasp_mode", "auto")
        d("mid_finish_mode", "release_50_and_retract")
        d("target_z_mm", 30.0)
        d("shaft_place_extra_depth_mm", 5.0)
        d("bearing_place_extra_depth_mm", 10.0)
        d("approach_mm", 80.0)
        d("vision_rail_ref_m", 0.0)
        d("planner_hz", 60.0)
        d("robot_cmd_hz", 40.0)
        d("joint_path_speed_deg_s", 25.0)
        d("vertical_speed_mps", 0.02)
        d("limit_joint_speed", True)
        d("speed_margin", 0.90)
        d("check_collision", True)
        d("arm_urdf_name", "robot_arm_5axis_suction.urdf")

        g = self.get_parameter
        self.auto_start_on_vision = bool(g("auto_start_on_vision").value)
        self.rail_travel_pose_deg = self._read_pose("rail_travel_pose_deg")
        self.shaft1_rail_mm = float(g("shaft1_rail_mm").value)
        self.shaft2_rail_mm = float(g("shaft2_rail_mm").value)
        self.bearing1_rail_mm = float(g("bearing1_rail_mm").value)
        self.bearing2_rail_mm = float(g("bearing2_rail_mm").value)
        self.gear1_rail_mm = float(g("gear1_rail_mm").value)
        self.gear2_rail_mm = float(g("gear2_rail_mm").value)
        self.gear_grasp_pose_deg = self._read_pose("gear_grasp_pose_deg")
        self.bearing_preclose_angle_deg = float(g("bearing_preclose_angle_deg").value)
        self.gear_preclose_angle_deg = float(g("gear_preclose_angle_deg").value)
        self.centric_full_close_angle_deg = float(
            g("centric_full_close_angle_deg").value
        )
        self.centric_preclose_angle_deg = float(
            g("centric_preclose_angle_deg").value
        )
        if not math.isfinite(self.centric_full_close_angle_deg) or self.centric_full_close_angle_deg <= 0.0:
            raise ValueError("centric_full_close_angle_deg must be positive")
        if not 0.0 <= self.centric_preclose_angle_deg <= self.centric_full_close_angle_deg:
            raise ValueError(
                "centric_preclose_angle_deg must be between 0 and full close angle"
            )
        for name in ("bearing_preclose_angle_deg", "gear_preclose_angle_deg"):
            angle = getattr(self, name)
            if not 0.0 <= angle <= self.centric_full_close_angle_deg:
                raise ValueError("%s must be between 0 and full close angle" % name)
        self.preclose_ratio = (
            self.centric_preclose_angle_deg / self.centric_full_close_angle_deg
        )
        self.post_grasp_lift_mm = float(g("post_grasp_lift_mm").value)
        if self.post_grasp_lift_mm < 0.0:
            raise ValueError("post_grasp_lift_mm must be non-negative")
        self.post_grasp_level_tolerance = np.radians(
            max(1.0, float(g("post_grasp_level_tolerance_deg").value))
        )
        self.release_ratio = self._ratio_parameter("release_ratio")
        self.full_grasp_mode = str(g("full_grasp_mode").value).strip().lower()
        self.mid_finish_mode = str(g("mid_finish_mode").value).strip().lower()
        self.target_z_mm = float(g("target_z_mm").value)
        if not math.isfinite(self.target_z_mm):
            raise ValueError("target_z_mm must be finite")
        for name in ("shaft_place_extra_depth_mm", "bearing_place_extra_depth_mm"):
            value = float(g(name).value)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError("%s must be finite and non-negative" % name)
            setattr(self, name, value)
        self.shaft_place_z_mm = self.target_z_mm - self.shaft_place_extra_depth_mm
        self.bearing_place_z_mm = self.target_z_mm - self.bearing_place_extra_depth_mm
        self.gear_place_z_mm = self.shaft_place_z_mm
        self.approach_mm = float(g("approach_mm").value)
        self.vision_rail_ref = float(g("vision_rail_ref_m").value)
        self.planner_hz = max(20.0, float(g("planner_hz").value))
        self.robot_cmd_hz = max(1.0, float(g("robot_cmd_hz").value))
        self.joint_path_speed = max(
            1.0, float(g("joint_path_speed_deg_s").value)
        )
        self.vertical_speed = max(0.003, float(g("vertical_speed_mps").value))
        self.limit_joint_speed = bool(g("limit_joint_speed").value)
        self.speed_margin = float(
            np.clip(float(g("speed_margin").value), 0.1, 1.0)
        )
        self.arm_urdf_name = str(g("arm_urdf_name").value)

        if self.full_grasp_mode not in ("auto", "fixed_100"):
            raise ValueError("full_grasp_mode must be auto or fixed_100")
        if self.mid_finish_mode not in (
            "release_50_and_retract",
            "hold_at_mid",
            "open_and_retract",
        ):
            raise ValueError(
                "mid_finish_mode must be release_50_and_retract, "
                "hold_at_mid, or open_and_retract"
            )
        if self.approach_mm <= 0.0:
            raise ValueError("approach_mm must be positive")
        for rail_mm in (
            self.shaft1_rail_mm,
            self.shaft2_rail_mm,
            self.bearing1_rail_mm,
            self.bearing2_rail_mm,
            self.gear1_rail_mm,
            self.gear2_rail_mm,
        ):
            if not RAIL_MIN_M * 1000.0 <= rail_mm <= RAIL_MAX_M * 1000.0:
                raise ValueError(
                    "pick rail target %.1f mm is outside %.1f~%.1f mm"
                    % (rail_mm, RAIL_MIN_M * 1000.0, RAIL_MAX_M * 1000.0)
                )

        self.rail_origin = self._read_rail_origin()
        self.latest_holes = None
        self.latest_holes_stamp = None
        self.sim_gripper_ratio = 0.0
        self._marker_ids = {}
        self._holes_text_id = None
        self._vertical_test_ascent = None
        self._vertical_test_tag = None

        # Collision checking is also isolated from the real motor process.
        self.checker = None  # kept only for backward-compatible status checks
        self.collision_enabled = False
        self._collision_process = None
        self._collision_request_queue = None
        self._collision_response_queue = None
        self._collision_stop_event = None
        self._collision_rpc_lock = threading.Lock()
        self._collision_request_id = 0

        self.vmax = None

        if self.limit_joint_speed:
            gears = [cfg["gear"] for cfg in JOINT_MOTORS[:N_JOINTS]]
            self.vmax = joint_speed_limits(PROFILE_VELOCITY, gears)

        if bool(g("check_collision").value):
            self._start_collision_process()

        self.create_subscription(
            Float64MultiArray,
            str(g("holes_topic").value),
            self._on_holes,
            10,
        )

        if self.use_simulation:
            # All extra buttons and target markers are created in the isolated
            # PyBullet process.  The control process only sends state via IPC.
            self._send_viewer_state()

        self._set_phase("WAITING_FOR_HOLE_COORDINATES")
        self.get_logger().info(
            "baseplate sequence ready; hardware defaults OFF, START button required"
        )
        if self.use_robot and self.require_reference_home:
            self.get_logger().warning(
                "REAL A-ARM STARTUP: reference not ready. "
                "Use MECH LIMIT -> STRAIGHT HOME from %s deg, "
                "or CURRENT POSE = STRAIGHT HOME if the arm is already straight."
                % self.reference_pose_deg
            )
        self.get_logger().info(
            "A-arm targets | HOME=%s | RAIL_TRAVEL=%s | SHAFT_PICK=%s"
            % (
                self.home_pose_deg,
                self.rail_travel_pose_deg,
                self.grasp_pose_deg,
            )
        )
        self.get_logger().info(
            "centric pre-close: %.1f/%.1f deg = %.1f%% of full travel; "
            "post-grasp lift=%.1f mm"
            % (
                self.centric_preclose_angle_deg,
                self.centric_full_close_angle_deg,
                100.0 * self.preclose_ratio,
                self.post_grasp_lift_mm,
            )
        )
        self.get_logger().info(
            "pick/place order | shaft1 %.1f mm -> MID | shaft2 %.1f mm -> SIDE | "
            "bearing1 %.1f mm -> MID | bearing2 %.1f mm -> SIDE | "
            "gear1 %.1f mm -> MID | gear2 %.1f mm -> SIDE"
            % (
                self.shaft1_rail_mm,
                self.shaft2_rail_mm,
                self.bearing1_rail_mm,
                self.bearing2_rail_mm,
                self.gear1_rail_mm,
                self.gear2_rail_mm,
            )
        )

        self.get_logger().info(
            "pick profiles | shaft/bearing pose=%s | gear pose=%s | "
            "preclose shaft=%.1f bearing=%.1f gear=%.1f deg"
            % (self.grasp_pose_deg, self.gear_grasp_pose_deg,
               self.centric_preclose_angle_deg, self.bearing_preclose_angle_deg,
               self.gear_preclose_angle_deg)
        )

        self.get_logger().info(
            "placement Z | shaft=%.1f mm | bearing=%.1f mm | gear=%.1f mm"
            % (self.shaft_place_z_mm, self.bearing_place_z_mm, self.gear_place_z_mm)
        )

    # ----------------------------------------------------------- parameters
    def _read_pose(self, name):
        values = [float(v) for v in self.get_parameter(name).value]
        if len(values) != 5 or not np.isfinite(values).all():
            raise ValueError("%s must contain five finite angles" % name)
        return values

    def _ratio_parameter(self, name):
        value = float(self.get_parameter(name).value)
        if not 0.0 <= value <= 1.0:
            raise ValueError("%s must be between 0.0 and 1.0" % name)
        return value

    # --------------------------------------------------------------- vision
    def _on_holes(self, msg):
        self._accept_holes(list(msg.data), source="/detected_holes")

    def _accept_holes(self, values, source):
        if len(values) < 4:
            self.get_logger().error(
                "hole message needs at least [mid_x, mid_y, side_x, side_y]"
            )
            return False
        data = np.asarray(values[:6], dtype=float)
        if data.size < 6:
            data = np.pad(data, (0, 6 - data.size), constant_values=np.nan)
        if not np.isfinite(data[:4]).all():
            self.get_logger().error("mid/side coordinates must be finite")
            return False
        mid = data[:2]
        side = data[2:4]
        if not self._in_a3(mid) or not self._in_a3(side):
            self.get_logger().error(
                "coordinates outside A3: mid=%s side=%s"
                % (mid.round(3).tolist(), side.round(3).tolist())
            )
            return False
        if self.busy:
            self.get_logger().warning(
                "sequence is busy; new hole coordinates were ignored"
            )
            return False

        self.latest_holes = data.copy()
        self.latest_holes_stamp = time.monotonic()
        self._update_target_markers()
        self._set_phase("COORDINATES_READY_PRESS_START")
        self.get_logger().info(
            "holes cached from %s | mid=(%.3f, %.3f) side=(%.3f, %.3f)"
            % (source, mid[0], mid[1], side[0], side[1])
        )
        if self.auto_start_on_vision:
            self._start_sequence()
        return True

    @staticmethod
    def _in_a3(point):
        return 0.0 <= point[0] <= A3_W_MM and 0.0 <= point[1] <= A3_H_MM

    def _start_sequence(self):
        if self.latest_holes is None:
            self.get_logger().warning(
                "no hole coordinates; run baseplate_hole_wonseok or LOAD SAMPLE first"
            )
            return
        if self._vertical_test_ascent is not None:
            self.get_logger().warning(
                "vertical test is holding at Z=%.1f mm; RETREAT before START"
                % self.target_z_mm
            )
            return
        super()._start_sequence()

    # ------------------------------------------------------------- sequence
    def _run_sequence(self):
        holes = np.asarray(self.latest_holes, float).copy()
        mid = holes[:2]
        side = holes[2:4]
        try:
            cycles = (
                ("SHAFT_1", self.shaft1_rail_mm, mid, "MID", self.grasp_pose_deg, self.centric_preclose_angle_deg, self.shaft_place_z_mm),
                ("SHAFT_2", self.shaft2_rail_mm, side, "SIDE", self.grasp_pose_deg, self.centric_preclose_angle_deg, self.shaft_place_z_mm),
                ("BEARING_1", self.bearing1_rail_mm, mid, "MID", self.grasp_pose_deg, self.bearing_preclose_angle_deg, self.bearing_place_z_mm),
                ("BEARING_2", self.bearing2_rail_mm, side, "SIDE", self.grasp_pose_deg, self.bearing_preclose_angle_deg, self.bearing_place_z_mm),
                ("GEAR_1", self.gear1_rail_mm, mid, "MID", self.gear_grasp_pose_deg, self.gear_preclose_angle_deg, self.gear_place_z_mm),
                ("GEAR_2", self.gear2_rail_mm, side, "SIDE", self.gear_grasp_pose_deg, self.gear_preclose_angle_deg, self.gear_place_z_mm),
            )
            for item, rail_mm, point, target_tag, pose, preclose_deg, place_z_mm in cycles:
                self._pick_and_place_cycle(
                    item,
                    rail_mm,
                    point,
                    target_tag,
                    pose,
                    preclose_deg,
                    place_z_mm,
                )

            self._set_phase("A_RAIL_TRAVEL_FINAL_RETURN")
            self._move_arm(self.rail_travel_pose_deg)
            self._set_phase("RAIL_CONFIRM_INITIAL_AFTER_PROCESS")
            self._move_rail(0.0)
            self._set_phase("A_RETURN_INITIAL_AFTER_PROCESS")
            self._move_arm(self.home_pose_deg)
            self._set_phase("READY_NEXT_COMMAND_HOME")
            self.get_logger().info(
                "complete: shaft1->MID, shaft2->SIDE, bearing1->MID, "
                "bearing2->SIDE, gear1->MID, gear2->SIDE; A HOME=%s, rail 0 mm; waiting for next command"
                % self.home_pose_deg
            )
        except SequenceStopped:
            self.get_logger().warning(
                "process stopped at current position; node remains ready"
            )
            self._set_phase("STOPPED_READY")
        except Exception as exc:
            self._fail(str(exc))
        finally:
            with self._state_lock:
                self.busy = False

    def _pick_and_place_cycle(self, item, rail_mm, point, target_tag, pose, preclose_deg, place_z_mm):
        """Pick one item from the rail and place it at MID or SIDE."""
        self._set_phase("A_RAIL_TRAVEL_BEFORE_%s" % item)
        self._move_arm(self.rail_travel_pose_deg)

        self._set_phase("RAIL_TO_%s_%.0fMM" % (item, rail_mm))
        self._move_rail(rail_mm)

        self._set_phase(
            "%s_PICK_AND_PRECLOSE_%.0fDEG"
            % (item, preclose_deg)
        )
        self._move_arm_with_preclose(
            pose,
            preclose_deg / self.centric_full_close_angle_deg,
        )

        self._set_phase("%s_FINAL_GRASP" % item)
        self._finish_grasp()

        self._set_phase("%s_VERTICAL_LIFT_30MM" % item)
        self._lift_after_grasp()

        self._set_phase("A_RAIL_TRAVEL_WITH_%s" % item)
        self._move_arm(self.rail_travel_pose_deg)

        self._set_phase("RAIL_HOME_WITH_%s" % item)
        self._move_rail(0.0)

        self._place_at_coordinate(
            point,
            "%s_%s" % (item, target_tag),
            always_release=True,
            place_z_mm=place_z_mm,
        )

    def _move_arm_with_preclose(self, goal_deg, ratio):
        self._check_stop()
        ratio = float(ratio)
        if self.gripper is not None:
            self.gripper.command_ratio(ratio)
            self._set_gripper_ratio_state(ratio, moving=True)
            self._move_arm(goal_deg)
            self._set_gripper_ratio_state(ratio, moving=False)
            return
        if self.use_gripper:
            raise RuntimeError("centric gripper connection is unavailable")

        errors = []

        def animate_gripper():
            try:
                self._animate_gripper_ratio(ratio, 0.9)
            except Exception as exc:
                errors.append(exc)

        worker = threading.Thread(target=animate_gripper, daemon=True)
        worker.start()
        self._move_arm(goal_deg)
        worker.join()
        if errors:
            raise errors[0]

    def _pick_pose_with_preclose_only(self):
        """Move A to the shaft-pick pose while pre-closing the gripper."""
        self.get_logger().warning(
            "MANUAL PICK POSE: A -> %s deg while centric closes to %.1f deg "
            "(%.1f%%); check clearance before motion"
            % (
                self.grasp_pose_deg,
                self.centric_preclose_angle_deg,
                100.0 * self.preclose_ratio,
            )
        )
        self._move_arm_with_preclose(
            self.grasp_pose_deg,
            self.preclose_ratio,
        )

    def _home_only(self):
        if self._vertical_test_ascent is not None:
            self.get_logger().warning(
                "HOME requested during vertical test hold; retreating vertically first"
            )
            self._vertical_test_retreat()
        self.get_logger().info(
            "GO A INITIAL/HOME command -> %s deg" % self.home_pose_deg
        )
        self._move_arm(self.home_pose_deg)

    def _rail_home_only(self):
        if self._vertical_test_ascent is not None:
            self.get_logger().warning(
                "RAIL HOME requested during vertical test hold; "
                "retreating vertically first"
            )
            self._vertical_test_retreat()
        self.get_logger().info(
            "RAIL HOME: A rail-travel pose %s -> rail 0 mm"
            % self.rail_travel_pose_deg
        )
        self._move_arm(self.rail_travel_pose_deg)
        self._move_rail(0.0)

    def _all_initial(self):
        """Recover both mechanisms without terminating the ROS node.

        A first moves to the rail-travel pose, the rail returns to 0 mm, and
        A then moves to the captured [0,0,0,0,0] startup pose.
        """
        if self._vertical_test_ascent is not None:
            self.get_logger().warning(
                "ALL INITIAL requested during vertical test hold; "
                "retreating vertically first"
            )
            self._vertical_test_retreat()
        self.get_logger().warning(
            "ALL INITIAL: A RAIL_TRAVEL %s -> rail 0 mm -> A HOME %s"
            % (self.rail_travel_pose_deg, self.home_pose_deg)
        )
        self._move_arm(self.rail_travel_pose_deg)
        self._move_rail(0.0)
        self._move_arm(self.home_pose_deg)

    def _finish_grasp(self):
        self._check_stop()
        if self.full_grasp_mode == "fixed_100":
            if not self._move_gripper_ratio(1.0):
                raise RuntimeError("fixed 100% centric close failed")
            self._set_sim_state(gripper="HOLDING (100% target)")
            return

        self._set_sim_state(gripper="AUTO GRASPING")
        if self.gripper is None:
            if self.use_gripper:
                raise RuntimeError("centric gripper connection is unavailable")
            self._animate_gripper_ratio(1.0, 0.8)
            self._set_sim_state(gripper="HOLDING (SIM)")
            return

        # Pre-close has already covered most of the travel.  A zero minimum
        # lets contact detection work immediately from the present position.
        result = self.gripper.auto_grasp(
            self._stop_event.is_set,
            min_grasp_travel_pulse=0,
        )
        if not result.success:
            raise RuntimeError("centric grasp failed: " + result.reason)
        self._set_sim_state(gripper="HOLDING")

    def _lift_after_grasp(self):
        """Raise the grasped shaft approximately vertically by the set amount."""
        self._check_stop()
        if self.post_grasp_lift_mm <= 0.0:
            return

        q_now5 = self._current_q_rad()
        q_now = q_now5[:N_JOINTS]
        p_start = fk_pos(q_now)
        p_end = p_start + np.array(
            [0.0, 0.0, self.post_grasp_lift_mm / 1000.0], dtype=float
        )
        trajectory, _duration, info = plan_level_descent_retimed(
            q_now,
            p_start,
            p_end,
            hz=self.planner_hz,
            lin_speed=self.vertical_speed,
            vmax=self.vmax if self.limit_joint_speed else None,
            margin=self.speed_margin,
            relaxed=True,
            level_tol=self.post_grasp_level_tolerance,
        )
        if trajectory is None:
            raise RuntimeError("post-grasp vertical lift failed: %s" % info)
        error = self._collision_error(trajectory, "post-grasp 30 mm lift")
        if error:
            raise RuntimeError(error)

        trajectory5 = np.hstack(
            [
                trajectory[:, :N_JOINTS],
                np.full((len(trajectory), 1), q_now5[4]),
            ]
        )
        self.get_logger().info(
            "post-grasp lift: %.1f mm, start=%s, end=%s"
            % (
                self.post_grasp_lift_mm,
                p_start.round(4).tolist(),
                p_end.round(4).tolist(),
            )
        )
        self._execute_joint_trajectory(trajectory5)

    def _place_at_coordinate(self, point, tag, always_release, place_z_mm=None):
        if place_z_mm is None:
            place_z_mm = self.target_z_mm
        self.get_logger().info(
            "%s placement Z=%.1f mm" % (tag, place_z_mm)
        )
        approach, descent, ascent, world_low = self._build_vertical_paths(
            float(point[0]), float(point[1]), target_z_mm=place_z_mm
        )
        target_kind = str(tag).rsplit("_", 1)[-1].upper()
        self._move_marker(target_kind.lower(), world_low)

        self._set_phase("%s_VERTICAL_TOP_APPROACH" % tag)
        self._execute_joint_trajectory(approach)

        self._set_phase("%s_VERTICAL_DESCENT_Z%gMM" % (tag, place_z_mm))
        self._execute_joint_trajectory(descent)

        if target_kind == "MID" and not always_release:
            if self.mid_finish_mode == "hold_at_mid":
                return False
            release_ratio = (
                0.0 if self.mid_finish_mode == "open_and_retract"
                else self.release_ratio
            )
        else:
            release_ratio = self.release_ratio

        self._set_phase("%s_RELEASE" % tag)
        if not self._move_gripper_ratio(release_ratio):
            raise RuntimeError("%s centric release failed" % tag)

        self._set_phase("%s_VERTICAL_RETREAT" % tag)
        self._execute_joint_trajectory(ascent)

        return True

    def _start_vertical_test(self, target_kind):
        """Start a MID/SIDE accuracy check without operating the gripper."""
        if self.latest_holes is None:
            self.get_logger().warning(
                "no hole coordinates; run baseplate_hole_wonseok or LOAD SAMPLE first"
            )
            return
        if self._vertical_test_ascent is not None:
            self.get_logger().warning(
                "vertical test is holding at Z=%.1f mm; RETREAT before another test"
                % self.target_z_mm
            )
            return

        kind = str(target_kind).upper()
        if kind == "MID":
            point = np.asarray(self.latest_holes[:2], dtype=float).copy()
        elif kind == "SIDE":
            point = np.asarray(self.latest_holes[2:4], dtype=float).copy()
        else:
            self.get_logger().error("vertical test target must be MID or SIDE")
            return

        self._start_auxiliary(
            "TEST_%s_VERTICAL_APPROACH" % kind,
            lambda point=point, kind=kind: self._run_vertical_test(point, kind),
        )

    def _run_vertical_test(self, point, target_kind):
        """Reproduce the insertion approach and hold at the detected hole."""
        self.get_logger().warning(
            "VERTICAL TEST %s: gripper will not move; A rail-travel pose -> "
            "rail 0 mm -> vertical descent to Z=%.1f mm"
            % (target_kind, self.target_z_mm)
        )
        self._move_arm(self.rail_travel_pose_deg)
        self._move_rail(0.0)

        approach, descent, ascent, world_low = self._build_vertical_paths(
            float(point[0]), float(point[1])
        )
        self._move_marker(target_kind.lower(), world_low)

        self._set_phase("TEST_%s_VERTICAL_TOP_APPROACH" % target_kind)
        self._execute_joint_trajectory(approach)
        self._set_phase("TEST_%s_VERTICAL_DESCENT_Z30" % target_kind)
        self._execute_joint_trajectory(descent)

        self._vertical_test_ascent = np.asarray(ascent, dtype=float).copy()
        self._vertical_test_tag = target_kind
        self._set_phase("TEST_%s_HOLD_Z30" % target_kind)
        self.get_logger().warning(
            "VERTICAL TEST %s reached detected coordinate %s at Z=%.1f mm; "
            "inspect accuracy, then press VERTICAL TEST RETREAT"
            % (target_kind, np.asarray(point).round(3).tolist(), self.target_z_mm)
        )

    def _vertical_test_retreat(self):
        """Return from the last test along the saved vertical ascent path."""
        if self._vertical_test_ascent is None:
            self.get_logger().warning("no completed vertical test to retreat from")
            return
        target_kind = self._vertical_test_tag or "TARGET"
        ascent = np.asarray(self._vertical_test_ascent, dtype=float).copy()
        self._set_phase("TEST_%s_VERTICAL_RETREAT" % target_kind)
        self._execute_joint_trajectory(ascent)
        self._vertical_test_ascent = None
        self._vertical_test_tag = None
        self.get_logger().info(
            "vertical test retreat complete; holding at the vertical approach top"
        )

    # -------------------------------------------------------------- planner
    def _current_q_rad(self):
        if self.bridge is not None:
            self._assert_reference_ready()
            actual = np.asarray(self.bridge.read_joints(), dtype=float).ravel()
            if actual.size < 5:
                raise RuntimeError("A-arm returned fewer than five joints")
            return actual[:5].copy()
        with self._state_lock:
            return np.radians(np.asarray(self.sim_q_deg, dtype=float))

    def _move_arm(self, goal_deg):
        self._check_stop()
        self._assert_reference_ready()
        q_from = self._current_q_rad()
        q_goal = np.radians(np.asarray(goal_deg, dtype=float))
        trajectory, _duration, _count = plan_joint_path(
            q_from,
            q_goal,
            hz=self.planner_hz,
            speed_deg_per_s=self.joint_path_speed,
        )
        error = self._collision_error(trajectory[:, :N_JOINTS], "joint move")
        if error:
            raise RuntimeError(error)
        self._execute_joint_trajectory(trajectory)

    def _build_vertical_paths(self, a3_x_mm, a3_y_mm, target_z_mm=None):
        q_now5 = self._current_q_rad()
        q_now = q_now5[:N_JOINTS]
        low_z_mm = self.target_z_mm if target_z_mm is None else float(target_z_mm)
        if not math.isfinite(low_z_mm):
            raise ValueError("vertical target Z must be finite")
        world_low = self.a3_to_world(a3_x_mm, a3_y_mm, low_z_mm)
        # Keep the original approach plane; only extend the downward segment.
        world_high = self.a3_to_world(
            a3_x_mm, a3_y_mm, self.target_z_mm + self.approach_mm
        )
        arm_low = self.world_to_arm(world_low)
        arm_high = self.world_to_arm(world_high)

        candidates = ik_fix_candidates(arm_high, q0=q_now)
        candidates = [
            q
            for q, _error in candidates
            if (not self.collision_enabled) or self._collision_check_config(q)[0]
        ]
        if not candidates:
            raise RuntimeError(
                "no exact vertical IK at approach point: %s"
                % arm_high.round(4).tolist()
            )
        candidates.sort(
            key=lambda q: float(np.linalg.norm(wrap_to_pi(q - q_now)))
        )

        last_error = ""
        for q_high in candidates:
            if level_error(q_now) <= 1e-3:
                approach, error = self._vertical_line(
                    q_now, fk_pos(q_now), arm_high, "vertical top approach"
                )
            else:
                approach, error = self._joint_approach(q_now, q_high)
            if approach is None:
                last_error = error
                continue

            q_high_actual = np.asarray(approach[-1], float)[:N_JOINTS]
            if level_error(q_high_actual) > 1e-4:
                last_error = "approach did not finish in a vertical pose"
                continue
            descent, error = self._vertical_line(
                q_high_actual, arm_high, arm_low, "vertical descent"
            )
            if descent is None:
                last_error = error
                continue
            ascent = np.asarray(descent[::-1], dtype=float).copy()
            error = self._collision_error(ascent, "vertical retreat")
            if error:
                last_error = error
                continue

            q5_column = np.full((len(approach), 1), q_now5[4])
            approach5 = np.hstack([approach[:, :N_JOINTS], q5_column])
            descent5 = np.hstack(
                [descent[:, :N_JOINTS], np.full((len(descent), 1), q_now5[4])]
            )
            ascent5 = np.hstack(
                [ascent[:, :N_JOINTS], np.full((len(ascent), 1), q_now5[4])]
            )
            return approach5, descent5, ascent5, world_low

        raise RuntimeError(last_error or "vertical path planning failed")

    def _start_collision_process(self):
        if self.collision_enabled:
            return

        self._collision_request_queue = self._mp_ctx.Queue(maxsize=4)
        self._collision_response_queue = self._mp_ctx.Queue(maxsize=4)
        self._collision_stop_event = self._mp_ctx.Event()

        config = {
            "arm_urdf_path": self._find_urdf(self.arm_urdf_name),
            "pkg_dir": self.pkg_dir,
            "n_joints": N_JOINTS,
            "mesh_dir": self.mesh_dir,
        }

        self._collision_process = self._mp_ctx.Process(
            target=_collision_worker,
            args=(
                config,
                self._collision_request_queue,
                self._collision_response_queue,
                self._collision_stop_event,
            ),
            name="centric_collision_worker",
            daemon=True,
        )
        self._collision_process.start()
        self.collision_enabled = True
        # Non-None sentinel for older internal checks/logging.
        self.checker = True

        self.get_logger().warning(
            "CollisionChecker isolated in separate process pid=%s"
            % self._collision_process.pid
        )

    def _stop_collision_process(self):
        self.collision_enabled = False
        self.checker = None

        if self._collision_stop_event is not None:
            try:
                self._collision_stop_event.set()
            except Exception:
                pass

        if self._collision_request_queue is not None:
            try:
                self._collision_request_queue.put_nowait(None)
            except Exception:
                pass

        if self._collision_process is not None:
            try:
                self._collision_process.join(timeout=2.0)
            except Exception:
                pass
            if self._collision_process.is_alive():
                try:
                    self._collision_process.terminate()
                    self._collision_process.join(timeout=1.0)
                except Exception:
                    pass

        self._collision_process = None

        for q in (
            self._collision_request_queue,
            self._collision_response_queue,
        ):
            if q is not None:
                try:
                    q.close()
                except Exception:
                    pass

        self._collision_request_queue = None
        self._collision_response_queue = None
        self._collision_stop_event = None

    def _collision_rpc(self, kind, payload, timeout=60.0):
        if not self.collision_enabled:
            raise RuntimeError("collision checker is not enabled")

        if (
            self._collision_process is None
            or not self._collision_process.is_alive()
        ):
            raise RuntimeError("collision checker process is not alive")

        with self._collision_rpc_lock:
            self._collision_request_id += 1
            request_id = self._collision_request_id

            self._collision_request_queue.put(
                (request_id, str(kind), np.asarray(payload, dtype=float)),
                timeout=2.0,
            )

            deadline = time.monotonic() + float(timeout)
            while time.monotonic() < deadline:
                self._check_stop()

                if not self._collision_process.is_alive():
                    raise RuntimeError(
                        "collision checker process stopped unexpectedly"
                    )

                remaining = max(0.01, min(0.25, deadline - time.monotonic()))
                try:
                    result_id, ok, result = self._collision_response_queue.get(
                        timeout=remaining
                    )
                except queue.Empty:
                    continue

                if result_id != request_id:
                    # Only one RPC is allowed at a time by _collision_rpc_lock,
                    # so this should never happen.
                    continue

                if not ok:
                    raise RuntimeError(
                        "collision worker error: %s" % result
                    )
                return result

        raise RuntimeError(
            "collision checker timeout after %.1f s" % float(timeout)
        )

    def _collision_check_config(self, q):
        if not self.collision_enabled:
            return True, ""
        result = self._collision_rpc("config", q)
        return result

    def _joint_approach(self, q_from, q_goal):
        trajectory, _duration, _count = plan_joint_path(
            q_from,
            q_goal,
            hz=self.planner_hz,
            speed_deg_per_s=self.joint_path_speed,
        )
        if self.vmax is not None and self.limit_joint_speed:
            trajectory, _duration, info = retime_path(
                trajectory,
                self.vmax,
                self.planner_hz,
                margin=self.speed_margin,
            )
            if trajectory is None:
                return None, "approach retiming failed: %s" % info
        error = self._collision_error(trajectory, "approach")
        return (None, error) if error else (trajectory, "")

    def _vertical_line(self, q_from, p_from, p_to, tag):
        trajectory, _duration, info = plan_level_descent_retimed(
            q_from,
            p_from,
            p_to,
            hz=self.planner_hz,
            lin_speed=self.vertical_speed,
            vmax=self.vmax if self.limit_joint_speed else None,
            margin=self.speed_margin,
            relaxed=False,
        )
        if trajectory is None:
            return None, "%s failed: %s" % (tag, info)
        error = self._collision_error(trajectory, tag)
        return (None, error) if error else (trajectory, "")

    def _collision_error(self, trajectory, tag):
        if not self.collision_enabled:
            return ""
        ok, index, reason = self._collision_rpc("path", trajectory)
        if ok:
            return ""
        return "%s collision at sample %d: %s" % (tag, index, reason)

    def _execute_joint_trajectory(self, trajectory):
        path = np.asarray(trajectory, dtype=float)
        if path.ndim != 2 or path.shape[0] == 0:
            raise RuntimeError("empty joint trajectory")
        if path.shape[1] == N_JOINTS:
            q5 = self._current_q_rad()[4]
            path = np.hstack([path, np.full((len(path), 1), q5)])

        started = time.monotonic()
        next_robot_command = 0.0
        for index, q in enumerate(path):
            target_time = started + index / self.planner_hz
            while time.monotonic() < target_time:
                self._check_stop()
                time.sleep(min(0.01, target_time - time.monotonic()))
            self._check_stop()
            now = time.monotonic()
            if self.bridge is not None and (
                now >= next_robot_command or index == len(path) - 1
            ):
                self.bridge.command_joints(q[:5].tolist())
                next_robot_command = now + 1.0 / self.robot_cmd_hz
            self._set_sim_state(q_deg=np.degrees(q[:5]).tolist())

        if self.bridge is not None:
            self._wait_arm_goal(path[-1, :5])

    def _wait_arm_goal(self, goal_rad):
        # The trajectory has already sent its final goal. Keep the existing
        # arrival criterion and timeout; only add sparse diagnostic logging.
        clock = getattr(self, '_process_time', time.monotonic)
        started = clock()
        deadline = started + self.joint_timeout
        next_report = started + 2.0
        motion_epoch = getattr(self, '_motion_epoch', None)
        goal_deg = np.degrees(np.asarray(goal_rad[:5], dtype=float))
        actual_deg = None
        errors_deg = None
        reads = 0
        slowest_read_ms = 0.0
        while clock() < deadline:
            self._check_stop()
            if motion_epoch != getattr(self, '_motion_epoch', None):
                resume_other = getattr(self, '_resume_concurrent_motion', None)
                if resume_other is not None:
                    resume_other()
                self.bridge.command_joints(np.asarray(goal_rad[:5]).tolist())
                motion_epoch = getattr(self, '_motion_epoch', None)
            read_started = time.monotonic()
            try:
                actual = np.asarray(self.bridge.read_joints(), dtype=float).ravel()
            except Exception as exc:
                self.get_logger().error(
                    "A-arm position read failed after %.1fs, %d reads; "
                    "goal=%s deg: %s"
                    % (clock() - started, reads,
                       np.round(goal_deg, 2).tolist(), exc)
                )
                raise
            read_ms = 1000.0 * (time.monotonic() - read_started)
            slowest_read_ms = max(slowest_read_ms, read_ms)
            reads += 1
            if actual.size < 5:
                raise RuntimeError("A-arm returned fewer than five joints")
            self._set_sim_state(q_deg=np.degrees(actual[:5]).tolist())
            actual_deg = np.degrees(actual[:5])
            errors_deg = np.degrees(np.abs(
                wrap_to_pi(actual[:5] - goal_rad[:5])
            ))
            worst = int(np.argmax(errors_deg))
            tolerances = np.full(5, math.degrees(self.joint_tol_rad))
            arrival_tolerances = getattr(self, '_arrival_tolerances_deg', None)
            if arrival_tolerances is not None:
                tolerances = np.asarray(arrival_tolerances(), dtype=float)
            if np.all(errors_deg <= tolerances):
                if errors_deg[worst] > math.degrees(self.joint_tol_rad):
                    self.get_logger().warning(
                        'Inspection pose accepted within scoped tolerance: '
                        'error=%s deg, limit=%s deg'
                        % (np.round(errors_deg, 2).tolist(),
                           np.round(tolerances, 2).tolist()))
                return
            now = clock()
            if now >= next_report:
                self.get_logger().warning(
                    "A-arm waiting %.1f/%.1fs: goal=%s, actual=%s, "
                    "error=%s deg; worst=J%d %.2f deg; "
                    "last_read=%.1f ms, slowest_read=%.1f ms, reads=%d"
                    % (now - started, self.joint_timeout,
                       np.round(goal_deg, 2).tolist(),
                       np.round(actual_deg, 2).tolist(),
                       np.round(errors_deg, 2).tolist(), worst + 1,
                       errors_deg[worst], read_ms, slowest_read_ms, reads)
                )
                next_report = now + 2.0
            time.sleep(0.08)
        self.get_logger().error(
            "A-arm TIMEOUT after %.1fs: goal=%s, last_actual=%s, "
            "last_error=%s deg; slowest_read=%.1f ms, reads=%d"
            % (clock() - started, np.round(goal_deg, 2).tolist(),
               np.round(actual_deg, 2).tolist() if actual_deg is not None else None,
               np.round(errors_deg, 2).tolist() if errors_deg is not None else None,
               slowest_read_ms, reads)
        )
        raise RuntimeError("A-arm arrival timeout")

    # ------------------------------------------------------------- gripper
    def _move_gripper_ratio(self, ratio):
        self._check_stop()
        ratio = float(np.clip(ratio, 0.0, 1.0))
        if self.gripper is None:
            if self.use_gripper:
                raise RuntimeError("centric gripper connection is unavailable")
            self._animate_gripper_ratio(ratio, 0.7)
            return True
        self._set_gripper_ratio_state(ratio, moving=True)
        ok = self.gripper.move_ratio(ratio, self._stop_event.is_set)
        self._set_gripper_ratio_state(ratio, moving=False)
        return bool(ok)

    def _animate_gripper_ratio(self, target_ratio, duration):
        start = float(self.sim_gripper_ratio)
        started = time.monotonic()
        while True:
            self._check_stop()
            elapsed = time.monotonic() - started
            u = 1.0 if duration <= 0.0 else min(1.0, elapsed / duration)
            smooth = u * u * (3.0 - 2.0 * u)
            ratio = start + (float(target_ratio) - start) * smooth
            self.sim_gripper_ratio = ratio
            self._set_gripper_ratio_state(ratio, moving=u < 1.0)
            if u >= 1.0:
                return
            time.sleep(1.0 / self.sim_hz)

    def _set_gripper_ratio_state(self, ratio, moving):
        self.sim_gripper_ratio = float(ratio)
        angle = float(ratio) * float(
            getattr(self, "centric_full_close_angle_deg", 170.0)
        )
        self._set_sim_state(
            gripper=("MOVING " if moving else "AT ")
            + "%.1f deg / %.0f%% CLOSED"
            % (angle, 100.0 * float(ratio))
        )

    # ---------------------------------------------------------- coordinates
    def world_to_arm(self, world_xyz):
        rail_m = self._current_rail_m()
        return np.asarray(world_xyz, dtype=float) - (
            self.rail_origin + np.array([0.0, rail_m, 0.0])
        )

    def a3_to_world(self, a3_x_mm, a3_y_mm, z_mm):
        robot_x_mm = A3_LT_ROBOT_X_MM - float(a3_y_mm)
        robot_y_mm = float(a3_x_mm) - A3_LT_ROBOT_Y_MM
        arm = _user_mm_to_arm_m(robot_x_mm, robot_y_mm, z_mm)
        return arm + self.rail_origin + np.array(
            [0.0, self.vision_rail_ref, 0.0]
        )

    def _current_rail_m(self):
        if self.rail is not None:
            return float(self.rail.position_mm) / 1000.0
        with self._state_lock:
            return float(self.sim_rail_mm) / 1000.0

    def _read_rail_origin(self):
        try:
            root = ET.parse(self._find_urdf(self.urdf_name)).getroot()
            for joint in root.findall("joint"):
                if joint.get("name") == "joint_rail":
                    origin = joint.find("origin")
                    if origin is not None and origin.get("xyz"):
                        return np.array(
                            [float(v) for v in origin.get("xyz").split()]
                        )
        except Exception as exc:
            self.get_logger().warning(
                "rail origin read failed; assuming [0,0,0]: %s" % exc
            )
        return np.zeros(3)

    # --------------------------------------------------------------- UI
    def _tick(self):
        if self._button_clicked("test_mid_vertical"):
            self._start_vertical_test("MID")
        if self._button_clicked("test_side_vertical"):
            self._start_vertical_test("SIDE")
        if (
            hasattr(self, "preclose_ratio")
            and self._button_clicked("pick_pose_preclose")
        ):
            self._start_auxiliary(
                "A_PICK_POSE_WITH_GRIPPER_PRECLOSE",
                self._pick_pose_with_preclose_only,
            )
        if hasattr(self, "latest_holes") and self._button_clicked("sample"):
            self._accept_holes(SAMPLE_HOLES, source="PyBullet sample")
        super()._tick()

    def _handle_command(self, raw):
        parts = str(raw).strip().lower().split()
        if not parts:
            return
        if parts[0] == "sample":
            self._accept_holes(SAMPLE_HOLES, source="terminal sample")
            return
        if parts[0] in ("pickpose", "grasp_pose"):
            self._start_auxiliary(
                "A_PICK_POSE_WITH_GRIPPER_PRECLOSE",
                self._pick_pose_with_preclose_only,
            )
            return
        if parts[0] in ("testmid", "vmid"):
            self._start_vertical_test("MID")
            return
        if parts[0] in ("testside", "vside"):
            self._start_vertical_test("SIDE")
            return
        if parts[0] in ("testretreat", "vup"):
            self._start_auxiliary(
                "VERTICAL_TEST_RETREAT",
                self._vertical_test_retreat,
            )
            return
        if parts[0] == "coords" and len(parts) >= 5:
            try:
                values = [float(v) for v in parts[1:]]
            except ValueError:
                self.get_logger().error("coords values must be numbers")
                return
            self._accept_holes(values, source="terminal coords")
            return
        joint_parts = parts[1:] if parts[0] in ("joint", "j", "aj") else parts
        if len(joint_parts) == 5:
            try:
                goal_deg = [float(value) for value in joint_parts]
            except ValueError:
                goal_deg = None
            if goal_deg is not None:
                if not all(math.isfinite(value) for value in goal_deg):
                    self.get_logger().error(
                        "joint angles must be five finite numbers in degrees"
                    )
                    return
                self.get_logger().warning(
                    "MANUAL JOINT MOVE: A -> %s deg; check rail/tool clearance"
                    % goal_deg
                )
                self._start_auxiliary(
                    "A_MANUAL_JOINT_MOVE",
                    lambda goal=list(goal_deg): self._move_arm(goal),
                )
                return
        super()._handle_command(raw)

    def _create_target_markers(self):
        # Marker geometry is created in the isolated PyBullet process.
        self._viewer_markers = {}
        self._send_viewer_state()

    def _update_target_markers(self):
        if not self.use_simulation or self.latest_holes is None:
            return
        mid = self.latest_holes[:2]
        side = self.latest_holes[2:4]
        self._move_marker(
            "mid",
            self.a3_to_world(mid[0], mid[1], self.target_z_mm),
        )
        self._move_marker(
            "side",
            self.a3_to_world(side[0], side[1], self.target_z_mm),
        )
        self._viewer_holes_text = (
            "MID green %s | SIDE blue %s"
            % (mid.round(1).tolist(), side.round(1).tolist())
        )
        self._send_viewer_state()

    def _move_marker(self, key, world_position):
        if not self.use_simulation:
            return
        self._viewer_markers[str(key)] = (
            np.asarray(world_position, dtype=float).ravel()[:3].tolist()
        )
        self._send_viewer_state()

    @staticmethod
    def _print_help():
        print("\nBaseplate commands: sample | coords mx my sx sy [angle distance]")
        print(
            "mechhome | herehome | refhome | refcapture | refstatus | start | "
            "stop(s) | status | open | grasp | pickpose | home | railhome | "
            "allinitial | rail <mm>"
        )
        print(
            "Reference pose: [0, 56.7312, -134.3335, 108.6579, 0] deg "
            "(J1/J5는 0도 마크 정렬)"
        )
        print("Startup option A: 기계적 기준 자세 -> MECH LIMIT -> STRAIGHT HOME")
        print("Startup option B: 실제 로봇이 이미 1자 자세 -> CURRENT POSE = STRAIGHT HOME")
        print(
            "Vertical accuracy test: testmid | testside | testretreat "
            "(aliases: vmid | vside | vup)"
        )
        print("Manual A joints: q1 q2 q3 q4 q5  (example: 0 -10 -131 -38 0)")
        print(
            "Keys: SPACE=start, X=stop, H=A home, I=all initial, "
            "O=open, G=grasp, M=motor status"
        )
        print(
            "Rail-only keys: 0=0 mm, 1=620 mm, 2=683 mm, 3=805 mm, 4=864 mm"
        )
        print(
            "PICK POSE button/pickpose: A -> [0,-12,-132,-35,0] while "
            "gripper pre-closes to 100 deg"
        )
        print(
            "TEST MID/SIDE: rail 0 mm -> same vertical insertion path -> "
            "hold at Z=30 mm; RETREAT returns vertically"
        )
        print("Load coordinates first, then press START PROCESS.\n")

    def destroy_node(self):
        self._stop_collision_process()
        super().destroy_node()




def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = _BaseplateCentricSequenceNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
