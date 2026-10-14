"""Combined carpart/baseplate process using isolated PyBullet from centric_sequence.

Requires the laptop's existing centric_sequence_5axis,
dxl_bridge_centric_5axis, centric_gripper_5axis and B_arm_v2_tolerance modules.
The included cell_rail_serial provides the suction/rail ACK interface.
This process owns /dev/ttyUSB0 (A and removable centric ID8),
/dev/ttyUSB1 (B) and /dev/ttyACM0 (rail and suction).
"""
import json
import math
import os
import threading
import time
import uuid

import numpy as np
import rclpy
from std_msgs.msg import String
from dynamixel_sdk import PacketHandler, PortHandler

from .B_arm_v2_tolerance import DxlConfig, DynamixelArmB, RobotCalibration
from .centric_gripper_5axis import CentricGripper
from .centric_sequence_5axis import (
    A3_LT_ROBOT_X_MM, A3_LT_ROBOT_Y_MM, DxlBridge,
    JOINT_MOTORS, PROFILE_VELOCITY,
    N_JOINTS, SequenceStopped, _BaseplateCentricSequenceNode, fk_pos, plan_joint_path,
)
from .cell_rail_serial import RAIL_MAX_M, RailSerial


class Cell(_BaseplateCentricSequenceNode):
    """A single owner for A, B, rail, suction and the removable gripper."""

    def _probe_arm(self, label, port, baud, ids):
        """Read-only motor ping before torque, rail zero or any cell motion."""
        if not os.path.exists(port):
            self.get_logger().error('%s 로봇 연결 안 됨: 포트 %s가 없습니다 (ID %s)'
                                    % (label, port, ','.join(map(str, ids))))
            return False
        handler = None
        try:
            handler = PortHandler(port)
            if not handler.openPort():
                raise RuntimeError('포트를 열 수 없습니다 (권한 또는 다른 프로세스 사용 여부 확인)')
            if not handler.setBaudRate(baud):
                raise RuntimeError('%d bps 설정 실패' % baud)
            packet = PacketHandler(2.0)
            missing = []
            for motor_id in ids:
                error = '응답 없음'
                for _ in range(2):
                    try:
                        _, comm, dxl_error = packet.ping(handler, motor_id)
                        if comm == 0 and dxl_error == 0:
                            error = None
                            break
                        error = ('통신=%s, 모터오류=%s'
                                 % (packet.getTxRxResult(comm), dxl_error))
                    except Exception as exc:
                        error = str(exc)
                if error is not None:
                    missing.append('%d(%s)' % (motor_id, error))
            if missing:
                self.get_logger().error('%s 로봇 연결 안 됨: %s, 응답 없는 ID: %s'
                                        % (label, port, ', '.join(missing)))
                return False
            self.get_logger().info('%s 로봇 연결 확인: %s, ID %s'
                                   % (label, port, ','.join(map(str, ids))))
            return True
        except Exception as exc:
            self.get_logger().error('%s 로봇 연결 안 됨: %s (%s)'
                                    % (label, port, exc))
            return False
        finally:
            if handler is not None:
                try:
                    handler.closePort()
                except Exception:
                    pass

    def _setup_hardware(self, g):
        # The parent creates the PyBullet viewer in another OS process first.
        # Both suction and rail use the SAME Arduino serial owner here.
        self.declare_parameter('baseplate_test_mode', False)
        self.baseplate_test_mode = bool(g('baseplate_test_mode').value)
        self.declare_parameter('use_b_robot', False)
        self.declare_parameter('b_dxl_port', '/dev/ttyUSB1')
        self.declare_parameter('b_dxl_baud', 2000000)
        self.declare_parameter('b_move_timeout_sec', 25.0)
        self.declare_parameter('assembly_debug_mode', False)
        self.assembly_debug_mode = bool(g('assembly_debug_mode').value)
        try:
            if self.use_robot:
                if not self.assembly_debug_mode and not bool(g('use_b_robot').value):
                    raise RuntimeError('실물 전체 공정에는 use_b_robot:=true가 필요합니다')
                a_port = str(g('dxl_port').value)
                b_port = str(g('b_dxl_port').value)
                if not self.assembly_debug_mode and os.path.realpath(a_port) == os.path.realpath(b_port):
                    raise RuntimeError('A/B 로봇은 서로 다른 U2D2 포트가 필요합니다')
                a_ok = self._probe_arm('A팔', a_port, int(g('dxl_baud').value), range(1, 8))
                b_ok = (True if self.assembly_debug_mode else
                        self._probe_arm('B팔', b_port, int(g('b_dxl_baud').value), range(9, 17)))
                if not (a_ok and b_ok):
                    raise RuntimeError('A/B 로봇 연결 확인 실패: 위 로그의 팔/모터 ID를 확인하세요')
        except Exception:
            if self.use_simulation:
                self._stop_simulation_process()
            raise
        try:
            if self.use_rail:
                try:
                    self.rail = RailSerial(port=str(g('rail_port').value),
                                           baud=int(g('rail_baud').value),
                                           logger=self.get_logger())
                    self.rail.connect()
                    if bool(g('rail_auto_zero').value):
                        if not self.rail.zero(wait=True, timeout=3.0):
                            raise RuntimeError('rail zero acknowledgement missing')
                    if not self.rail.zero_set:
                        raise RuntimeError('rail zero is unset')
                    if not self.rail.suction(False, wait=True, timeout=2.0):
                        raise RuntimeError('suction OFF acknowledgement missing')
                except Exception as exc:
                    self.get_logger().error('레일/흡착 컨트롤러 연결 또는 초기화 실패: %s' % exc)
                    raise
            if self.use_robot:
                self.bridge = DxlBridge(port=str(g('dxl_port').value),
                                        baud=int(g('dxl_baud').value))
                try:
                    self.bridge.connect()
                    # The physical straight pose at startup is q=[0,0,0,0,0].
                    self.bridge.capture_home()
                    self.reference_ready = True
                    self.get_logger().info('A팔 초기 영점 저장 완료 (실물 1자 자세 기준)')
                except Exception as exc:
                    self.get_logger().error('A팔 연결/초기화 실패: %s' % exc)
                    raise
            # ID 8 is physically absent while suction tool is installed.
            self.use_gripper = False
            self.gripper = None
        except Exception:
            self._close_hardware()
            self._stop_simulation_process()
            raise

    def __init__(self):
        super().__init__()
        for key, value in (
            ('simulate_inspection', False), ('side_timeout_sec', 8.0),
            ('top_camera_enabled', True), ('arm_camera_enabled', True),
            ('top_z_mm', 30.0), ('base_z_mm', 30.0),
            ('suction_dwell_sec', 0.5),
            ('assembly_holes_timeout_sec', 180.0),
            ('good_rail_mm', 350.0), ('reject_rail_mm', 550.0),
            ('sort_pose_deg', [180.0, -25.0, -35.0, -50.0, 0.0]),
            ('reject_sort_pose_deg', [180.0, -25.0, -35.0, -50.0, 0.0]),
            ('base_release_pose_deg', [0.0, 0.0, 0.0, 0.0, 0.0]),
            ('base_assembly_x_mm', 210.0),
            ('base_assembly_y_mm', 140.0),
            ('assembly_release_open_deg', 20.0),
            ('lid_rail_mm', 680.0),
            ('lid_pick_pose_deg', '0 -45 -65 -65 0'),
            ('lid_approach_z_mm', 80.0), ('lid_probe_start_z_mm', 50.0),
            ('lid_probe_extra_mm', 40.0), ('lid_probe_load_pct', 30.0),
            ('lid_probe_poll_sec', 0.05), ('lid_probe_hold_sec', 1.0),
            ('tool_change_timeout_sec', 600.0),
            ('inspection_j5_tolerance_deg', 2.0),
        ):
            self.declare_parameter(key, value)
        g = lambda key: self.get_parameter(key).value
        if self.use_robot and not self.use_rail:
            raise RuntimeError('real suction requires Arduino rail/suction controller')
        self.b_arm = None
        self.b_ready = not self.use_robot
        self.use_b_robot = bool(g('use_b_robot')) and not self.assembly_debug_mode
        if self.use_robot and not self.use_b_robot and not self.assembly_debug_mode:
            raise RuntimeError('real two-arm process requires use_b_robot:=true')
        if self.use_b_robot:
            if not self.use_robot:
                raise RuntimeError('B robot requires use_robot:=true')
            if str(g('b_dxl_port')) == str(g('dxl_port')):
                raise RuntimeError('A and B must use separate U2D2 ports')
            cfg = DxlConfig(device_name=str(g('b_dxl_port')),
                            baudrate=int(g('b_dxl_baud')))
            cfg.move_timeout_sec = float(g('b_move_timeout_sec'))
            self.b_arm = DynamixelArmB(cfg, RobotCalibration())
            try:
                self.b_arm.open()
                # ping_all() prints failures but does not raise; the preflight
                # above independently checks every ID and blocks absent arms.
                self.b_arm.set_extended_position_mode_all()
                self.b_arm.ping_all()
                self.b_arm.set_current_as_zero()
                self.b_ready = True
                self.get_logger().info('B팔 초기 영점 저장 완료 (현재 실물 자세 기준)')
            except Exception as exc:
                self.get_logger().error('B팔 연결/초기화 실패: %s' % exc)
                self.b_arm.close()
                self._close_hardware()
                self._stop_simulation_process()
                raise
        self.simulate_inspection = bool(g('simulate_inspection'))
        self.top_camera_enabled = bool(g('top_camera_enabled'))
        self.arm_camera_enabled = bool(g('arm_camera_enabled'))
        self.side_timeout = float(g('side_timeout_sec'))
        self.top_z = float(g('top_z_mm'))
        self.base_z = float(g('base_z_mm'))
        self.suction_dwell = max(0.0, float(g('suction_dwell_sec')))
        self.good_rail = float(g('good_rail_mm'))
        self.reject_rail = float(g('reject_rail_mm'))
        self.base_assembly_point = np.array(
            [float(g('base_assembly_x_mm')), float(g('base_assembly_y_mm'))],
            dtype=float)
        if not np.isfinite(self.base_assembly_point).all() or \
                not self._in_a3(self.base_assembly_point):
            raise ValueError('base assembly XY must be inside calibrated A3')
        self.assembly_release_open_deg = float(g('assembly_release_open_deg'))
        if not 0.0 < self.assembly_release_open_deg <= 180.0:
            raise ValueError('assembly_release_open_deg must be in (0, 180]')
        self.sort_pose = [float(x) for x in g('sort_pose_deg')]
        self.reject_sort_pose = [float(x) for x in g('reject_sort_pose_deg')]
        self.lid_rail = float(g('lid_rail_mm'))
        self.lid_approach_z = float(g('lid_approach_z_mm'))
        self.lid_probe_start_z = float(g('lid_probe_start_z_mm'))
        self.lid_probe_extra = float(g('lid_probe_extra_mm'))
        self.lid_probe_load = float(g('lid_probe_load_pct'))
        self.lid_probe_poll = float(g('lid_probe_poll_sec'))
        self.lid_probe_hold = float(g('lid_probe_hold_sec'))
        self.tool_timeout = float(g('tool_change_timeout_sec'))
        # The removable ID8 is constructed at tool change. Keep the angle
        # used for preclose ratios identical to its configured travel limit.
        grip_limit = float(g('centric_planet_max_angle_deg'))
        if not math.isfinite(grip_limit) or grip_limit < max(
                self.centric_preclose_angle_deg,
                self.bearing_preclose_angle_deg,
                self.gear_preclose_angle_deg):
            raise ValueError('centric travel limit must cover all preclose angles')
        if self.assembly_release_open_deg > grip_limit:
            raise ValueError('assembly release opening exceeds centric travel limit')
        self.centric_full_close_angle_deg = grip_limit
        self.preclose_ratio = self.centric_preclose_angle_deg / grip_limit
        self.inspection_j5_tolerance_deg = float(g('inspection_j5_tolerance_deg'))
        if not 1.5 <= self.inspection_j5_tolerance_deg <= 2.0:
            raise ValueError('inspection_j5_tolerance_deg must be between 1.5 and 2.0')
        self.holes_timeout = float(g('assembly_holes_timeout_sec'))
        if self.holes_timeout <= 0:
            raise ValueError('assembly_holes_timeout_sec must be positive')
        self.lid_pick_pose = self._parse_lid_pose(str(g('lid_pick_pose_deg')))
        if any(not 0 <= v <= RAIL_MAX_M * 1000 for v in
               (self.good_rail, self.reject_rail, self.lid_rail)):
            raise ValueError('sorting/lid rail target outside software stroke')
        if len(self.sort_pose) != 5 or not all(math.isfinite(x) for x in self.sort_pose):
            raise ValueError('sort_pose_deg needs five finite angles')
        if len(self.reject_sort_pose) != 5 or not all(math.isfinite(x) for x in self.reject_sort_pose):
            raise ValueError('reject_sort_pose_deg needs five finite angles')
        if not (self.lid_approach_z > self.lid_probe_start_z > 0 and
                0 < self.lid_probe_extra < self.lid_probe_start_z and
                0 < self.lid_probe_load <= 100 and
                self.lid_probe_poll > 0 and self.lid_probe_hold >= 0):
            raise ValueError('invalid lid approach/probe parameters')
        self.target = None
        self._test_top = None
        self._test_holding = False
        self._test_result = None
        self._test_records = []
        self._test_last_view = None
        self._assembly_cycle_active = False
        self._grasp_hold_position = None
        self._grasp_hold_ratio = None
        self.tool = 'suction'
        self._last_b_pose = None
        self._resume_event = threading.Event()
        self._pause_event = threading.Event()
        self._pause_started = None
        self._paused_seconds = 0.0
        self._motion_epoch = 0
        self._paused_phase = None
        self._tool_change_waiting = False
        self._pause_hold_ok = True
        self._pause_needs_recovery = False
        self._paused_a_rad = None
        self._paused_b_ticks = None
        self._paused_rail_mm = None
        self._paused_suction_on = None
        self._result_event = threading.Event()
        self._last_result = None
        self._waiting_request = None
        self.cycle_id = None
        self.ready_pub = self.create_publisher(String, '/cell/ready', 10)
        self.side_pub = self.create_publisher(String, '/cell/side_command', 10)
        self.hole_pub = self.create_publisher(String, '/cell/hole_scan', 10)
        self.test_result_pub = self.create_publisher(String, '/cell/baseplate_test_result', 10)
        self.create_subscription(String, '/cell/top_result', self._on_top, 10)
        self.create_subscription(String, '/cell/side_result', self._on_side, 10)
        self.get_logger().info('A/B current poses captured as zero; commands: resume, stop')
        if self.baseplate_test_mode:
            self.auto_start_on_vision = False
            self.get_logger().info('Test cameras: TOP=%s, B=%s' %
                                   (self.top_camera_enabled, self.arm_camera_enabled))
            self.get_logger().info('1자 시작 자세를 영점으로 저장했습니다. '
                                   'aj/bj/bottom/back/flash 명령을 바로 사용할 수 있습니다. '
                                   'mechhome은 기계적 끝점에서 실측 틱 이동량으로 복귀할 때 사용합니다.')
            self._test_help()

    @staticmethod
    def _parse_lid_pose(raw):
        if not raw.strip():
            return None
        try:
            values = [float(x) for x in raw.replace(',', ' ').split()]
        except ValueError as exc:
            raise ValueError('lid_pick_pose_deg needs five numbers') from exc
        if len(values) != 5 or not all(math.isfinite(x) for x in values):
            raise ValueError('lid_pick_pose_deg needs five finite angles')
        return values

    def _on_top(self, msg):
        if self.baseplate_test_mode and not self.top_camera_enabled:
            return
        try:
            d = json.loads(msg.data)
            kind = str(d['product']).lower()
            judge = str(d['judge']).upper()
            if kind not in ('carpart', 'baseplate') or judge not in ('OK', 'NG'):
                raise ValueError('unexpected product/judge')
            # Baseplate pick XY comes from the fresh hole scan requested below.
            # Only the carpart top detector supplies its own pick coordinates.
            point = None
            if kind == 'carpart':
                point = [float(d['x_mm']), float(d['y_mm'])]
                if not self._in_a3(point) or not all(np.isfinite(point)):
                    raise ValueError('target outside A3')
            if self.busy:
                self.get_logger().warning('Top result ignored: cell busy')
                return
            self.target = dict(product=kind, judge=judge, point=point)
            if self.baseplate_test_mode:
                if kind == 'baseplate':
                    self._test_top = dict(self.target)
                    self.get_logger().info('Baseplate TOP=%s cached; enter baseplate to pick' % judge)
                self.target = None
                return
            self._start_sequence()
        except (ValueError, KeyError, TypeError) as exc:
            self.get_logger().error('Invalid top result: %s' % exc)

    def _on_side(self, msg):
        try:
            d = json.loads(msg.data)
        except ValueError:
            return
        if self._waiting_request and d.get('request_id') == self._waiting_request:
            if d.get('cycle_id') == self.cycle_id:
                self._last_result = d
                self._result_event.set()

    def _on_holes(self, msg):
        # The parent rejects new detections while busy. This cell explicitly
        # requests a fresh measurement midway through the same busy cycle.
        if self.busy and getattr(self, '_holes_needed_after', None) is not None:
            values = np.asarray(msg.data, dtype=float)
            if values.size >= 6 and np.isfinite(values[:4]).all() and \
                    self._in_a3(values[:2]) and self._in_a3(values[2:4]):
                self.latest_holes = values[:6].copy()
                self.latest_holes_stamp = time.monotonic()
                self.get_logger().info('Baseplate holes received: mid=(%.3f, %.3f) mm, '
                                       'side=(%.3f, %.3f) mm'
                                       % tuple(self.latest_holes[:4]))
            else:
                self.get_logger().error('Invalid assembly hole measurement')
            return
        super()._on_holes(msg)

    def _handle_command(self, raw):
        command = str(raw).strip().lower()
        parts = command.split()
        if self.baseplate_test_mode and self._handle_test_command(parts):
            return
        if parts and parts[0] in ('aj', 'bj', 'r'):
            name = parts[0]
            if self.busy or self._pause_event.is_set() or self._tool_change_waiting:
                self.get_logger().warning('%s unavailable during a process, pause, or tool change'
                                          % name)
                return
            if self.phase == 'ERROR_HOLDING':
                self.get_logger().error('%s blocked: fix the hardware fault before moving'
                                        % name)
                return
            expected = 2 if name == 'r' else (6 if name == 'aj' else 7)
            if len(parts) != expected:
                self.get_logger().error('usage: aj q1 q2 q3 q4 q5 | '
                                        'bj q1 q2 q3 q4 q5 q6 | r mm')
                return
            try:
                values = [float(value) for value in parts[1:]]
            except ValueError:
                self.get_logger().error('%s needs numeric values' % name)
                return
            if not all(math.isfinite(value) for value in values):
                self.get_logger().error('%s needs finite numeric values' % name)
                return
            if name == 'aj':
                if self.use_robot and (not self.reference_ready or self.bridge is None):
                    self.get_logger().error('A arm is unavailable or its zero is not ready')
                    return
                self._start_auxiliary('A_MANUAL_JOINT_MOVE',
                                      lambda goal=values: self._move_arm(goal))
            elif name == 'bj':
                if self.use_robot and (not self.b_ready or self.b_arm is None):
                    self.get_logger().error('B arm is unavailable or its zero is not ready')
                    return
                self._start_auxiliary('B_MANUAL_JOINT_MOVE',
                                      lambda goal=values: self._b_move(goal))
            else:
                if self.use_robot and (not self.use_rail or self.rail is None):
                    self.get_logger().error('rail is unavailable')
                    return
                self._start_rail_only(values[0])
            return
        if parts and (parts[0] in (
                'start', 'run', 'rail', 'testretreat', 'vup',
                'mechhome', 'mechstraight', 'limit2home',
                'herehome', 'currenthome', 'currentstraight')):
            self.get_logger().warning('This manual cell command has been removed: %s'
                                      % parts[0])
            return
        if command == 'resume':
            if self._pause_event.is_set():
                self._resume_cycle()
            elif self._tool_change_waiting:
                self._confirm_tool_change()
            else:
                self.get_logger().warning('No paused process or tool change to resume')
        elif command in ('tool_done', 'tooldone'):
            self._confirm_tool_change()
        elif command in ('bhome', 'b_home'):
            self._start_b_home()
        elif command in ('abort', 'abort_cycle'):
            self._abort_cycle()
        elif command in ('stop', 's', 'estop', '0'):
            self._pause_cycle()
        else:
            super()._handle_command(raw)

    def _tick(self):
        # Consume cell-specific commands before the inherited E-stop handler
        # discards the worker's call stack.
        self._poll_viewer_commands()
        if getattr(self, 'baseplate_test_mode', False):
            # Old assembly/viewer buttons cannot start inherited assembly cycles.
            if self._button_clicked('stop'):
                self._pause_cycle()
            if self._button_clicked('resume_cycle'):
                self._resume_cycle()
            if self._button_clicked('abort_cycle'):
                self._abort_cycle()
            self._send_viewer_state()
            return
        if self._button_clicked('stop'):
            self._pause_cycle()
        if self._button_clicked('resume_cycle'):
            self._resume_cycle()
        if self._button_clicked('abort_cycle'):
            self._abort_cycle()
        if self._button_clicked('tool_done'):
            self._confirm_tool_change()
        if self._button_clicked('b_home'):
            self._start_b_home()
        super()._tick()

    def _process_time(self):
        # Timeouts and scheduled trajectory samples do not run while paused.
        with self._state_lock:
            paused_at = self._pause_started
            offset = self._paused_seconds
        return (paused_at if paused_at is not None else time.monotonic()) - offset

    def _check_stop(self):
        super()._check_stop()
        event = getattr(self, '_pause_event', None)
        while event is not None and event.is_set():
            super()._check_stop()
            time.sleep(.05)
        super()._check_stop()

    def _pause_cycle(self):
        if not self.busy:
            self.get_logger().warning('No active cell process to pause')
            return
        if self._pause_event.is_set():
            return
        with self._state_lock:
            self._paused_phase = self.phase
            self._pause_started = time.monotonic()
            self._pause_event.set()
            self._pause_hold_ok = True
            self._pause_needs_recovery = (
                self.phase.endswith(('_FINAL_GRASP', '_RELEASE')) or
                '_PICK_AND_PRECLOSE_' in self.phase or
                self.phase == 'LID_VERTICAL_PROBE')
        self.side_pub.publish(String(data=json.dumps({'command': 'STOP'})))
        self.hole_pub.publish(String(data='STOP'))
        if self.rail is not None:
            try:
                if not self.rail.estop():
                    raise RuntimeError('rail stop command was not sent')
                self._paused_rail_mm = self.rail.position_mm
                self._paused_suction_on = self.rail.suction_on
            except Exception as exc:
                self._pause_hold_ok = False
                self.get_logger().error('Rail hold failed: %s' % exc)
        if self.bridge is not None:
            try:
                self.bridge.emergency_stop()
                self._paused_a_rad = np.asarray(self.bridge.read_joints(), dtype=float)
            except Exception as exc:
                self._pause_hold_ok = False
                self.get_logger().error('A-arm hold failed: %s' % exc)
        if self.b_arm is not None:
            try:
                present = self.b_arm.read_present_ticks(retry=1, use_fallback=False)
                self.b_arm.write_goal_ticks(present)
                self._paused_b_ticks = present
            except Exception as exc:
                self._pause_hold_ok = False
                self.get_logger().error('B-arm hold failed: %s' % exc)
        if self.gripper is not None:
            try:
                if self.gripper.emergency_stop('cell paused') is None:
                    raise RuntimeError('centric hold not confirmed')
            except Exception as exc:
                self._pause_hold_ok = False
                self.get_logger().error('Centric hold failed: %s' % exc)
        self._set_phase('PROCESS_PAUSED')
        self.get_logger().warning('Motion paused; inspect the cell, then press RESUME PAUSED PROCESS')

    def _resume_cycle(self):
        if not self._pause_event.is_set() or not self.busy:
            self.get_logger().warning('No paused cell process to resume')
            return
        if not self._pause_hold_ok:
            self.get_logger().error('Hold failed on one or more axes; automatic resume blocked')
            return
        if self._pause_needs_recovery:
            self.get_logger().error('Paused during contact/gripper operation; '
                                    'automatic resume blocked. Inspect the tool and '
                                    'use ABORT PAUSED PROCESS before restarting.')
            return
        # Read both arms before releasing the waiting worker. Communication
        # failures leave the process paused at its held positions.
        try:
            if self.bridge is not None:
                a_now = np.asarray(self.bridge.read_joints(), dtype=float)
                a_delta = np.degrees(np.abs(np.arctan2(
                    np.sin(a_now[:5] - self._paused_a_rad[:5]),
                    np.cos(a_now[:5] - self._paused_a_rad[:5]))))
                if np.max(a_delta) > 5.0:
                    raise RuntimeError('A arm changed >5 deg during pause')
            if self.b_arm is not None:
                b_now = self.b_arm.read_present_ticks(retry=1, use_fallback=False)
                b_joint_now = self.b_arm.mapper.present_tick_to_joint_deg(b_now)
                b_joint_held = self.b_arm.mapper.present_tick_to_joint_deg(self._paused_b_ticks)
                if max(abs(b_joint_now[k] - b_joint_held[k]) for k in b_joint_held) > 5.0:
                    raise RuntimeError('B arm changed >5 deg during pause')
            if self.rail is not None:
                self.rail.query()
                if abs(self.rail.position_mm - self._paused_rail_mm) > 5.0:
                    raise RuntimeError('rail position changed >5 mm during pause')
                if self.rail.suction_on != self._paused_suction_on:
                    raise RuntimeError('suction state changed during pause')
        except Exception as exc:
            self.get_logger().error('Resume preflight failed; still paused: %s' % exc)
            return
        with self._state_lock:
            self._paused_seconds += time.monotonic() - self._pause_started
            self._pause_started = None
            self._motion_epoch += 1
            self._pause_event.clear()
            phase = self._paused_phase
            self._paused_phase = None
        self._set_phase(phase or 'CELL_PROCESS')
        self.get_logger().info('Resuming the interrupted stage (epoch %d)' % self._motion_epoch)

    def _abort_cycle(self):
        if not self._pause_event.is_set():
            self.get_logger().warning('No paused process to abort')
            return
        # Cancellation removes the worker's stack; the robot remains held.
        self._stop_event.set()
        self._pause_event.clear()
        self._resume_event.set()
        self._result_event.set()
        self.get_logger().warning('Paused process aborted; held positions remain unchanged')

    def _confirm_tool_change(self):
        if self._tool_change_waiting and not self._pause_event.is_set():
            self._resume_event.set()
            self.get_logger().info('Manual tool change acknowledged')
        else:
            self.get_logger().warning('Tool confirmation ignored: no active tool-change wait')

    def _start_b_home(self):
        if self.busy or self._pause_event.is_set():
            self.get_logger().warning('B HOME unavailable during a process or pause')
            return
        if self.use_robot and not self.b_ready:
            self.get_logger().error('B arm not connected')
            return
        self._start_auxiliary('B_RETURN_HOME', lambda: self._b_move([90, 0, 0, 0, 0, 0]))

    def _start_sequence(self):
        if getattr(self, 'baseplate_test_mode', False):
            self.get_logger().info('Test mode waits for baseplate or inspect command')
            return
        if self.busy:
            self.get_logger().warning('Cell is busy; resume or finish the current process')
            return
        if not hasattr(self, 'target') or self.target is None:
            self.get_logger().warning('Waiting for classified top-view target')
            return
        if self.use_robot and (not self.reference_ready or not self.b_ready):
            self.get_logger().error('A/B startup zero capture is not ready')
            return
        if self.use_robot and (self.bridge is None or self.rail is None or self.b_arm is None):
            self.get_logger().error('Hardware unavailable')
            return
        if self.target['product'] == 'baseplate' and self.target['judge'] == 'OK' \
                and self.lid_pick_pose is None:
            self.get_logger().error('Baseplate start blocked: lid pick pose is missing')
            return
        self.cycle_id = uuid.uuid4().hex[:12]
        self._holes_needed_after = None
        self._pause_event.clear()
        self._paused_seconds = 0.0
        self._pause_started = None
        self._start_auxiliary('CELL_PROCESS', self._run_cell)

    def _execute_joint_trajectory(self, trajectory):
        """Rejoin an interrupted path from the measured pose at planner speed."""
        path = np.asarray(trajectory, dtype=float)
        if path.ndim != 2 or not len(path):
            raise RuntimeError('empty A-arm trajectory')
        if path.shape[1] == N_JOINTS:
            q5 = self._current_q_rad()[4]
            path = np.column_stack((path, np.full(len(path), q5)))
        epoch = self._motion_epoch
        index = 0
        started = self._process_time()
        last_command = 0.0
        while index < len(path):
            if self._motion_epoch != epoch:
                if getattr(self, '_dual_b_goal', None) is not None:
                    self._b_command(self._dual_b_goal)
                actual = self._current_q_rad()
                # Rejoin at the interrupted sample; a large discrepancy
                # requires operator review instead of an unsafe jump.
                target = path[index, :5]
                delta = np.degrees(np.abs(np.arctan2(
                    np.sin(actual - target), np.cos(actual - target))))
                tcp_gap_mm = 1000.0 * np.linalg.norm(
                    fk_pos(actual[:4]) - fk_pos(target[:4]))
                if np.max(delta) > 8.0 or tcp_gap_mm > 15.0:
                    raise RuntimeError('A-arm resume path differs from held pose: '
                                       'joint=%s deg, TCP=%.1f mm; inspect manually'
                                       % (np.round(delta, 2).tolist(), tcp_gap_mm))
                reconnect, _, _ = plan_joint_path(
                    actual, target, hz=self.planner_hz,
                    speed_deg_per_s=self.joint_path_speed)
                path = np.vstack((reconnect, path[index + 1:]))
                error = self._collision_error(path[:, :N_JOINTS], 'resume A motion')
                if error:
                    raise RuntimeError(error)
                self.get_logger().info('A-arm resumed at measured pose, remaining %d samples'
                                       % len(path))
                epoch = self._motion_epoch
                index = 0
                started = self._process_time()
                last_command = 0.0
            target_time = started + index / self.planner_hz
            while self._process_time() < target_time:
                self._check_stop()
                if epoch != self._motion_epoch:
                    break
                time.sleep(min(.01, max(.001, target_time - self._process_time())))
            self._check_stop()
            if epoch != self._motion_epoch:
                continue
            q = path[index]
            now = self._process_time()
            if self.bridge is not None and (now >= last_command + 1.0 / self.robot_cmd_hz
                                             or index == len(path) - 1):
                self.bridge.command_joints(q[:5].tolist())
                last_command = now
            self._set_sim_state(q_deg=np.degrees(q[:5]).tolist())
            index += 1
        if self.bridge is not None:
            self._wait_arm_goal(path[-1, :5])

    def _resume_concurrent_motion(self):
        if getattr(self, '_dual_b_goal', None) is not None:
            self._b_command(self._dual_b_goal)

    def _wait_arm_goal(self, goal_rad):
        # The stop command replaces the motor goal with a hold. Reassert the
        # destination on entry in case pause happened at the path boundary.
        if self.bridge is not None:
            self._check_stop()
            self._resume_concurrent_motion()
            self.bridge.command_joints(np.asarray(goal_rad[:5]).tolist())
        return super()._wait_arm_goal(goal_rad)

    def _arrival_tolerances_deg(self):
        thresholds = np.full(5, math.degrees(self.joint_tol_rad))
        if self.phase.endswith('_VIEW') or self.phase.startswith('CARPART_'):
            thresholds[4] = self.inspection_j5_tolerance_deg
        return thresholds

    def _suction(self, on):
        self._check_stop()
        if self.use_robot:
            if self.rail is None or not self.rail.suction(bool(on), wait=True, timeout=2.0):
                raise RuntimeError('suction acknowledgement failed')
        self._set_sim_state(gripper='SUCTION ON' if on else 'SUCTION OFF')
        if on and self.suction_dwell:
            deadline = time.monotonic() + self.suction_dwell
            while time.monotonic() < deadline:
                self._check_stop()
                time.sleep(min(.05, deadline - time.monotonic()))

    def _move_gripper_ratio(self, ratio):
        if self._assembly_cycle_active and self.phase.endswith('_RELEASE'):
            if self.gripper is not None:
                if self._grasp_hold_position is None:
                    raise RuntimeError('assembly release has no recorded grasp position')
                travel = int(round(
                    self.gripper.max_move_pulse * self.assembly_release_open_deg
                    / self.centric_full_close_angle_deg))
                target = self.gripper.clamp_position(
                    self._grasp_hold_position - self.gripper.close_direction * travel)
                ratio = ((target - self.gripper.open_position)
                         * self.gripper.close_direction / self.gripper.safe_move_pulse)
                self.get_logger().info(
                    'Assembly release: grasp=%d, open target=%d (%.1f planet deg)'
                    % (self._grasp_hold_position, target,
                       self.assembly_release_open_deg))
            else:
                if self._grasp_hold_ratio is None:
                    raise RuntimeError('assembly release has no simulated grasp position')
                ratio = max(0.0, self._grasp_hold_ratio -
                            self.assembly_release_open_deg /
                            self.centric_full_close_angle_deg)
        if self.gripper is None:
            return super()._move_gripper_ratio(ratio)
        self._check_stop()
        ratio = float(np.clip(ratio, 0.0, 1.0))
        self._set_gripper_ratio_state(ratio, moving=True)
        try:
            result = self.gripper.move_ratio(
                ratio, lambda: self._pause_event.is_set() or self._stop_event.is_set())
        finally:
            self._set_gripper_ratio_state(ratio, moving=False)
        self._check_stop()
        return bool(result)

    def _finish_grasp(self):
        self._grasp_hold_position = None
        self._grasp_hold_ratio = None
        if self.full_grasp_mode == 'fixed_100' or self.gripper is None:
            result = super()._finish_grasp()
            if self.gripper is not None:
                self._grasp_hold_position = self.gripper.read_position()
            else:
                self._grasp_hold_ratio = float(self.sim_gripper_ratio)
            return result
        self._check_stop()
        self._set_sim_state(gripper='AUTO GRASPING')
        result = self.gripper.auto_grasp(
            lambda: self._pause_event.is_set() or self._stop_event.is_set(),
            min_grasp_travel_pulse=0)
        self._check_stop()
        if not result.success:
            raise RuntimeError('centric grasp failed: ' + result.reason)
        self._grasp_hold_position = int(result.position)
        self._set_sim_state(gripper='HOLDING')

    def _pick_and_place_cycle(self, *args, **kwargs):
        self._assembly_cycle_active = True
        self._grasp_hold_position = None
        self._grasp_hold_ratio = None
        try:
            return super()._pick_and_place_cycle(*args, **kwargs)
        finally:
            self._assembly_cycle_active = False
            self._grasp_hold_position = None
            self._grasp_hold_ratio = None

    def _vertical_at(self, point, z_mm, action):
        approach, descent, ascent, _ = self._build_vertical_paths(
            point[0], point[1], target_z_mm=z_mm)
        self._execute_joint_trajectory(approach)
        self._execute_joint_trajectory(descent)
        action()
        self._execute_joint_trajectory(ascent)

    def _b_move(self, pose):
        self._b_command(pose)
        self._b_wait(pose)

    def _b_command(self, pose):
        self._check_stop()
        if self.b_arm is None:
            if self.use_robot:
                raise RuntimeError('B arm is unavailable')
            self._last_b_pose = tuple(pose)
            return
        q = dict(zip(('q1', 'q2', 'q3', 'q4', 'q5', 'q6'), map(float, pose)))
        self.b_arm.move_joint_deg(q, wait=False)
        self._last_b_goal = tuple(pose)

    def _b_wait(self, pose):
        self._check_stop()
        if self.b_arm is None:
            self._last_b_pose = tuple(pose)
            return
        q = dict(zip(('q1', 'q2', 'q3', 'q4', 'q5', 'q6'), map(float, pose)))
        goal = self.b_arm.mapper.joint_deg_to_goal_tick(q)
        end = self._process_time() + self.b_arm.cfg.move_timeout_sec
        epoch = self._motion_epoch
        while self._process_time() < end:
            self._check_stop()
            if epoch != self._motion_epoch:
                self.b_arm.move_joint_deg(q, wait=False)
                epoch = self._motion_epoch
            present = self.b_arm.read_present_ticks()
            if all(abs(present[i] - goal[i]) <=
                   self.b_arm.cfg.position_tolerance_tick_by_id.get(
                       i, self.b_arm.cfg.position_tolerance_tick) for i in goal):
                self._last_b_pose = tuple(pose)
                return
            time.sleep(.1)
        raise RuntimeError('B arrival timeout')

    def _dual_pose(self, a_pose, b_pose):
        """Start B before streaming A; both must settle before inspection."""
        self._check_stop()
        self._assert_reference_ready()
        from_q = self._current_q_rad()
        to_q = np.radians(np.asarray(a_pose, dtype=float))
        trajectory, _, _ = plan_joint_path(
            from_q, to_q, hz=self.planner_hz,
            speed_deg_per_s=self.joint_path_speed)
        error = self._collision_error(trajectory[:, :N_JOINTS], 'dual A/B motion')
        if error:
            raise RuntimeError(error)
        self._dual_b_goal = tuple(b_pose)
        try:
            self._b_command(b_pose)
            self._execute_joint_trajectory(trajectory)
        finally:
            self._dual_b_goal = None
        self._b_wait(b_pose)

    def _inspect(self, view):
        self._check_stop()
        if self.simulate_inspection and not self.use_robot:
            time.sleep(.1)
            return 'OK'
        if self.baseplate_test_mode and not self.arm_camera_enabled:
            self.get_logger().info('%s: B camera disabled; observation skipped' % view)
            return 'SKIPPED'
        request = uuid.uuid4().hex[:12]
        self._waiting_request = request
        self._last_result = None
        self._result_event.clear()
        self.side_pub.publish(String(data=json.dumps(dict(
            command='START', product=self.target['product'],
            view=view, cycle_id=self.cycle_id, request_id=request))))
        deadline = self._process_time() + self.side_timeout
        epoch = self._motion_epoch
        try:
            while self._process_time() < deadline:
                self._check_stop()
                if epoch != self._motion_epoch:
                    # STOP canceled the old camera request. Start a new
                    # observation after the arms have remained at their pose.
                    request = uuid.uuid4().hex[:12]
                    self._waiting_request = request
                    self._last_result = None
                    self._result_event.clear()
                    self.side_pub.publish(String(data=json.dumps(dict(
                        command='START', product=self.target['product'],
                        view=view, cycle_id=self.cycle_id, request_id=request))))
                    deadline = self._process_time() + self.side_timeout
                    epoch = self._motion_epoch
                if self._result_event.wait(.05):
                    result = str(self._last_result.get('judge', '')).upper()
                    if result not in ('OK', 'NG'):
                        raise RuntimeError('side inspection error: %s' % self._last_result)
                    return result
            raise RuntimeError('side inspection timeout: %s' % view)
        finally:
            self._waiting_request = None

    def _views_carpart(self):
        """Original carpart BOTTOM, BACK, FRONT, LEFT, RIGHT inspection."""
        views = (
            ('BOTTOM', [0, 5, -10, -88, 0], [180, 0, -10, 0, 84, 0]),
            ('BACK', [0, 30, -120, 60, 0], [180, 0, -100, 0, -39, 0]),
            ('FRONT', [0, -30, -21, -103, 0], [180, 0, -15, 0, 110, 0]),
            ('LEFT', [-40, 35, -100, -38, 0], [140, 58, 0, -5, 100, -60]),
            ('RIGHT', [40, 35, -100, -38, 0], [220, -58, 0, 15, 100, 60]),
        )
        for name, a_pose, b_pose in views:
            self._set_phase('CARPART_' + name + '_VIEW')
            if name == 'BACK':
                self._move_arm([0, 45, -45, -80, 0])
            if name == 'FRONT':
                self._b_move([270, 0, -30, 0, -10, 0])
            if name == 'RIGHT':
                self._b_move([180, 0, 0, 0, 0, 0])
            self._dual_pose(a_pose, b_pose)
            if self._inspect(name) == 'NG':
                return False
        return True

    def _views_baseplate(self):
        """Baseplate-only BOTTOM once, BACK eight, FLASH eight."""
        bottom_a, bottom_b = self._BASEPLATE_POSES['BOTTOM']
        self._set_phase('BOTTOM_VIEW')
        self._dual_pose(bottom_a, bottom_b)
        if self._inspect('BOTTOM_1') == 'NG':
            return False
        for name in ('BACK', 'FLASH'):
            a_pose, b_pose = self._BASEPLATE_POSES[name]
            if name == 'BACK':
                # From BOTTOM, B rotates only q1 to 75 deg while A moves
                # directly to its BACK inspection pose.
                self._set_phase('BOTTOM_TO_BACK_INTERMEDIATE')
                self._dual_pose(a_pose, [75] + list(bottom_b[1:]))
            else:
                # From BACK, B rotates only q1 to 75 deg while A moves
                # directly to its FLASH inspection pose.
                self._set_phase('BACK_TO_FLASH_INTERMEDIATE')
                self._dual_pose(a_pose, [75] + list(self._BASEPLATE_POSES['BACK'][1][1:]))
            self._set_phase(name + '_VIEW')
            self._b_move(b_pose)
            for index, q5 in enumerate(self._BASEPLATE_YAWS, 1):
                self._move_arm(a_pose[:4] + [q5])
                if self._inspect('%s_%d' % (name, index)) == 'NG':
                    return False
        self._b_move([90, 0, 0, 0, 0, 0])
        return True

    def _sort(self, good):
        self._set_phase('SORT_GOOD' if good else 'SORT_NG')
        if self._last_b_pose == (220, -58, 0, 15, 100, 60):
            self._b_move([180, 0, 0, 0, 0, 0])
        self._dual_pose(self.rail_travel_pose_deg, [90, 0, 0, 0, 0, 0])
        self._move_rail(self.good_rail if good else self.reject_rail)
        self._move_arm(self.sort_pose if good else self.reject_sort_pose)
        self._suction(False)
        self._move_arm(self.rail_travel_pose_deg)
        self._move_rail(0)
        self._move_arm(self.home_pose_deg)

    def _change_tool(self, target):
        self._move_arm(self.rail_travel_pose_deg)
        self._move_rail(0)
        self._set_phase('WAIT_TOOL_' + target.upper())
        if self.use_robot:
            self.get_logger().warning('Manually install %s tool, then press TOOL CHANGE COMPLETE' % target)
            self._resume_event.clear()
            self._tool_change_waiting = True
            try:
                end = self._process_time() + self.tool_timeout
                while self._process_time() < end:
                    self._check_stop()
                    if self._resume_event.wait(.1):
                        break
                else:
                    raise RuntimeError('manual tool change timed out')
            finally:
                self._tool_change_waiting = False
        if target == 'centric':
            self.gripper = CentricGripper(
                port=str(self.get_parameter('dxl_port').value),
                baud=int(self.get_parameter('dxl_baud').value),
                dxl_id=int(self.get_parameter('centric_id').value),
                logger=self.get_logger(),
                planet_max_angle_deg=float(
                    self.get_parameter('centric_planet_max_angle_deg').value),
                # Opening and the final automatic closing use the same
                # Dynamixel profile velocity in this cell.
                normal_profile_velocity=44,
                auto_profile_velocity=44,
                auto_load_threshold_percent=float(
                    self.get_parameter('centric_auto_load_threshold_percent').value),
                auto_load_count_limit=int(
                    self.get_parameter('centric_auto_load_count_limit').value),
                auto_grasp_timeout_sec=float(
                    self.get_parameter('centric_auto_grasp_timeout_sec').value),
                emergency_load_percent=float(
                    self.get_parameter('centric_emergency_load_percent').value),
                communication_retry_count=int(
                    self.get_parameter('centric_communication_retry_count').value),
            ) if self.use_robot else None
            if self.gripper is not None:
                self.gripper.connect_shared(self.bridge.ph, self.bridge.pk,
                    self.bridge.bus_lock, capture_open_on_start=True)
            self.use_gripper = self.use_robot
        else:
            if self.gripper is not None:
                self.gripper.close(torque_off=False)
                self.gripper = None
            self.use_gripper = False
        self.tool = target

    def _wait_holes(self):
        started = self._process_time()
        deadline = started + self.holes_timeout
        next_log = started + 15.0
        epoch = self._motion_epoch
        while self._process_time() < deadline:
            self._check_stop()
            if epoch != self._motion_epoch:
                self.latest_holes = None
                self._holes_needed_after = time.monotonic()
                self.hole_pub.publish(String(data='START'))
                epoch = self._motion_epoch
            if self.latest_holes is not None and self.latest_holes_stamp and \
                    self.latest_holes_stamp >= self._holes_needed_after:
                self.get_logger().info('Fresh baseplate hole coordinates ready after %.1f s'
                                       % (self._process_time() - started))
                return
            if self._process_time() >= next_log:
                self.get_logger().info('Waiting for stable baseplate mid/side hole coordinates '
                                       '(%.0f/%.0f s)'
                                       % (self._process_time() - started, self.holes_timeout))
                next_log = self._process_time() + 15.0
            time.sleep(.1)
        raise RuntimeError('fresh baseplate hole coordinates not received within %.0f s'
                           % self.holes_timeout)

    def _current_a3_point(self):
        """A3 XY under the TCP after turning joint 1 to zero at rail home."""
        arm_xyz = fk_pos(self._current_q_rad()[:4])
        x_mm = A3_LT_ROBOT_Y_MM + arm_xyz[1] * 1000 - self.vision_rail_ref * 1000
        y_mm = A3_LT_ROBOT_X_MM + arm_xyz[0] * 1000
        point = [float(x_mm), float(y_mm)]
        if not self._in_a3(point):
            raise RuntimeError('q1=0 placement is outside calibrated A3: %s' % point)
        return point

    def _lid(self, point):
        self._set_phase('LID_PICK')
        self._move_arm(self.rail_travel_pose_deg)
        self._move_rail(self.lid_rail)
        self._move_arm(self.lid_pick_pose)
        self._suction(True)
        self._move_arm(self.rail_travel_pose_deg)
        self._move_rail(0)
        self._set_phase('LID_VERTICAL_PROBE')
        self._lid_probe_at(point)

    def _lid_probe_at(self, point):
        """Lower from 80 to 50 mm, then require 30% load before lifting.

        Suction stays ON: the assembled part is released only at SORT_GOOD.
        Reaching the end without the load threshold is a failed lid insertion.
        """
        low_z = self.lid_probe_start_z - self.lid_probe_extra
        approach, descent, _, _ = self._build_vertical_paths(
            point[0], point[1], target_z_mm=low_z)
        start_z_arm = self.world_to_arm(self.a3_to_world(
            point[0], point[1], self.lid_probe_start_z))[2]
        split = next((i for i, q in enumerate(descent)
                      if fk_pos(q[:4])[2] <= start_z_arm + 1e-4), None)
        if split is None or split >= len(descent) - 1:
            raise RuntimeError('lid probe path does not reach its start height')

        self._execute_joint_trajectory(approach)
        self._execute_joint_trajectory(descent[:split + 1])
        probe = descent[split + 1:]
        started = time.monotonic()
        last_command = 0.0
        last_poll = 0.0
        contact = False
        valid_samples = 0
        for index, q in enumerate(probe):
            target_time = started + index / self.planner_hz
            while time.monotonic() < target_time:
                self._check_stop()
                time.sleep(min(0.01, target_time - time.monotonic()))
            self._check_stop()
            now = time.monotonic()
            if self.bridge is not None and now - last_poll >= self.lid_probe_poll:
                loads = self.bridge.read_status()
                samples = [abs(float(loads[i][0])) for i in (5, 6)
                           if i in loads and loads[i][0] is not None]
                if samples:
                    valid_samples += 1
                    if max(samples) >= self.lid_probe_load:
                        actual = self._current_q_rad()
                        self.bridge.command_joints(actual.tolist())
                        self._set_sim_state(q_deg=np.degrees(actual).tolist())
                        self.get_logger().info('Lid contact: %.1f%% load' % max(samples))
                        contact = True
                        break
                last_poll = now
            if self.bridge is not None and (now - last_command >= 1.0 / self.robot_cmd_hz
                                             or index == len(probe) - 1):
                self.bridge.command_joints(q.tolist())
                last_command = now
            self._set_sim_state(q_deg=np.degrees(q).tolist())

        if self.bridge is not None:
            if not valid_samples:
                raise RuntimeError('lid probe has no valid load readings from ID5/6')
            if not contact:
                raise RuntimeError('lid probe reached %.1f mm without %.1f%% load; stopping'
                                   % (low_z, self.lid_probe_load))
        if self.lid_probe_hold:
            deadline = time.monotonic() + self.lid_probe_hold
            while time.monotonic() < deadline:
                self._check_stop()
                time.sleep(min(0.05, deadline - time.monotonic()))

        current = self._current_q_rad()
        upper = self.world_to_arm(self.a3_to_world(
            point[0], point[1], self.lid_approach_z))
        retreat, error = self._vertical_line(
            current[:4], fk_pos(current[:4]), upper, 'lid vertical retreat')
        if retreat is None:
            raise RuntimeError(error)
        self._execute_joint_trajectory(np.column_stack((
            retreat[:, :4], np.full(len(retreat), current[4]))))

    def _run_cell(self):
        target = dict(self.target)
        point = target['point']
        success = False
        try:
            if target['product'] == 'baseplate':
                # The TOP model judges defects only. The suction pick uses
                # the fresh hole detector's mid XY.
                # Wait in the worker; ROS callbacks remain free to deliver holes.
                self._set_phase('BASEPLATE_WAIT_PICK_MID')
                self.latest_holes = None
                self._holes_needed_after = time.monotonic()
                self.hole_pub.publish(String(data='START'))
                self._wait_holes()
                point = np.asarray(self.latest_holes[:2], dtype=float).copy()
            self._set_phase('TOP_PICK_' + target['product'].upper())
            self._vertical_at(point, self.top_z if target['product'] == 'carpart'
                               else self.base_z, lambda: self._suction(True))
            if target['judge'] == 'NG':
                self._sort(False)
                success = True
                return
            views = (self._views_carpart if target['product'] == 'carpart'
                     else self._views_baseplate)
            if not views():
                self._sort(False)
                success = True
                return
            if target['product'] == 'carpart':
                self._sort(True)
                success = True
                return
            self._set_phase('BASEPLATE_Q1_TO_ZERO')
            q = np.degrees(self._current_q_rad()).tolist()
            q[0] = 0.0
            self._move_arm(q)
            assembly_point = self.base_assembly_point.copy()
            self.get_logger().info(
                'Baseplate assembly placement A3=(%.1f, %.1f) mm'
                % tuple(assembly_point))
            self._vertical_at(assembly_point, self.base_z,
                              lambda: self._suction(False))
            self._holes_needed_after = time.monotonic()
            self.hole_pub.publish(String(data='START'))
            self._change_tool('centric')
            self._wait_holes()
            self._set_phase('ASSEMBLY_SHAFT_BEARING_GEAR')
            # Reuse each calibrated cycle directly so any failure stops this cell.
            holes = np.asarray(self.latest_holes, dtype=float)
            mid, side = holes[:2], holes[2:4]
            cycles = (
                ('SHAFT_1', self.shaft1_rail_mm, mid, 'MID', self.grasp_pose_deg,
                 self.centric_preclose_angle_deg, self.shaft_place_z_mm),
                ('SHAFT_2', self.shaft2_rail_mm, side, 'SIDE', self.grasp_pose_deg,
                 self.centric_preclose_angle_deg, self.shaft_place_z_mm),
                ('BEARING_1', self.bearing1_rail_mm, mid, 'MID', self.grasp_pose_deg,
                 self.bearing_preclose_angle_deg, self.bearing_place_z_mm),
                ('BEARING_2', self.bearing2_rail_mm, side, 'SIDE', self.grasp_pose_deg,
                 self.bearing_preclose_angle_deg, self.bearing_place_z_mm),
                ('GEAR_1', self.gear1_rail_mm, mid, 'MID', self.gear_grasp_pose_deg,
                 self.gear_preclose_angle_deg, self.gear_place_z_mm),
                ('GEAR_2', self.gear2_rail_mm, side, 'SIDE', self.gear_grasp_pose_deg,
                 self.gear_preclose_angle_deg, self.gear_place_z_mm),
            )
            for item, rail_mm, destination, tag, pose, angle, z in cycles:
                self._pick_and_place_cycle(item, rail_mm, destination, tag,
                                           pose, angle, z)
            self._change_tool('suction')
            self._lid(assembly_point)
            self._sort(True)
            success = True
        finally:
            self.target = None
            self._holes_needed_after = None
            if success:
                self.ready_pub.publish(String(data='READY'))

    # Shared baseplate poses; test commands are enabled by baseplate.launch.py.
    _BASEPLATE_POSES = {
        'BOTTOM': ([0, 10, 5, -100, 0], [0, 0, -37, 0, -50, 180]),
        'BACK': ([0, 0, -85, 90, 0], [0, 0, -25, 0, -60, 180]),
        'FLASH': ([0, -10, -50, -95, 0], [0, 0, -50, 0, -93, 180]),
    }
    _BASEPLATE_YAWS = (0, 45, 90, 135, 180, -45, -90, -135)
    # Same measured stop-to-straight travel as assembly_debug._mechhome.
    # Coupled motors use equal/opposite travel to avoid fighting each other.
    _MECHANICAL_HOME_OFFSETS = {
        2: 3473.5 - (-4279.0),
        3: 1562.0 - 8414.5,
        4: -(1562.0 - 8414.5),
        5: 995.8 - (-3140.0),
        6: -(995.8 - (-3140.0)),
    }

    def _measured_mechhome(self):
        """Current stop ticks + measured travel = the new straight-pose zero."""
        offsets = self._MECHANICAL_HOME_OFFSETS
        reference = np.array([
            0.0,
            offsets[2] * 360.0 / (4096 * JOINT_MOTORS[1]['gear']),
            offsets[3] * 360.0 / (4096 * JOINT_MOTORS[2]['gear']),
            offsets[5] * 360.0 / (4096 * JOINT_MOTORS[3]['gear']),
            0.0,
        ])
        self._check_stop()
        if self.bridge is None:
            self.reference_ready = True
            self._set_sim_state(q_deg=reference.tolist())
            self._move_arm([0.0] * 5)
            return
        # The operator supplies the physical stops, with J1/J5 at their marks.
        self.reference_ready = False
        present = {motor_id: self.bridge._present(motor_id)
                   for joint in JOINT_MOTORS[:5] for motor_id in joint['ids']}
        zero = {motor_id: tick + offsets.get(motor_id, 0.0)
                for motor_id, tick in present.items()}
        self.get_logger().warning('MECH HOME current stop ticks=%s; measured travel=%s; '
                                  'straight goal ticks=%s' %
                                  (present, offsets, {i: round(v) for i, v in zero.items()}))
        self.bridge.zero_tick.update(zero)
        actual = np.asarray(self.bridge.read_joints()[:5], dtype=float)
        measured = np.degrees(actual)
        if len(measured) != 5 or not np.isfinite(measured).all() or \
                max(abs(measured - reference)) > 0.25:
            raise RuntimeError('measured mechanical zero verification failed: %s' % measured)
        self.reference_ready = True
        self._set_sim_state(q_deg=reference.tolist())
        speed_deg_s = 8.0
        profile = {}
        for joint in JOINT_MOTORS[:5]:
            pv = max(1, int(round(speed_deg_s * joint['gear'] / (0.229 * 6.0))))
            for motor_id in joint['ids']:
                profile[motor_id] = min(PROFILE_VELOCITY[motor_id], pv)
        try:
            for motor_id, pv in profile.items():
                self.bridge._w4(motor_id, 112, pv)
            self._check_stop()
            self.bridge.command_joints([0.0] * 5)
            epoch = self._motion_epoch
            deadline = self._process_time() + max(
                self.joint_timeout, max(abs(reference)) / speed_deg_s + 30.0)
            while self._process_time() < deadline:
                self._check_stop()
                if epoch != self._motion_epoch:
                    self.bridge.command_joints([0.0] * 5)
                    epoch = self._motion_epoch
                actual = np.asarray(self.bridge.read_joints()[:5], dtype=float)
                if len(actual) != 5 or not np.isfinite(actual).all():
                    raise RuntimeError('invalid A-arm reading during measured mechhome')
                self._set_sim_state(q_deg=np.degrees(actual).tolist())
                if max(abs(np.degrees(np.arctan2(np.sin(actual), np.cos(actual))))) <= 1.5:
                    paired_ok = True
                    for joint in JOINT_MOTORS[2:4]:
                        motor_id = joint['ids'][1]
                        tolerance = 1.5 * joint['gear'] * 4096 / 360.0
                        if abs(self.bridge._present(motor_id) - round(zero[motor_id])) > tolerance:
                            paired_ok = False
                    if paired_ok:
                        self.get_logger().info('Measured MECH HOME complete; ID4/ID6 verified')
                        return
                time.sleep(.15)
            raise RuntimeError('Measured MECH HOME arrival timeout')
        finally:
            for motor_id in profile:
                try:
                    self.bridge._w4(motor_id, 112, PROFILE_VELOCITY[motor_id])
                except Exception as exc:
                    self.get_logger().error('restore profile ID%d failed: %s' % (motor_id, exc))

    def _test_can_move(self, require_reference=True):
        if self.busy or self._pause_event.is_set() or self._tool_change_waiting:
            self.get_logger().warning('Test command unavailable while busy or paused')
            return False
        if self.phase == 'ERROR_HOLDING':
            self.get_logger().error('Hardware fault must be resolved before moving')
            return False
        if self.use_robot and ((require_reference and not self.reference_ready) or not self.b_ready or
                               self.bridge is None or self.b_arm is None or
                               self.rail is None):
            self.get_logger().error('A/B arms or suction controller unavailable')
            return False
        return True

    def _test_help(self):
        self.get_logger().info(
            'BASEPLATE TEST: baseplate = pick + 17 views + hold; '
            'inspect = 17 views without pick; bottom|back|flash [q5_deg] '
            '= pose only; inspect bottom|back|flash [q5_deg] = one observation; '
            'aj q1 q2 q3 q4 q5; bj q1 q2 q3 q4 q5 q6; '
            'views; status; suction on|off; mechhome; stop; resume; abort')

    def _handle_test_command(self, parts):
        if not parts:
            return True
        name = parts[0]
        if name in ('help', 'h', '?'):
            self._test_help()
            return True
        if name == 'views':
            for view, (a, b) in self._BASEPLATE_POSES.items():
                self.get_logger().info('%s: A=%s deg, B=%s deg' % (view, a, b))
            self.get_logger().info('BACK/FLASH scan q5=%s deg' % (self._BASEPLATE_YAWS,))
            return True
        if name == 'status':
            self._log_status()
            self.get_logger().info('Test holding=%s; last result=%s; observations=%s'
                                   % (self._test_holding, self._test_result,
                                      self._test_records))
            return True
        if name == 'baseplate' or (name == 'inspect' and len(parts) == 1):
            if len(parts) != 1:
                self.get_logger().error('usage: baseplate')
            else:
                self._start_baseplate_test(pick=(name == 'baseplate'))
            return True
        if name == 'mechhome':
            if len(parts) != 1:
                self.get_logger().error('usage: mechhome')
            elif self._test_holding or (self.rail is not None and self.rail.suction_on):
                self.get_logger().error('mechhome blocked while suction/part hold is active')
            elif self._test_can_move(require_reference=False):
                rail_mm = self.rail.position_mm if self.rail is not None else self.sim_rail_mm
                if abs(rail_mm) > self.rail_tol_mm:
                    self.get_logger().error('mechhome requires rail at 0 mm')
                    return True
                self._test_last_view = None
                self._start_auxiliary('MECH_REFERENCE_TO_STRAIGHT_HOME',
                                      self._measured_mechhome)
            return True
        if name in ('inspect', 'bottom', 'back', 'flash'):
            pose_parts = parts[1:] if name == 'inspect' else parts
            if not 1 <= len(pose_parts) <= 2:
                self.get_logger().error('usage: bottom|back|flash [q5_deg] or inspect bottom|back|flash [q5_deg]')
                return True
            view = pose_parts[0].upper()
            try:
                q5 = float(pose_parts[1]) if len(pose_parts) == 2 else 0.0
                if view not in self._BASEPLATE_POSES or not math.isfinite(q5):
                    raise ValueError()
                if not -180 <= q5 <= 180:
                    raise ValueError()
            except ValueError:
                self.get_logger().error('pose needs bottom/back/flash and q5 in [-180, 180]')
                return True
            if self._test_can_move():
                self.target = dict(product='baseplate', judge='OK', point=None)
                self.cycle_id = uuid.uuid4().hex[:12]
                self._start_auxiliary(
                    view + '_VIEW',
                    lambda: self._run_single_baseplate_view(view, q5, name == 'inspect'))
            return True
        if name in ('on', 'off'):
            if len(parts) != 1:
                self.get_logger().error('usage: on | off')
            elif self._test_can_move():
                self._start_auxiliary(
                    'SUCTION_' + name.upper(),
                    lambda: self._test_suction(name == 'on'))
            return True
        if name in ('aj', 'bj', 'bhome', 'b_home', 'stop', 's', 'estop', '0',
                    'resume', 'abort', 'abort_cycle'):
            if name in ('aj', 'bj', 'bhome', 'b_home'):
                if not self._test_can_move():
                    return True
                self._test_last_view = None
            return False
        self.get_logger().warning('Command unavailable in baseplate test mode: %s' % name)
        self._test_help()
        return True

    def _test_suction(self, on):
        # Conservatively mark a possible held part before an ON operation.
        if on:
            self._test_holding = True
        self._suction(on)
        self._test_holding = bool(on)

    def _start_baseplate_test(self, pick):
        if not self._test_can_move():
            return
        if pick and not self.top_camera_enabled:
            self.get_logger().warning('baseplate pick needs TOP camera coordinates. '
                                      'With TOP disabled, manually hold the part and use inspect; '
                                      'aj/bj/bottom/back/flash remain available.')
            return
        if self.tool != 'suction':
            self.get_logger().error('Baseplate test requires the suction tool')
            return
        if pick and self._test_holding:
            self.get_logger().warning('A part may already be held. Use inspect to test it; '
                                      'place it down before suction off and another pick.')
            return
        if not pick:
            self._test_holding = True  # The operator supplies an already held part.
        self.target = dict(self._test_top or
                           dict(product='baseplate', judge='OK', point=None))
        # A cached TOP verdict belongs to one pick only.
        self._test_top = None
        self.cycle_id = uuid.uuid4().hex[:12]
        self._test_records = []
        self._test_result = None
        self._holes_needed_after = None
        self._start_auxiliary('BASEPLATE_TEST', lambda: self._run_baseplate_test(pick))

    def _move_baseplate_view(self, view, q5=0.0):
        a, b = self._BASEPLATE_POSES[view]
        a = list(a[:4]) + [float(q5)]
        previous_view = self._test_last_view
        self._test_last_view = None
        # Preserve the existing B q1=75 deg transition, using the actual last
        # commanded B pose so reverse/manual view selection does not assume BOTTOM.
        if previous_view != view and self._last_b_pose is not None:
            intermediate = list(self._last_b_pose)
            intermediate[0] = 75.0
            self._set_phase('BASEPLATE_TO_' + view + '_INTERMEDIATE')
            self._dual_pose(a, intermediate)
            self._set_phase(view + '_VIEW')
            self._b_move(b)
        else:
            self._set_phase(view + '_VIEW')
            self._dual_pose(a, b)
        self._test_last_view = view

    def _record_baseplate_inspection(self, label, q5):
        judge = self._inspect(label)
        self._test_records.append(dict(view=label, q5_deg=float(q5), judge=judge))
        self.get_logger().info('BASEPLATE %s q5=%.1f deg: %s' % (label, q5, judge))
        return judge

    def _publish_baseplate_test_result(self):
        if any(r['judge'] == 'NG' for r in self._test_records):
            self._test_result = 'NG'
        elif self._test_records and all(r['judge'] == 'OK' for r in self._test_records):
            self._test_result = 'OK'
        else:
            self._test_result = 'NOT_INSPECTED'
        mode = ('SIMULATED' if self.simulate_inspection and not self.use_robot
                else 'CAMERA' if self.arm_camera_enabled else 'MOTION_ONLY')
        self.test_result_pub.publish(String(data=json.dumps(dict(
            product='baseplate', cycle_id=self.cycle_id, judge=self._test_result,
            inspection_mode=mode,
            observations=self._test_records))))

    def _finish_baseplate_test(self):
        self._set_phase('VIEW_MOTION_DONE_HOLDING' if self._test_result == 'NOT_INSPECTED'
                        else 'INSPECTION_DONE_HOLDING')

    def _run_single_baseplate_view(self, view, q5, observe):
        try:
            self._move_baseplate_view(view, q5)
            if observe:
                self._test_holding = True
                self._test_records = []
                self._test_result = None
                self._record_baseplate_inspection(view + '_1', q5)
                self._publish_baseplate_test_result()
                self._finish_baseplate_test()
            else:
                self._set_phase('BASEPLATE_' + view + '_POSE_READY')
        finally:
            if observe:
                self.side_pub.publish(String(data=json.dumps(dict(command='STOP'))))
            self.target = None

    def _run_baseplate_test(self, pick):
        try:
            if pick:
                self._set_phase('BASEPLATE_WAIT_PICK_MID')
                self.latest_holes = None
                self._holes_needed_after = time.monotonic()
                self.hole_pub.publish(String(data='START'))
                self._wait_holes()
                point = np.asarray(self.latest_holes[:2], dtype=float).copy()
                self.hole_pub.publish(String(data='STOP'))
                self._holes_needed_after = None
                self._set_phase('TOP_PICK_BASEPLATE')
                self._vertical_at(point, self.base_z, lambda: self._test_suction(True))
                if self.target['judge'] == 'NG':
                    self.get_logger().warning('TOP verdict NG; test continues through all views')
            self._move_baseplate_view('BOTTOM')
            self._record_baseplate_inspection('BOTTOM_1', 0)
            for view in ('BACK', 'FLASH'):
                self._move_baseplate_view(view)
                for index, q5 in enumerate(self._BASEPLATE_YAWS, 1):
                    self._set_phase(view + '_VIEW')
                    a, _ = self._BASEPLATE_POSES[view]
                    self._move_arm(list(a[:4]) + [q5])
                    self._record_baseplate_inspection('%s_%d' % (view, index), q5)
            self._publish_baseplate_test_result()
            self._finish_baseplate_test()
            self.get_logger().info(
                '17-view test complete: %s. Both arms hold their last poses; '
                'suction stays on. Use aj/bj/bottom/back/flash/inspect for further tests.'
                % self._test_result)
        finally:
            self.hole_pub.publish(String(data='STOP'))
            self.side_pub.publish(String(data=json.dumps(dict(command='STOP'))))
            self._waiting_request = None
            self._holes_needed_after = None
            self.target = None

    def _start_auxiliary(self, phase, callback):
        if not getattr(self, 'baseplate_test_mode', False):
            return super()._start_auxiliary(phase, callback)
        with self._state_lock:
            if self.busy:
                self.get_logger().warning('Test is busy')
                return
            self.busy = True
            self._stop_event.clear()
            self._paused_seconds = 0.0
            self._pause_started = None
        def worker():
            try:
                self._set_phase(phase)
                callback()
                # The inherited worker overwrites successful phases with READY.
                # Keep the inspection result visible while allowing manual commands.
                if self.phase not in ('INSPECTION_DONE_HOLDING', 'VIEW_MOTION_DONE_HOLDING') and not (
                        self.phase.startswith('BASEPLATE_') and
                        self.phase.endswith('_POSE_READY')):
                    self._set_phase('READY')
            except SequenceStopped:
                self._set_phase('STOPPED_READY')
            except Exception as exc:
                self._fail(str(exc))
            finally:
                with self._state_lock:
                    self.busy = False
                self._publish_state()
                self._send_viewer_state(force=True)
        threading.Thread(target=worker, daemon=True).start()

    def _emergency_stop(self, reason):
        super()._emergency_stop(reason)
        # The inherited stop holds A/rail/gripper; test faults also hold B.
        if getattr(self, 'baseplate_test_mode', False):
            arm = getattr(self, 'b_arm', None)
            if arm is not None:
                try:
                    present = arm.read_present_ticks(retry=1, use_fallback=False)
                    arm.write_goal_ticks(present)
                except Exception as exc:
                    self.get_logger().error('B-arm emergency hold failed: %s' % exc)


    def _close_hardware(self):
        if self.rail is not None and self.rail.connected:
            try:
                self.rail.suction(False, wait=False)
            except Exception:
                pass
        super()._close_hardware()

    def destroy_node(self):
        self._resume_event.set()
        if self.b_arm is not None:
            self.b_arm.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = Cell()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
