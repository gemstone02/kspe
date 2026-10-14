"""One A arm, rail, removable centric tool and suction for assembly trials.

Use assembly.launch.py. This node deliberately ignores product detection and
never starts the production inspection/sorting sequence or connects B.
"""

import csv
import datetime
import math
import os
import threading
import time

import numpy as np
import rclpy
from std_msgs.msg import String

from .cell import Cell
from .centric_sequence_5axis import JOINT_MOTORS, PROFILE_VELOCITY
from .centric_gripper_5axis import CentricGripper
from .cell_rail_serial import RAIL_MAX_M
from .kinematics_ee_5axis import fk_pos


ITEMS = ('shaft1', 'shaft2', 'bearing1', 'bearing2', 'gear1', 'gear2')
ASSEMBLY_RAIL_TRAVEL_POSE_DEG = (0.0, 30.0, -40.0, -90.0, 0.0)
ASSEMBLY_INSERT_Z_MM = {'shaft1': 0.0, 'shaft2': 0.0,
                        'bearing1': 0.0, 'bearing2': 0.0,
                        'gear1': -3.0, 'gear2': -3.0}
INSERT_RELEASE_DWELL_SEC = 0.3
# Manual insertion trials may request a small negative Z, but all of the
# inherited IK and floor-collision checks still have to approve the path.
MIN_MANUAL_INSERT_Z_MM = -5.0
DEFAULT_MOTOR_TRACE_DIR = '/media/wonseok/새 볼륨/project/a_motors_log'
DEFAULT_MOTOR_TRACE_MOUNT = '/media/wonseok/새 볼륨'
# Measured change in motor ticks from the physical limit to the desired
# straight pose. At mechhome we read the *current* ticks at the limit and add
# these offsets; the absolute encoder readings are not stored as home values.
# ID3/ID4 and ID5/ID6 drive shared links in opposite directions. A measured
# 51-tick discrepancy between ID3/ID4 would make their targets disagree on
# the link angle, so derive each second motor's travel from the first motor.
# ID6's independently recorded straight reading was unreliable as well.
MECHANICAL_HOME_OFFSETS = {
    2: 3473.5 - (-4279.0),
    3: 1562.0 - 8414.5,
    5: 995.8 - (-3140.0),
}
# Relative to the original measured stop-to-straight offset above, the ID2
# home goal must travel 4 degrees farther in the negative q2 direction.
# This adds about 682.7 motor ticks total (341.3 beyond the previous -2 deg
# correction). After mechhome the corrected straight pose is reported as q2=0.
MECH_Q2_STRAIGHT_CORRECTION_DEG = -4.0
MECHANICAL_HOME_OFFSETS[2] += (
    JOINT_MOTORS[1]['dir'] * JOINT_MOTORS[1]['pair_sign'][0]
    * MECH_Q2_STRAIGHT_CORRECTION_DEG
    * JOINT_MOTORS[1]['gear'] * 4096.0 / 360.0
)
MEASURED_ID4_OFFSET = 1833.5 - (-4968.0)  # 51 ticks short of opposite ID3
MECHANICAL_HOME_OFFSETS[4] = -MECHANICAL_HOME_OFFSETS[3]
MECHANICAL_HOME_OFFSETS[6] = -MECHANICAL_HOME_OFFSETS[5]
# These joint angles describe the measured limit posture for the simulator,
# progress reporting and the homing timeout. ID3 supplies the q3 angle.
MECHANICAL_REFERENCE_DEG = (
    0.0,
    MECHANICAL_HOME_OFFSETS[2] / (4096 * JOINT_MOTORS[1]['gear']) * 360.0,
    MECHANICAL_HOME_OFFSETS[3] / (4096 * JOINT_MOTORS[2]['gear']) * 360.0,
    MECHANICAL_HOME_OFFSETS[5] / (4096 * JOINT_MOTORS[3]['gear']) * 360.0,
    0.0,
)
SHAFT_PICK_DEG = (0.0, -18.0, -129.0, -36.0, 50.0)
BEARING_PICK_DEG = (0.0, -53.0, -59.0, -69.0, 0.0)


class InsertionPlanRejected(RuntimeError):
    """A proposed insertion path was rejected before its A-arm motion began."""


class AssemblyDebug(Cell):
    def _setup_hardware(self, g):
        # Cell normally records the startup pose as straight. For mechanical
        # homing, prohibit motion until mechhome replaces this temporary zero.
        self.declare_parameter('assembly_reference_mode', 'current')
        self.assembly_reference_mode = str(g('assembly_reference_mode').value).lower()
        if self.assembly_reference_mode not in ('current', 'mechanical'):
            raise ValueError('assembly_reference_mode must be current or mechanical')
        super()._setup_hardware(g)
        if self.use_robot and self.assembly_reference_mode == 'mechanical':
            self.reference_ready = False
            self._set_sim_state(q_deg=list(MECHANICAL_REFERENCE_DEG))
            self.get_logger().warning(
                'Mechanical reference selected: align J1/J5 to 0 marks and '
                'J2/J3/J4 to mechanical limits, then type mechhome. '
                'A-arm movement is blocked until calibration completes.')

    def __init__(self):
        super().__init__()
        if not self.assembly_debug_mode or self.use_b_robot:
            raise RuntimeError('assembly_debug requires assembly_debug_mode:=true and B disabled')
        # Assembly-only rail pose. The full cell and its own travel pose are unchanged.
        self.rail_travel_pose_deg = list(ASSEMBLY_RAIL_TRAVEL_POSE_DEG)
        self.declare_parameter('insert_approach_clearance_mm', 25.0)
        self.declare_parameter('insert_q5_rotation_deg', 15.0)
        self.insert_approach_clearance_mm = float(
            self.get_parameter('insert_approach_clearance_mm').value)
        self.insert_q5_rotation_deg = float(
            self.get_parameter('insert_q5_rotation_deg').value)
        if not math.isfinite(self.insert_approach_clearance_mm) or \
                self.insert_approach_clearance_mm <= 0:
            raise ValueError('insert_approach_clearance_mm must be positive and finite')
        if not math.isfinite(self.insert_q5_rotation_deg):
            raise ValueError('insert_q5_rotation_deg must be finite')
        if self.assembly_reference_mode == 'current':
            self.get_logger().info('A-arm startup pose recorded as straight [0,0,0,0,0]')
        self.pick_overrides = {}
        self.place_z_overrides = {}
        self.hole_overrides = {}
        self.held_item = None
        self.suction_enabled = False
        self._debug_fault = False
        self._last_debug_fault = None
        self.manual_grasp_ready = False
        self.vision_scan_started = None
        self.insert_retreat_path = None
        self._cycle_index = None             # Next of the six parts to complete.
        self._cycle_holes = None             # Vision coordinates frozen at start.
        self._cycle_released_item = None     # Release succeeded, retreat may remain.
        self._cycle_running = False
        self._cycle_uncertain = False        # Interrupted during grasp/release.
        self._cycle_needs_clearance = False # Aborted inside vertical motion.
        self._pending_manual_command = None
        self._manual_after_stop_timer = self.create_timer(
            0.05, self._dispatch_manual_after_stop)
        self.product_pub = self.create_publisher(String, '/cell/product', 10)
        self.declare_parameter('motor_trace', True)
        self.declare_parameter('motor_trace_period_sec', 0.5)
        self.declare_parameter('motor_trace_ids', '1,2,3,4,5,6,7')
        self.declare_parameter('motor_trace_dir', DEFAULT_MOTOR_TRACE_DIR)
        self._trace_stop = threading.Event()
        self._trace_thread = None
        self._trace_file = None
        self.motor_trace_path = None
        if self.bridge is not None and self.get_parameter('motor_trace').value:
            self._start_motor_trace()
        self.get_logger().info('ASSEMBLY DEBUG: B disconnected. Type help in cell_cli.')

    @staticmethod
    def _trace_signed(raw, bits):
        return raw - (1 << bits) if raw & (1 << (bits - 1)) else raw

    def _trace_one(self, motor_id):
        """One read of XL430's contiguous 124..146 block; never retries.

        Do not use read_detailed_status here: it retries each register and can
        monopolize the shared A-arm/ID8 bus during a motion.
        """
        bridge = self.bridge
        if bridge is None or not bridge.bus_lock.acquire(blocking=False):
            return None  # A motion owns the bus; skip this sample.
        try:
            start = time.monotonic()
            try:
                data, comm, alert = bridge.pk.readTxRx(bridge.ph, motor_id, 124, 23)
            except Exception as exc:
                data, comm, alert = [], '%s: %s' % (type(exc).__name__, exc), 0
            elapsed_ms = (time.monotonic() - start) * 1000.0
            row = dict(voltage_v='', load_percent='', pwm_percent='',
                       velocity_raw='', position_tick='', temperature_c='',
                       torque_enable='', hardware_error='',
                       comm_result=comm, packet_error=alert,
                       latency_ms=round(elapsed_ms, 2))
            if comm == 0 and len(data) != 23:
                row['comm_result'] = 'short packet: %d/23 bytes' % len(data)
            if comm == 0 and len(data) == 23:
                u16 = lambda offset: int.from_bytes(bytes(data[offset:offset + 2]), 'little')
                u32 = lambda offset: int.from_bytes(bytes(data[offset:offset + 4]), 'little')
                row.update(
                    pwm_percent=round(self._trace_signed(u16(0), 16) / 8.85, 2),
                    load_percent=round(self._trace_signed(u16(2), 16) / 10.0, 2),
                    velocity_raw=self._trace_signed(u32(4), 32),
                    position_tick=self._trace_signed(u32(8), 32),
                    voltage_v=round(u16(20) / 10.0, 2),
                    temperature_c=data[22],
                )
            if alert and comm == 0:
                # A read may still return the data with the hardware alert bit.
                # Extra reads are attempted only for a received hardware alert.
                # On a timeout, two more reads would delay the motion thread.
                try:
                    error_value, error_comm, _ = bridge.pk.read1ByteTxRx(
                        bridge.ph, motor_id, 70)
                    if error_comm == 0:
                        row['hardware_error'] = '0x%02X' % error_value
                    torque_value, torque_comm, _ = bridge.pk.read1ByteTxRx(
                        bridge.ph, motor_id, 64)
                    if torque_comm == 0:
                        row['torque_enable'] = torque_value
                except Exception:
                    pass  # Preserve the original alert in CSV.
            return row
        finally:
            bridge.bus_lock.release()

    def _start_motor_trace(self):
        ids_value = str(self.get_parameter('motor_trace_ids').value)
        try:
            ids = list(dict.fromkeys(int(s.strip()) for s in ids_value.split(',')))
            if not ids or any(i not in self.bridge.all_ids for i in ids):
                raise ValueError('only A-arm IDs 1..7 are supported')
        except ValueError as exc:
            raise ValueError('motor_trace_ids must be comma-separated A-arm IDs') from exc
        period = float(self.get_parameter('motor_trace_period_sec').value)
        if not math.isfinite(period) or period < 0.5:
            raise ValueError('motor_trace_period_sec must be >= 0.5')
        directory = os.path.abspath(os.path.expanduser(
            str(self.get_parameter('motor_trace_dir').value)))
        # Do not create the mount directory on the system drive if the SSD is
        # disconnected. No silent fallback to ~/.ros/log.
        if os.path.commonpath((directory, DEFAULT_MOTOR_TRACE_MOUNT)) == \
                DEFAULT_MOTOR_TRACE_MOUNT and not os.path.ismount(DEFAULT_MOTOR_TRACE_MOUNT):
            self.get_logger().error(
                'Motor trace disabled: SSD is not mounted at %s'
                % DEFAULT_MOTOR_TRACE_MOUNT)
            return
        try:
            os.makedirs(directory, exist_ok=True)
        except OSError as exc:
            self.get_logger().error('Motor trace disabled: cannot create %s: %s'
                                    % (directory, exc))
            return
        stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        self.motor_trace_path = os.path.join(
            directory, 'assembly_motor_%s_%d.csv' % (stamp, os.getpid()))
        try:
            self._trace_file = open(self.motor_trace_path, 'w', newline='', buffering=1)
        except OSError as exc:
            self.get_logger().error('Motor trace disabled: cannot open %s: %s'
                                    % (self.motor_trace_path, exc))
            self.motor_trace_path = None
            return
        fieldnames = ('timestamp', 'monotonic_sec', 'phase', 'motor_id',
                      'voltage_v', 'load_percent', 'pwm_percent', 'velocity_raw',
                      'position_tick', 'temperature_c', 'torque_enable',
                      'hardware_error', 'comm_result', 'packet_error', 'latency_ms')
        writer = csv.DictWriter(self._trace_file, fieldnames=fieldnames)
        writer.writeheader()

        def trace_loop():
            last_warn = {}
            while not self._trace_stop.is_set():
                cycle_start = time.monotonic()
                for motor_id in ids:
                    if self._trace_stop.is_set():
                        break
                    try:
                        row = self._trace_one(motor_id)
                        if row is None:
                            continue
                        row.update(timestamp=datetime.datetime.now().astimezone().isoformat(
                            timespec='milliseconds'), monotonic_sec=round(time.monotonic(), 3),
                            phase=self.phase, motor_id=motor_id)
                        writer.writerow(row)
                        self._trace_file.flush()
                        if ((row['comm_result'] or row['packet_error'] or
                             row['hardware_error'] not in ('', '0x00')) and
                                time.monotonic() - last_warn.get(motor_id, -1e9) > 5.0):
                            last_warn[motor_id] = time.monotonic()
                            self.get_logger().warning(
                                'MOTOR TRACE ID%d comm=%s alert=0x%02X hwerr=%s '
                                'torque=%s V=%s; CSV=%s'
                                % (motor_id, row['comm_result'], row['packet_error'],
                                   row['hardware_error'], row['torque_enable'],
                                   row['voltage_v'], self.motor_trace_path))
                    except Exception as exc:
                        if time.monotonic() - last_warn.get(motor_id, -1e9) > 5.0:
                            last_warn[motor_id] = time.monotonic()
                            self.get_logger().warning('MOTOR TRACE ID%d processing failed: %s'
                                                      % (motor_id, exc))
                self._trace_stop.wait(max(0.0, period - (time.monotonic() - cycle_start)))

        self._trace_thread = threading.Thread(target=trace_loop,
                                               name='assembly_motor_trace', daemon=True)
        self._trace_thread.start()
        self.get_logger().info('Motor trace IDs=%s every %.2fs: %s'
                               % (ids, period, self.motor_trace_path))

    def destroy_node(self):
        self._trace_stop.set()
        if self._trace_thread is not None:
            self._trace_thread.join(timeout=2.0)
        if self._trace_file is not None:
            self._trace_file.close()
        super().destroy_node()

    def _start_sequence(self):
        self._start_assembly_cycle()

    def _start_assembly_cycle(self):
        if not self._ready('start', centric=True):
            return
        if self._pending_manual_command is not None:
            self.get_logger().warning('Wait for the queued manual movement to complete')
            return
        if self._cycle_uncertain:
            self.get_logger().error('Cycle state uncertain after stop during grasp/release. '
                                    'Inspect the part and tool; empty the gripper, then '
                                    'use cycle reset and scan before starting again.')
            return
        if self._cycle_index == len(ITEMS):
            self.get_logger().error('Cycle completed before stop; inspect the parts, '
                                    'then use cycle reset for a new cycle')
            return
        try:
            holes = {kind: self._fresh_vision_hole(kind) for kind in ('mid', 'side')}
        except RuntimeError as exc:
            self.get_logger().error('start requires scan and fresh mid/side: %s' % exc)
            return
        if self._cycle_index is None:
            if self.held_item is not None or self.insert_retreat_path is not None:
                self.get_logger().error('start requires empty gripper and no pending retreat')
                return
            self._cycle_holes = holes
            self._cycle_index = 0
        elif any(np.linalg.norm(holes[kind] - self._cycle_holes[kind]) > 5.0
                 for kind in ('mid', 'side')):
            self.get_logger().error('Vision holes moved >5 mm since cycle start; '
                                    'inspect assembly and use cycle reset if needed')
            return
        if self.held_item not in (None, ITEMS[self._cycle_index]):
            self.get_logger().error('held item does not match cycle checkpoint')
            return
        if self.use_robot:
            try:
                actual_q = np.degrees(self._current_q_rad())
                if not self.rail.query():
                    raise RuntimeError('rail status query could not be sent')
            except Exception as exc:
                self.get_logger().error('start blocked: A-arm/rail state unreadable: %s' % exc)
                return
        else:
            actual_q = np.degrees(self._current_q_rad())
        if self._cycle_needs_clearance:
            travel = np.asarray(self.rail_travel_pose_deg, dtype=float)
            error = np.abs((actual_q[:5] - travel + 180.0) % 360.0 - 180.0)
            rail_mm = self.rail.position_mm if self.rail is not None else self.sim_rail_mm
            if np.max(error) > 5.0 or abs(rail_mm) > self.rail_tol_mm:
                self.get_logger().error('start blocked after interrupted insertion: '
                                        'inspect the tool, then manually move A to '
                                        '%s and rail to 0 mm before restarting'
                                        % list(ASSEMBLY_RAIL_TRAVEL_POSE_DEG))
                return
            # A manual correction made the old retreat trajectory obsolete.
            self.insert_retreat_path = None
            self._cycle_needs_clearance = False
        if self.held_item is not None:
            self.get_logger().warning('Resuming placement while holding %s; confirm the '
                                      'part is still secured in the gripper' % self.held_item)
        self._trial('ASSEMBLY_SIX_PARTS', self._run_assembly_cycle)

    def _run_assembly_cycle(self):
        self._cycle_running = True
        try:
            while self._cycle_index < len(ITEMS):
                self._check_stop()
                item = ITEMS[self._cycle_index]
                self.get_logger().info('Assembly checkpoint %d/%d: %s'
                                       % (self._cycle_index + 1, len(ITEMS), item))
                if self._cycle_released_item == item:
                    if self.insert_retreat_path is not None:
                        self._retreat_after_insert()
                    self._cycle_released_item = None
                    self._cycle_index += 1
                    continue
                if self.held_item is None:
                    self._pickup(item)
                elif self.held_item != item:
                    raise RuntimeError('unexpected held item %s at %s checkpoint'
                                       % (self.held_item, item))
                self._place(item, vision_point=self._cycle_holes[
                    self._item_settings(item)[3]])
                self._cycle_released_item = None
                self._cycle_index += 1
            self.get_logger().info('Assembly six-part cycle complete')
            self._cycle_index = None
            self._cycle_holes = None
        finally:
            self._cycle_running = False

    def _abort_cycle(self):
        if self._pause_event.is_set() and self._cycle_index is not None:
            phase = str(getattr(self, '_paused_phase', '') or '')
            if getattr(self, '_pause_needs_recovery', False):
                self._cycle_uncertain = True
            if '_VERTICAL_' in phase or '_RELEASE' in phase:
                self._cycle_needs_clearance = True
        super()._abort_cycle()
        if self._cycle_index is not None:
            if self._cycle_uncertain:
                self.get_logger().warning('Stopped during part contact at %s; '
                                          'automatic start is blocked until the part '
                                          'and gripper are reconciled'
                                          % ITEMS[self._cycle_index])
            else:
                self.get_logger().warning('Automatic cycle held at %s; start can '
                                          'resume from this part checkpoint'
                                          % ITEMS[self._cycle_index])

    def _dispatch_manual_after_stop(self):
        raw = self._pending_manual_command
        if raw is None or self.busy or self._pause_event.is_set():
            return
        self._pending_manual_command = None
        self.get_logger().warning('Stopped cycle cleared; executing requested manual command')
        self._handle_command(raw)

    def _on_top(self, _msg):
        # An unrelated top detector must never start the full process.
        return

    def _tick(self):
        self._poll_viewer_commands()
        if self._button_clicked('stop'):
            self._pause_cycle()
        if self._button_clicked('resume_cycle'):
            self._resume_cycle()
        if self._button_clicked('abort_cycle'):
            self._abort_cycle()
        self._send_viewer_state()

    def _fail(self, reason):
        self._last_debug_fault = str(reason)
        self._debug_fault = True
        super()._fail(reason)

    def _herehome(self):
        """Record an already-straight physical pose as A=[0,0,0,0,0]."""
        if self.bridge is not None:
            self.bridge.capture_home()
            actual = np.degrees(self.bridge.read_joints()[:5])
            if len(actual) != 5 or max(abs(actual)) > 0.25:
                self.reference_ready = False
                raise RuntimeError('straight zero verification failed: %s' % actual)
        self.reference_ready = True
        self._set_sim_state(q_deg=[0.0] * 5)
        self.get_logger().warning('CURRENT POSE stored as straight A=[0,0,0,0,0]; no motion')

    def _mechhome(self):
        """Apply measured stop-to-straight travel to live ticks, then move home."""
        if self.bridge is None:
            self.reference_ready = True
            self._set_sim_state(q_deg=list(MECHANICAL_REFERENCE_DEG))
            self._move_arm([0.0] * 5)
            return
        # The code cannot sense a hard stop. The operator must physically
        # place J2/J3/J4 there and align the unbounded J1/J5 to their marks.
        self.reference_ready = False
        present_ticks = {motor_id: self.bridge._present(motor_id)
                         for joint in JOINT_MOTORS[:5] for motor_id in joint['ids']}
        new_zero = {motor_id: present + MECHANICAL_HOME_OFFSETS.get(motor_id, 0.0)
                    for motor_id, present in present_ticks.items()}
        self.get_logger().warning(
            'MECH HOME measured limits=%s; straight goals=%s; '
            'ID4 offset derived from ID3 (recorded ID4 differs by %.1f ticks); '
            'ID6 offset derived from ID5' %
            (present_ticks, {i: round(v) for i, v in new_zero.items()},
             MECHANICAL_HOME_OFFSETS[4] - MEASURED_ID4_OFFSET))
        self.bridge.zero_tick.update(new_zero)
        measured = np.degrees(self.bridge.read_joints()[:5])
        if len(measured) != 5 or max(abs(measured - MECHANICAL_REFERENCE_DEG)) > 0.25:
            raise RuntimeError('mechanical zero verification failed: %s' % measured)
        self.reference_ready = True
        self._set_sim_state(q_deg=list(MECHANICAL_REFERENCE_DEG))
        speed_deg_s = 8.0
        profile = {}
        for joint in JOINT_MOTORS[:5]:
            pv = max(1, int(round(speed_deg_s * joint['gear'] / (0.229 * 6.0))))
            for motor_id in joint['ids']:
                profile[motor_id] = min(PROFILE_VELOCITY[motor_id], pv)
        self.get_logger().warning(
            'MECH HOME: assuming physical A=%s. Sending straight-home goal '
            'at <=%.1f deg/s' % (MECHANICAL_REFERENCE_DEG, speed_deg_s))
        try:
            for motor_id, pv in profile.items():
                self.bridge._w4(motor_id, 112, pv)
            self.bridge.command_joints([0.0] * 5)
            deadline = time.monotonic() + max(
                self.joint_timeout, max(abs(v) for v in MECHANICAL_REFERENCE_DEG)
                / speed_deg_s + 30.0)
            while time.monotonic() < deadline:
                self._check_stop()
                actual = np.asarray(self.bridge.read_joints()[:5], dtype=float)
                self._set_sim_state(q_deg=np.degrees(actual).tolist())
                if max(abs(np.degrees(np.arctan2(np.sin(actual),
                                                np.cos(actual))))) <= 1.5:
                    # read_joints uses only the first motor of each pair;
                    # check ID4 and ID6 too before reporting completion.
                    paired_ok = True
                    for joint in JOINT_MOTORS[2:4]:
                        motor_id = joint['ids'][1]
                        tolerance_ticks = 1.5 * joint['gear'] * 4096 / 360.0
                        actual_tick = self.bridge._present(motor_id)
                        if abs(actual_tick - round(new_zero[motor_id])) > tolerance_ticks:
                            paired_ok = False
                    if paired_ok:
                        self.get_logger().info('MECH HOME complete: A straight; '
                                               'ID4/ID6 verified')
                        return
                time.sleep(0.15)
            raise RuntimeError('MECH HOME position timeout; holding current pose')
        finally:
            for motor_id in profile:
                try:
                    self.bridge._w4(motor_id, 112, PROFILE_VELOCITY[motor_id])
                except Exception as exc:
                    self.get_logger().error('restore profile velocity ID%d failed: %s'
                                            % (motor_id, exc))

    def _item_settings(self, name):
        lookup = {
            'shaft1': (629.0, SHAFT_PICK_DEG,
                       self.centric_preclose_angle_deg, 'side', ASSEMBLY_INSERT_Z_MM['shaft1']),
            'shaft2': (690.0, SHAFT_PICK_DEG,
                       self.centric_preclose_angle_deg, 'mid', ASSEMBLY_INSERT_Z_MM['shaft2']),
            'bearing1': (629.0, BEARING_PICK_DEG,
                         self.bearing_preclose_angle_deg, 'side', ASSEMBLY_INSERT_Z_MM['bearing1']),
            'bearing2': (690.0, BEARING_PICK_DEG,
                         self.bearing_preclose_angle_deg, 'mid', ASSEMBLY_INSERT_Z_MM['bearing2']),
            'gear1': (802.0, SHAFT_PICK_DEG,
                      100.0, 'side', ASSEMBLY_INSERT_Z_MM['gear1']),
            'gear2': (889.0, SHAFT_PICK_DEG,
                      50.0, 'mid', ASSEMBLY_INSERT_Z_MM['gear2']),
        }
        rail, pose, preclose, hole, z = lookup[name]
        rail, pose = self.pick_overrides.get(name, (rail, pose))
        return rail, list(pose), preclose, hole, self.place_z_overrides.get(name, z)

    def _hole(self, kind):
        if kind in self.hole_overrides:
            return self.hole_overrides[kind].copy()
        if self.latest_holes is None:
            raise RuntimeError('Hole XY required: hole mid X Y / hole side X Y')
        return np.asarray(self.latest_holes[:2] if kind == 'mid'
                          else self.latest_holes[2:4], dtype=float).copy()

    def _ready(self, action, centric=False, item=None):
        if self.busy or self._pause_event.is_set() or self._tool_change_waiting:
            self.get_logger().warning('%s blocked: operation in progress or paused' % action)
            return False
        if self._debug_fault or self.phase == 'ERROR_HOLDING':
            # Debugging must permit an explicit manual retry without recapturing
            # A's straight-pose zero.  Never restart an automatic sequence on
            # the strength of a successful unrelated manual command.
            manual = ('aj', 'r', 'grip', 'grasp', 'open', 'motors',
                      'tool', 'suction', 'retreat', 'scan')
            if action not in manual:
                self.get_logger().error(
                    '%s blocked after fault; scan and manual '
                    'aj/r/grip/tool/suction/motors remain available. Last fault: %s'
                    % (action, self._last_debug_fault))
                return False
            if action == 'scan':
                self.get_logger().warning(
                    'scan after fault: refresh vision coordinates only; '
                    'automatic insertion remains blocked until the cause is corrected')
            else:
                self.get_logger().warning(
                    '%s manual retry after fault: %s. Inspect clearance, tool and '
                    'motor LEDs; hardware protection remains active.'
                    % (action, self._last_debug_fault))
        if action != 'scan' and self.use_robot:
            if self.bridge is None or self.rail is None:
                self.get_logger().error('%s blocked: A arm or rail unavailable' % action)
                return False
            if not self.reference_ready:
                self.get_logger().error('%s blocked: A-arm reference not ready; '
                                        'use mechhome at the mechanical limits '
                                        'or herehome when already straight' % action)
                return False
        if centric and (self.tool != 'centric' or (self.use_robot and self.gripper is None)):
            self.get_logger().error('%s requires tool centric after physically installing ID8' % action)
            return False
        if item is not None and self.held_item is not None and self.held_item != item:
            self.get_logger().error('%s already holds %s' % (action, self.held_item))
            return False
        return True

    def _leave_insert_if_needed(self, action):
        if self.insert_retreat_path is not None:
            if self._debug_fault:
                self.insert_retreat_path = None
                self.get_logger().warning(
                    '%s: insertion retreat discarded after fault; manual motion '
                    'requires direct clearance inspection' % action)
                return
            self.get_logger().warning('%s: automatically retreating from insertion '
                                      'depth before next motion' % action)
            self._set_phase('AUTO_RETREAT_BEFORE_' + action.upper())
            self._retreat_after_insert()

    def _trial(self, phase, action):
        def worker():
            try:
                action()
            except InsertionPlanRejected as exc:
                # The planner rejected this candidate before sending the
                # insertion trajectory. Keep the grasp and stored coordinates;
                # _start_auxiliary will return to READY and accept a new Z.
                self.get_logger().warning(
                    'Insertion path rejected: %s. Gripper remains closed; '
                    'try mid/side again with a safer Z. No A-arm insertion '
                    'trajectory was sent.' % exc)
            except Exception:
                # Communication, hardware and movement faults still stop the
                # cell; the actual arm/part state may have changed.
                raise
        self._start_auxiliary(phase, worker)

    def _manual_auto_grasp(self):
        # The gripper already commands HOLD at the reached position when it
        # reports a timeout or an empty grasp. Those two outcomes are normal
        # while tuning a feeder, not evidence of a motor hardware fault.
        try:
            self._finish_grasp()
        except RuntimeError as exc:
            if str(exc) not in ('centric grasp failed: timeout',
                                'centric grasp failed: no object'):
                raise
            self._grasp_hold_position = None
            self._grasp_hold_ratio = None
            self._assembly_cycle_active = False
            self.manual_grasp_ready = False
            self._set_sim_state(gripper='NO GRASP')
            self.get_logger().warning('%s; gripper holding position. '
                                      'Check part contact, then use grip open or grip 100.' % exc)
        else:
            self.manual_grasp_ready = True
            self.get_logger().info('Manual grasp confirmed; scan then use mid Z_mm or side Z_mm')

    def _pickup(self, item):
        self._leave_insert_if_needed('pickup_' + item)
        rail, pose, preclose, _, _ = self._item_settings(item)
        self._set_phase(item.upper() + '_TRAVEL_TO_SUPPLY')
        self._move_arm(self.rail_travel_pose_deg)
        self._move_rail(rail)
        self._set_phase('%s_PICK_AND_PRECLOSE_%.0fDEG' % (item.upper(), preclose))
        self._move_arm_with_preclose(pose, preclose / self.centric_full_close_angle_deg)
        self._set_phase(item.upper() + '_FINAL_GRASP')
        self._finish_grasp()
        self._assembly_cycle_active = True
        self.held_item = item
        self.manual_grasp_ready = False
        self._set_phase(item.upper() + '_VERTICAL_LIFT')
        self._lift_after_grasp()
        self.get_logger().info('%s lifted; use place %s or adjust and retry' % (item, item))

    def _place(self, item, z_override=None, vision_point=None):
        self._leave_insert_if_needed('place_' + item)
        _, _, _, hole, z = self._item_settings(item)
        if z_override is not None:
            z = z_override
        point = self._hole(hole) if vision_point is None else np.asarray(
            vision_point, dtype=float).copy()
        self._set_phase(item.upper() + '_TRAVEL_TO_BASEPLATE')
        self._move_arm(self.rail_travel_pose_deg)
        self._move_rail(0)
        self._insert_vision(point, hole, z, rotate_q5=(item == 'gear2'))
        self._retreat_after_insert()
        self.get_logger().info('%s placed at %s %s Z=%.1f mm'
                               % (item, hole, point.tolist(), z))

    def _part(self, item, z_override=None):
        self._pickup(item)
        self._place(item, z_override=z_override)

    def _fresh_vision_hole(self, kind):
        if self.vision_scan_started is None or self.latest_holes is None or \
                self.latest_holes_stamp is None or \
                self.latest_holes_stamp < self.vision_scan_started:
            raise RuntimeError('fresh vision holes not ready; run scan and wait for '
                               'holes cached from /detected_holes')
        point = np.asarray(self.latest_holes[:2] if kind == 'mid'
                           else self.latest_holes[2:4], dtype=float).copy()
        if not np.isfinite(point).all() or not self._in_a3(point):
            raise RuntimeError('invalid %s vision hole coordinate' % kind)
        return point

    def _insert_vision(self, point, hole, z, rotate_q5=False):
        # A dedicated diagnostic action. It stops immediately after opening
        # at the insertion depth; retreat is an explicit later command.
        self._leave_insert_if_needed('insert_' + hole)
        self.insert_retreat_path = None
        self._assembly_cycle_active = True
        # Do not visit the rail-travel joint pose merely to insert a part.
        # If the rail still needs to return, the operator must have already
        # positioned A safely for that translation.
        present_rail = (self.rail.position_mm if self.rail is not None
                        else self.sim_rail_mm)
        if abs(present_rail) > self.rail_tol_mm:
            self._set_phase(hole.upper() + '_RAIL_HOME_BEFORE_INSERT')
            self._move_rail(0.0)
        # The inherited planner puts its high point at target_z_mm + approach_mm,
        # regardless of the requested low Z. Scope its two settings so *this*
        # insertion begins exactly clearance_mm above the requested final Z.
        original_target_z, original_approach = self.target_z_mm, self.approach_mm
        try:
            self.target_z_mm = float(z)
            self.approach_mm = self.insert_approach_clearance_mm
            try:
                approach, descent, ascent, world_low = self._build_vertical_paths(
                    float(point[0]), float(point[1]), target_z_mm=float(z))
            except RuntimeError as exc:
                # The inherited planner checks the full descent before any
                # insertion trajectory is sent. Its geometry/collision
                # failures are safe to replan with another target Z. Read or
                # transport failures must still take the normal fault path.
                reason = str(exc)
                if (' collision at sample ' in reason or
                        reason.startswith(('no exact vertical IK at approach point',
                                           'vertical descent failed:',
                                           'vertical top approach failed:',
                                           'vertical path planning failed',
                                           'approach retiming failed:'))):
                    raise InsertionPlanRejected(reason) from exc
                raise
        finally:
            self.target_z_mm, self.approach_mm = original_target_z, original_approach
        descent = np.asarray(descent, dtype=float)
        ascent = np.asarray(ascent, dtype=float)
        if descent.ndim != 2 or descent.shape[0] < 2 or descent.shape[1] != 5:
            raise RuntimeError('insertion descent must have at least two five-joint samples')
        # Pick the sample at the halfway *height*, rather than halfway in
        # time: retiming makes those different. Stop at this vertical pose
        # before starting any q5 turn.
        if len(descent) < 3:
            raise InsertionPlanRejected('insertion path has no interior midpoint')
        mid_world = self.a3_to_world(
            float(point[0]), float(point[1]),
            float(z) + self.insert_approach_clearance_mm / 2.0)
        mid_arm_z = float(self.world_to_arm(mid_world)[2])
        heights = np.array([fk_pos(q[:4])[2] for q in descent])
        mid_index = int(np.clip(np.argmin(np.abs(heights - mid_arm_z)),
                                1, len(descent) - 2))
        q5_start = float(descent[0, 4])
        q5_end = q5_start
        turn_time = (len(descent) - mid_index - 1) / self.planner_hz
        if rotate_q5:
            q5_limit_deg_s = (PROFILE_VELOCITY[7] * 0.229 * 6.0 /
                              JOINT_MOTORS[4]['gear'] * self.speed_margin)
            if not math.isfinite(q5_limit_deg_s) or q5_limit_deg_s <= 0:
                raise InsertionPlanRejected(
                    'invalid q5 wrist speed limit: %s deg/s' % q5_limit_deg_s)
            # Rotate for the entire midpoint-to-target descent. The requested
            # angle is a maximum, not a reason to stretch the vertical motion.
            # At a typical 0.33 s descent, 45 deg/s gives about 15 degrees.
            safe_turn_speed = min(45.0, q5_limit_deg_s * 0.75)
            turn_deg = math.copysign(
                min(abs(self.insert_q5_rotation_deg),
                    safe_turn_speed * turn_time), self.insert_q5_rotation_deg)
            q5_end = q5_start + math.radians(turn_deg)
            descent[:mid_index + 1, 4] = q5_start
            descent[mid_index:, 4] = q5_start + (q5_end - q5_start) * \
                np.linspace(0.0, 1.0, len(descent) - mid_index)
            ascent[:, 4] = q5_end
            if abs(turn_deg) < abs(self.insert_q5_rotation_deg) - 1e-6:
                self.get_logger().info(
                    'q5 insertion turn %.2f deg (requested maximum %.2f deg) '
                    'during %.2f s descent, safe speed %.1f deg/s; '
                    'vertical descent timing unchanged'
                    % (turn_deg, self.insert_q5_rotation_deg, turn_time,
                       safe_turn_speed))
        error = self._collision_error(descent[:, :4],
                                      hole + ' insertion descent')
        if error:
            raise InsertionPlanRejected(error)
        error = self._collision_error(ascent[:, :4],
                                      hole + ' insertion retreat')
        if error:
            raise InsertionPlanRejected(error)
        self._move_marker(hole, world_low)
        self.get_logger().info('%s insertion: Z %.1f -> %.1f mm, q5 %.2f -> %.2f deg '
                               'over %.2fs (%s); release after descent'
                               % (hole, float(z) + self.insert_approach_clearance_mm,
                                  float(z), math.degrees(q5_start),
                                  math.degrees(q5_end), turn_time,
                                  'rotating' if rotate_q5 else 'fixed'))
        self._set_phase(hole.upper() + '_VERTICAL_TOP_APPROACH')
        self._execute_joint_trajectory(approach)
        self._set_phase(hole.upper() + '_VERTICAL_MIDPOINT')
        self._execute_joint_trajectory(descent[:mid_index + 1])
        actual_mid = np.degrees(self._current_q_rad())
        actual_pitch_sum = float(sum(actual_mid[1:4]))
        pitch_error = (actual_pitch_sum - 180.0 + 180.0) % 360.0 - 180.0
        self.get_logger().info(
            '%s midpoint: Z=%.1f mm, q2+q3+q4=%.2f deg '
            '(model vertical error=%.2f deg); q5 turn starts on remaining descent'
            % (hole, float(z) + self.insert_approach_clearance_mm / 2.0,
               actual_pitch_sum, pitch_error))
        self._check_stop()
        self._set_phase(hole.upper() + '_VERTICAL_DESCENT')
        self._execute_joint_trajectory(descent[mid_index:])
        self.insert_retreat_path = ascent
        self._set_phase(hole.upper() + '_WAIT_BEFORE_RELEASE')
        deadline = self._process_time() + INSERT_RELEASE_DWELL_SEC
        while self._process_time() < deadline:
            self._check_stop()  # Stop immediately; pause freezes the dwell clock.
            time.sleep(min(0.01, max(0.001, deadline - self._process_time())))
        self._check_stop()
        self._set_phase(hole.upper() + '_RELEASE')
        if not self._move_gripper_ratio(self.release_ratio):
            raise RuntimeError('%s centric release failed' % hole)
        if self._cycle_running:
            self._cycle_released_item = self.held_item
        self.held_item = None
        self.manual_grasp_ready = False
        self._assembly_cycle_active = False
        self._grasp_hold_position = None
        self._grasp_hold_ratio = None
        self.get_logger().info('%s vision XY=(%.3f, %.3f) Z=%.1f mm: '
                               'released; holding at insertion depth. Use retreat.'
                               % (hole, point[0], point[1], z))

    def _retreat_after_insert(self):
        path = self.insert_retreat_path
        if path is None:
            raise RuntimeError('no pending insertion retreat')
        self._execute_joint_trajectory(path)
        self.insert_retreat_path = None

    def _attach_centric(self):
        # ID8 shares the bridge port and lock; it must already be installed
        # and mechanically in its OPEN reference pose before this call.
        if not self.use_robot:
            self.tool = 'centric'
            return
        grip = CentricGripper(
            port=str(self.get_parameter('dxl_port').value),
            baud=int(self.get_parameter('dxl_baud').value),
            dxl_id=int(self.get_parameter('centric_id').value),
            logger=self.get_logger(),
            planet_max_angle_deg=float(self.get_parameter('centric_planet_max_angle_deg').value),
            normal_profile_velocity=44, auto_profile_velocity=44,
            auto_load_threshold_percent=float(
                self.get_parameter('centric_auto_load_threshold_percent').value),
            auto_load_count_limit=int(self.get_parameter('centric_auto_load_count_limit').value),
            auto_grasp_timeout_sec=float(
                self.get_parameter('centric_auto_grasp_timeout_sec').value),
            emergency_load_percent=float(
                self.get_parameter('centric_emergency_load_percent').value),
            communication_retry_count=int(
                self.get_parameter('centric_communication_retry_count').value))
        grip.connect_shared(self.bridge.ph, self.bridge.pk,
                            self.bridge.bus_lock, capture_open_on_start=True)
        self.gripper = grip
        self.use_gripper = True
        self.tool = 'centric'
        usable_angle = (self.centric_full_close_angle_deg *
                        grip.safe_move_pulse / grip.max_move_pulse)
        self.get_logger().info('ID8 attached to shared A bus; OPEN position captured; '
                               'configured limit=%.1f deg, safe travel=%.1f deg'
                               % (self.centric_full_close_angle_deg, usable_angle))

    def _detach_centric(self):
        if self.gripper is not None:
            self.gripper.close(torque_off=True)
            self.gripper = None
        self.use_gripper = False
        self.tool = 'none'
        self.manual_grasp_ready = False
        self._grasp_hold_position = None
        self._grasp_hold_ratio = None
        self.get_logger().info('ID8 torque off. Now physically exchange the tool.')

    def _lidpick(self):
        self._leave_insert_if_needed('lidpick')
        if self.lid_pick_pose is None:
            raise RuntimeError('lid_pick_pose_deg is not configured')
        self._move_arm(self.rail_travel_pose_deg)
        self._move_rail(self.lid_rail)
        self._move_arm(self.lid_pick_pose)
        self.get_logger().info('Lid pick pose reached; suction on, then lidplace')

    def _lidplace(self):
        self._leave_insert_if_needed('lidplace')
        if not self.suction_enabled:
            raise RuntimeError('suction is OFF; use suction on after picking the lid')
        point = np.asarray(self.base_assembly_point, dtype=float).copy()
        self._move_arm(self.rail_travel_pose_deg)
        self._move_rail(0)
        self._set_phase('LID_VERTICAL_PROBE')
        self._lid_probe_at(point)
        self.get_logger().info('Lid probe/retreat complete, suction still ON')

    def _handle_command(self, raw):
        parts = str(raw).strip().lower().split()
        if not parts:
            return
        cmd = parts[0]
        if cmd in ('stop', 's', 'estop', '0', 'resume', 'abort', 'abort_cycle'):
            return super()._handle_command(raw)
        if cmd in ('aj', 'r') and self._pause_event.is_set():
            if self._pending_manual_command is not None:
                self.get_logger().warning('Another manual command is already queued')
                return
            # A manual move changes the measured pause pose. Cancel the
            # interrupted trajectory; never resume that old sample afterward.
            self._pending_manual_command = str(raw)
            self._abort_cycle()
            self.get_logger().warning('Manual %s queued after E-stop; '
                                      'start later resumes from the current part checkpoint'
                                      % cmd)
            return
        if cmd in ('help', '?'):
            self.get_logger().info(
                'mechhome (at mechanical stops) | herehome (already straight) | '
                'aj q1 q2 q3 q4 q5 | r mm | hole mid|side X Y | '
                'setpick item rail_mm q1 q2 q3 q4 q5 | setz item Z_mm | '
                'tool centric|off|suction | grip 0..180|auto|open | '
                'scan | mid/side Z_mm (fixed q5) | mid/side rZ_mm (rotate q5) | retreat | '
                'shaft1|shaft2|bearing1|bearing2|gear1|gear2 [Z_mm] | '
                'start (six parts, scan first) | cycle reset | '
                'assembly | part item | pickup item | place item | '
                'lidpick | lidplace | suction on|off | stop | resume | abort | status | motors '
                '(manual commands remain available after a fault)')
            return
        if cmd == 'status':
            vision_ready = (self.vision_scan_started is not None and
                            self.latest_holes_stamp is not None and
                            self.latest_holes_stamp >= self.vision_scan_started)
            checkpoint = (ITEMS[self._cycle_index] if self._cycle_index is not None
                          and self._cycle_index < len(ITEMS) else None)
            self.get_logger().info('debug reference=%s ready=%s, held=%s, tool=%s, suction=%s, holes=%s, '
                                   'vision_ready=%s, retreat_pending=%s, checkpoint=%s, '
                                   'cycle_uncertain=%s, fault=%s, last_fault=%s'
                                   % (self.assembly_reference_mode, self.reference_ready,
                                      self.held_item, self.tool, self.suction_enabled,
                                      {k: v.tolist() for k, v in self.hole_overrides.items()},
                                      vision_ready, self.insert_retreat_path is not None,
                                      checkpoint, self._cycle_uncertain,
                                      self._debug_fault, self._last_debug_fault))
            return
        if cmd == 'cycle' and parts[1:] == ['reset']:
            if self.busy or self._pause_event.is_set() or self.held_item is not None or \
                    self.insert_retreat_path is not None:
                self.get_logger().error('cycle reset requires idle arm, empty gripper, '
                                        'and no pending insertion retreat')
                return
            self._cycle_index = None
            self._cycle_holes = None
            self._cycle_released_item = None
            self._cycle_uncertain = False
            self._cycle_needs_clearance = False
            self.get_logger().warning('Cycle checkpoint cleared; remove any already '
                                      'assembled parts before restarting at shaft1')
            return
        if cmd in ('mechhome', 'herehome'):
            if len(parts) != 1 or self.busy or self._pause_event.is_set() or \
                    self._tool_change_waiting or self.held_item is not None or \
                    self.insert_retreat_path is not None or self._debug_fault:
                self.get_logger().error(
                    '%s requires idle arm with no held part, pending insertion '
                    'or hardware fault' % cmd)
                return
            rail_mm = self.rail.position_mm if self.rail is not None else self.sim_rail_mm
            if abs(rail_mm) > self.rail_tol_mm:
                self.get_logger().error('%s requires rail at 0 mm' % cmd)
                return
            self._trial('A_' + cmd.upper(), self._mechhome if cmd == 'mechhome'
                        else self._herehome)
            return
        if cmd == 'start':
            if len(parts) == 1:
                self._start_assembly_cycle()
            else:
                self.get_logger().error('usage: start')
            return
        if cmd in ('bj', 'bhome', 'b_home', 'run', 'sample',
                   'rail', 'home', 'refhome', 'joint', 'j'):
            self.get_logger().error('%s disabled in assembly debug; use aj/r/part' % cmd)
            return
        if cmd in ('aj', 'r'):
            if not self._ready(cmd):
                return
            if len(parts) != (6 if cmd == 'aj' else 2):
                self.get_logger().error('usage: aj q1 q2 q3 q4 q5 | r mm')
                return
            try:
                values = [float(v) for v in parts[1:]]
            except ValueError:
                values = [math.nan]
            if not all(math.isfinite(v) for v in values):
                self.get_logger().error('%s requires finite numbers' % cmd)
                return
            if cmd == 'r' and not 0.0 <= values[0] <= RAIL_MAX_M * 1000.0:
                self.get_logger().error('rail target outside allowed range')
                return
            if self.phase == 'ERROR_HOLDING':
                self._set_phase('FAULT_MANUAL_READY')
            def move_manually():
                self._leave_insert_if_needed(cmd)
                if cmd == 'aj':
                    self._move_arm(values)
                else:
                    self.get_logger().warning('RAIL ONLY: check arm/tool clearance')
                    self._move_rail(values[0])
            self._trial('A_MANUAL_JOINT_MOVE' if cmd == 'aj'
                        else 'RAIL_ONLY_TO_%.0fMM' % values[0], move_manually)
            return
        if cmd == 'scan':
            if len(parts) != 1 or not self._ready(cmd):
                return
            self.latest_holes = None
            self.latest_holes_stamp = None
            self.vision_scan_started = time.monotonic()
            self.product_pub.publish(String(data='baseplate'))
            self.hole_pub.publish(String(data='START'))
            self.get_logger().info('Vision scan requested: click four A3 corners in the '
                                   'hole window if needed; wait for fresh mid/side coordinates')
            return
        if cmd in ('mid', 'side'):
            if len(parts) != 2:
                self.get_logger().error('usage: mid/side Z_mm | mid/side rZ_mm')
                return
            if not self._ready(cmd, centric=True):
                return
            rotate_q5 = parts[1].startswith('r')
            z_text = parts[1][1:] if rotate_q5 else parts[1]
            try:
                z = float(z_text)
            except ValueError:
                z = math.nan
            # The collision planner must approve every sample, including Z=0
            # and small negative trials. Never bypass its floor check.
            max_z = self.target_z_mm + self.approach_mm - 5.0
            if not math.isfinite(z) or not MIN_MANUAL_INSERT_Z_MM <= z <= max_z:
                self.get_logger().error('target Z must be between %.1f and %.1f mm; '
                                        'a planned floor collision still blocks motion'
                                        % (MIN_MANUAL_INSERT_Z_MM, max_z))
                return
            if self.held_item is None and not self.manual_grasp_ready:
                self.get_logger().error('pickup an item or complete grip auto successfully first')
                return
            try:
                point = self._fresh_vision_hole(cmd)
            except RuntimeError as exc:
                self.get_logger().error(str(exc))
                return
            present_rail = (self.rail.position_mm if self.rail is not None
                            else self.sim_rail_mm)
            if self.rail is not None and self.rail.moving:
                self.get_logger().error('rail is still moving; wait before insertion')
                return
            if abs(present_rail) > self.rail_tol_mm:
                try:
                    actual = np.degrees(self._current_q_rad())
                except Exception as exc:
                    self.get_logger().error('A-arm pose unavailable: %s' % exc)
                    return
                travel = np.asarray(self.rail_travel_pose_deg, dtype=float)
                error = np.abs((actual - travel + 180.0) % 360.0 - 180.0)
                if np.max(error) > 5.0:
                    self.get_logger().error(
                        '%s: rail at %.1f mm. Set a safe rail-travel pose with '
                        'aj, or return rail to 0 mm with r 0 after checking '
                        'clearance; mid/side will not move A to that pose for you.'
                        % (cmd, present_rail))
                    return
            self._trial('INSERT_' + cmd.upper(),
                        lambda: self._insert_vision(point, cmd, z, rotate_q5=rotate_q5))
            return
        if cmd == 'insert':
            self.get_logger().error('insert ITEM was removed; use mid Z_mm or side Z_mm')
            return
        if cmd == 'retreat':
            if len(parts) != 1 or not self._ready(cmd):
                return
            if self.insert_retreat_path is None:
                self.get_logger().error('no insertion retreat pending')
                return
            self._trial('INSERT_VERTICAL_RETREAT', self._retreat_after_insert)
            return
        if cmd == 'motors':
            if self._ready(cmd):
                self._trial('READING_MOTOR_STATUS', self._log_motor_status)
            return
        if cmd == 'hole':
            if len(parts) != 4 or parts[1] not in ('mid', 'side'):
                self.get_logger().error('usage: hole mid|side X_mm Y_mm')
                return
            if self.busy or self._pause_event.is_set():
                self.get_logger().error('set holes only while idle and not paused')
                return
            try:
                xy = np.asarray([float(v) for v in parts[2:]], dtype=float)
            except ValueError:
                xy = np.array([np.nan, np.nan])
            if not np.isfinite(xy).all() or not self._in_a3(xy):
                self.get_logger().error('hole XY outside A3 or not finite')
                return
            self.hole_overrides[parts[1]] = xy
            self.get_logger().info('%s hole set to %s mm' % (parts[1], xy.tolist()))
            return
        if cmd == 'setpick':
            if len(parts) != 8 or parts[1] not in ITEMS:
                self.get_logger().error('usage: setpick shaft1|...|gear2 rail_mm q1 q2 q3 q4 q5')
                return
            if self.busy or self.held_item is not None:
                self.get_logger().error('set pick coordinates before pickup while idle')
                return
            try:
                numbers = [float(v) for v in parts[2:]]
            except ValueError:
                numbers = [math.nan] * 6
            if not all(math.isfinite(v) for v in numbers) or not 0 <= numbers[0] <= RAIL_MAX_M * 1000:
                self.get_logger().error('invalid rail or joint numbers')
                return
            self.pick_overrides[parts[1]] = (numbers[0], numbers[1:])
            self.get_logger().info('%s pick override rail=%.1f pose=%s'
                                   % (parts[1], numbers[0], numbers[1:]))
            return
        if cmd == 'setz':
            if len(parts) != 3 or parts[1] not in ITEMS or self.busy:
                self.get_logger().error('usage: setz shaft1|...|gear2 Z_mm while idle')
                return
            try:
                z = float(parts[2])
            except ValueError:
                z = math.nan
            if not math.isfinite(z) or not MIN_MANUAL_INSERT_Z_MM <= z <= self.target_z_mm:
                self.get_logger().error('Z must be between %.1f and %.1f mm'
                                        % (MIN_MANUAL_INSERT_Z_MM, self.target_z_mm))
                return
            self.place_z_overrides[parts[1]] = z
            self.get_logger().info('%s insertion Z=%.1f mm' % (parts[1], z))
            return
        if cmd == 'tool':
            if len(parts) != 2 or parts[1] not in ('centric', 'off', 'suction'):
                self.get_logger().error('usage: tool centric|off|suction')
                return
            if not self._ready(cmd) or self.held_item is not None or self.suction_enabled:
                self.get_logger().error('tool change requires empty tool and suction OFF')
                return
            target = parts[1]
            if target == 'off' and self.tool == 'centric':
                self._trial('CENTRIC_TORQUE_OFF', self._detach_centric)
            elif target == 'centric' and self.tool != 'centric':
                self._trial('CENTRIC_CONNECT', self._attach_centric)
            elif target == 'suction' and self.tool != 'centric':
                self.tool = 'suction'
                self.get_logger().info('Suction tool registered; ensure physical tool installed')
            else:
                self.get_logger().error('tool unchanged; detach centric with tool off first')
            return
        if cmd in ('grip', 'grasp', 'open'):
            if not self._ready(cmd, centric=True):
                return
            mode = ('auto' if cmd == 'grasp' else
                    'open' if cmd == 'open' else
                    parts[1] if len(parts) == 2 else None)
            if mode is None or (cmd != 'grip' and len(parts) != 1):
                self.get_logger().error('usage: grip 100 | grip auto | grip open')
                return
            if mode == 'auto':
                if self.held_item is not None:
                    self.get_logger().error('manual auto grasp blocked while holding an assembly item')
                    return
                self._trial('MANUAL_AUTO_GRASP', self._manual_auto_grasp)
                return
            if mode == 'open':
                def open_action():
                    if not self._move_gripper_ratio(0.0):
                        raise RuntimeError('manual centric opening failed')
                    self.held_item = None
                    self._assembly_cycle_active = False
                    self._grasp_hold_position = None
                    self._grasp_hold_ratio = None
                    self.manual_grasp_ready = False
                self._trial('MANUAL_GRIP_OPEN', open_action)
                return
            if self.held_item is not None:
                self.get_logger().error('manual close blocked while holding an assembly item')
                return
            try:
                angle = float(mode)
            except ValueError:
                angle = math.nan
            if not math.isfinite(angle) or not 0.0 <= angle <= self.centric_full_close_angle_deg:
                self.get_logger().error('grip angle must be between 0 and %.1f degrees'
                                        % self.centric_full_close_angle_deg)
                return
            def close_action():
                if not self._move_gripper_ratio(angle / self.centric_full_close_angle_deg):
                    raise RuntimeError('manual centric closing failed')
                self.manual_grasp_ready = False
            self._trial('MANUAL_GRIP_%.0fDEG' % angle, close_action)
            return
        if cmd == 'suction':
            if len(parts) != 2 or parts[1] not in ('on', 'off'):
                self.get_logger().error('usage: suction on|off')
                return
            if not self._ready(cmd) or self.tool != 'suction':
                self.get_logger().error('suction requires suction tool and idle state')
                return
            on = parts[1] == 'on'
            def suction_action():
                self._suction(on)
                self.suction_enabled = on
            self._trial('SUCTION_' + parts[1].upper(), suction_action)
            return
        if cmd == 'assembly':
            if len(parts) == 1:
                self._start_assembly_cycle()
            else:
                self.get_logger().error('usage: assembly')
            return
        if cmd in ITEMS:
            if len(parts) not in (1, 2):
                self.get_logger().error('usage: %s [Z_mm]' % cmd)
                return
            z_override = None
            if len(parts) == 2:
                try:
                    z_override = float(parts[1])
                except ValueError:
                    z_override = math.nan
                if not math.isfinite(z_override) or not MIN_MANUAL_INSERT_Z_MM <= z_override <= self.target_z_mm:
                    self.get_logger().error('%s Z must be between %.1f and %.1f mm'
                                            % (cmd, MIN_MANUAL_INSERT_Z_MM, self.target_z_mm))
                    return
            if not self._ready(cmd, centric=True, item=cmd) or self.held_item is not None:
                self.get_logger().error('%s requires empty centric gripper' % cmd)
                return
            try:
                hole = self._item_settings(cmd)[3]
                self._fresh_vision_hole(hole)
            except RuntimeError as exc:
                self.get_logger().error(str(exc))
                return
            self._trial('PART_' + cmd.upper(),
                        lambda item=cmd, z=z_override: self._part(item, z_override=z))
            return
        if cmd in ('part', 'pickup', 'place'):
            if len(parts) != 2 or parts[1] not in ITEMS:
                self.get_logger().error('%s requires shaft1, shaft2, bearing1, bearing2, gear1 or gear2' % cmd)
                return
            item = parts[1]
            if not self._ready(cmd, centric=True, item=item):
                return
            if cmd in ('part', 'pickup') and self.held_item is not None:
                self.get_logger().error('finish placing current item before picking another')
                return
            if cmd == 'place' and self.held_item != item:
                self.get_logger().error('pickup %s first' % item)
                return
            if cmd in ('part', 'place'):
                try:
                    self._hole(self._item_settings(item)[3])
                except RuntimeError as exc:
                    self.get_logger().error(str(exc))
                    return
            action = (lambda: self._part(item) if cmd == 'part' else
                      self._pickup(item) if cmd == 'pickup' else self._place(item))
            self._trial('%s_%s' % (cmd.upper(), item.upper()), action)
            return
        if cmd in ('lidpick', 'lidplace'):
            if not self._ready(cmd) or self.tool != 'suction' or self.held_item is not None:
                self.get_logger().error('lid trial requires suction tool and empty centric grip')
                return
            if cmd == 'lidplace' and not self.suction_enabled:
                self.get_logger().error('enable suction after lidpick before lidplace')
                return
            self._trial(cmd.upper(), self._lidpick if cmd == 'lidpick' else self._lidplace)
            return
        self.get_logger().warning('unknown assembly debug command: %s; type help' % cmd)


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = AssemblyDebug()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
