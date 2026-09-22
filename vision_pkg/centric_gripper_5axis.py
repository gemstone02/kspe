#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""3-jaw centric gripper driver and optional ROS 2 standalone node.

The :class:`CentricGripper` class is imported by centric_sequence_5axis.py.
This module can also be run independently through ROS 2 for gripper tests.

Important: with ``capture_open_on_start:=true`` (the default), the gripper
must be physically fully open before this program starts.  The present motor
position is then stored as the 0% OPEN reference, exactly like the original
centric_gripper_test.py.
"""

import json
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

from dynamixel_sdk import COMM_SUCCESS, PacketHandler, PortHandler


# XL430 control table
ADDR_OPERATING_MODE = 11
ADDR_TORQUE_ENABLE = 64
ADDR_HARDWARE_ERROR_STATUS = 70
ADDR_GOAL_PWM = 100
ADDR_PROFILE_ACCELERATION = 108
ADDR_PROFILE_VELOCITY = 112
ADDR_GOAL_POSITION = 116
ADDR_MOVING = 122
ADDR_PRESENT_PWM = 124
ADDR_PRESENT_LOAD = 126
ADDR_PRESENT_VELOCITY = 128
ADDR_PRESENT_POSITION = 132
ADDR_PRESENT_INPUT_VOLTAGE = 144
ADDR_PRESENT_TEMPERATURE = 146

TORQUE_DISABLE = 0
TORQUE_ENABLE = 1
EXTENDED_POSITION_MODE = 4
PROTOCOL_VERSION = 2.0


@dataclass(frozen=True)
class GraspResult:
    success: bool
    reason: str
    position: Optional[int] = None
    load_percent: Optional[float] = None


class CentricGripper:
    """Dynamixel ID 8 controller for the 18T/22T centric gripper."""

    def __init__(
        self,
        port="/dev/ttyUSB1",
        baud=2_000_000,
        dxl_id=8,
        logger=None,
        close_direction=+1,
        sun_teeth=18,
        planet_teeth=22,
        planet_max_angle_deg=170.0,
        safety_margin_pulse=31,
        normal_profile_velocity=44,
        auto_profile_velocity=22,
        profile_acceleration=5,
        goal_pwm_limit=250,
        auto_load_threshold_percent=20.0,
        auto_load_count_limit=8,
        auto_sample_sec=0.02,
        hold_extra_pulse=0,
        min_grasp_travel_pulse=100,
        auto_grasp_timeout_sec=10.0,
        emergency_load_percent=35.0,
        temperature_stop_c=60,
        position_tolerance_pulse=15,
        move_timeout_sec=10.0,
        communication_retry_count=5,
        communication_retry_delay_sec=0.02,
        temperature_sample_sec=0.5,
    ):
        self.port_name = str(port)
        self.baud = int(baud)
        self.dxl_id = int(dxl_id)
        self.logger = logger
        self.close_direction = 1 if int(close_direction) >= 0 else -1

        motor_max_angle = (
            float(planet_max_angle_deg) * float(planet_teeth) / float(sun_teeth)
        )
        self.max_move_pulse = int(round(motor_max_angle / 360.0 * 4096.0))
        self.safe_move_pulse = self.max_move_pulse - int(safety_margin_pulse)
        if self.safe_move_pulse <= 0:
            raise ValueError("safe_move_pulse must be positive")

        self.normal_profile_velocity = int(normal_profile_velocity)
        self.auto_profile_velocity = int(auto_profile_velocity)
        self.profile_acceleration = int(profile_acceleration)
        self.goal_pwm_limit = int(goal_pwm_limit)
        self.auto_load_threshold = float(auto_load_threshold_percent)
        self.auto_load_count_limit = int(auto_load_count_limit)
        self.auto_sample_sec = float(auto_sample_sec)
        self.hold_extra_pulse = int(hold_extra_pulse)
        self.min_grasp_travel = int(min_grasp_travel_pulse)
        self.auto_grasp_timeout = float(auto_grasp_timeout_sec)
        self.emergency_load = float(emergency_load_percent)
        self.temperature_stop_c = int(temperature_stop_c)
        self.position_tolerance = int(position_tolerance_pulse)
        self.move_timeout = float(move_timeout_sec)
        self.communication_retry_count = max(1, int(communication_retry_count))
        self.communication_retry_delay = max(
            0.0, float(communication_retry_delay_sec)
        )
        self.temperature_sample_sec = max(0.1, float(temperature_sample_sec))

        self.port_handler = None
        self.packet_handler = None
        self.open_position = None
        self.safe_close_position = None
        self.connected = False
        self._owns_port = False
        self._lock = threading.RLock()
        self._last_position = None

    # ---------------------------------------------------------- communication
    @staticmethod
    def _signed32(value):
        return value - 0x100000000 if value >= 0x80000000 else value

    @staticmethod
    def _signed16(value):
        return value - 0x10000 if value >= 0x8000 else value

    def _check(self, result, error, label):
        if result != COMM_SUCCESS:
            raise RuntimeError(
                "%s communication error: %s"
                % (label, self.packet_handler.getTxRxResult(result))
            )
        if error:
            raise RuntimeError(
                "%s Dynamixel error: %s"
                % (label, self.packet_handler.getRxPacketError(error))
            )

    def _txrx_with_retry(
        self,
        operation,
        label,
        read=False,
        transform=None,
        validator=None,
        valid_description="",
    ):
        """Run one SDK transaction with retries and reject corrupt values."""
        last_result = None
        last_error = 0
        last_exception = None
        last_value = None
        for attempt in range(1, self.communication_retry_count + 1):
            try:
                with self._lock:
                    response = operation()
            except Exception as exc:
                # The Dynamixel SDK can itself raise (for example an internal
                # list-index error) when a partial/corrupt packet is received.
                last_exception = exc
                if attempt < self.communication_retry_count:
                    time.sleep(self.communication_retry_delay)
                    continue
                raise RuntimeError(
                    "%s communication exception after %d attempts: %s"
                    % (label, self.communication_retry_count, exc)
                ) from exc
            if read:
                value, result, error = response
            else:
                result, error = response
                value = None
            last_result, last_error = result, error
            if result == COMM_SUCCESS:
                self._check(result, error, label)
                if read:
                    candidate = transform(value) if transform is not None else int(value)
                    last_value = candidate
                    if validator is not None and not validator(candidate):
                        if attempt < self.communication_retry_count:
                            self._log(
                                "%s rejected corrupt value %s; retry %d/%d"
                                % (
                                    label,
                                    candidate,
                                    attempt + 1,
                                    self.communication_retry_count,
                                ),
                                warning=True,
                            )
                            time.sleep(self.communication_retry_delay)
                            continue
                        raise RuntimeError(
                            "%s invalid value %s%s"
                            % (
                                label,
                                candidate,
                                ("; expected " + valid_description)
                                if valid_description
                                else "",
                            )
                        )
                if attempt > 1:
                    self._log(
                        "%s communication recovered on retry %d/%d"
                        % (label, attempt, self.communication_retry_count),
                        warning=True,
                    )
                return candidate if read else value
            if attempt < self.communication_retry_count:
                time.sleep(self.communication_retry_delay)
        self._check(last_result, last_error, label)
        raise RuntimeError(
            "%s communication failed (last exception=%s, last value=%s)"
            % (label, last_exception, last_value)
        )

    def _write1(self, address, value, label):
        self._txrx_with_retry(
            lambda: self.packet_handler.write1ByteTxRx(
                self.port_handler, self.dxl_id, int(address), int(value)
            ),
            label,
        )

    def _write2(self, address, value, label):
        self._txrx_with_retry(
            lambda: self.packet_handler.write2ByteTxRx(
                self.port_handler, self.dxl_id, int(address), int(value) & 0xFFFF
            ),
            label,
        )

    def _write4(self, address, value, label):
        self._txrx_with_retry(
            lambda: self.packet_handler.write4ByteTxRx(
                self.port_handler, self.dxl_id, int(address), int(value) & 0xFFFFFFFF
            ),
            label,
        )

    def _read1(
        self, address, label, validator=None, valid_description=""
    ):
        value = self._txrx_with_retry(
            lambda: self.packet_handler.read1ByteTxRx(
                self.port_handler, self.dxl_id, int(address)
            ),
            label,
            read=True,
            transform=int,
            validator=validator,
            valid_description=valid_description,
        )
        return int(value)

    def _read2(
        self,
        address,
        label,
        signed=False,
        validator=None,
        valid_description="",
    ):
        value = self._txrx_with_retry(
            lambda: self.packet_handler.read2ByteTxRx(
                self.port_handler, self.dxl_id, int(address)
            ),
            label,
            read=True,
            transform=self._signed16 if signed else int,
            validator=validator,
            valid_description=valid_description,
        )
        return int(value)

    def _read4(
        self,
        address,
        label,
        signed=False,
        validator=None,
        valid_description="",
    ):
        value = self._txrx_with_retry(
            lambda: self.packet_handler.read4ByteTxRx(
                self.port_handler, self.dxl_id, int(address)
            ),
            label,
            read=True,
            transform=self._signed32 if signed else int,
            validator=validator,
            valid_description=valid_description,
        )
        return int(value)

    # --------------------------------------------------------------- lifecycle
    def connect(self, capture_open_on_start=True, configured_open_position=506):
        with self._lock:
            if self.connected:
                return
            self.port_handler = PortHandler(self.port_name)
            self.packet_handler = PacketHandler(PROTOCOL_VERSION)
            if not self.port_handler.openPort():
                raise RuntimeError("centric gripper port open failed: " + self.port_name)
            self._owns_port = True
            try:
                if not self.port_handler.setBaudRate(self.baud):
                    raise RuntimeError("centric gripper baud-rate setting failed")
                self._configure_motor(
                    capture_open_on_start=capture_open_on_start,
                    configured_open_position=configured_open_position,
                    connection_label=self.port_name,
                )
            except Exception:
                try:
                    self.port_handler.closePort()
                finally:
                    self.port_handler = None
                    self.packet_handler = None
                    self._owns_port = False
                raise

    def connect_shared(
        self,
        port_handler,
        packet_handler,
        bus_lock,
        capture_open_on_start=True,
        configured_open_position=506,
    ):
        """Attach ID 8 to the U2D2 connection already opened by A-arm.

        The caller retains ownership of the physical port.  Every ID 8 Tx/Rx
        uses the same re-entrant lock as IDs 1~7, preventing overlapping SDK
        packets on the shared TTL bus.
        """
        if port_handler is None or packet_handler is None:
            raise ValueError("shared Dynamixel handlers are required")
        if bus_lock is None:
            raise ValueError("shared Dynamixel bus lock is required")
        self._lock = bus_lock
        with self._lock:
            if self.connected:
                return
            self.port_handler = port_handler
            self.packet_handler = packet_handler
            self._owns_port = False
            try:
                self._configure_motor(
                    capture_open_on_start=capture_open_on_start,
                    configured_open_position=configured_open_position,
                    connection_label="shared A-arm U2D2 (%s)" % self.port_name,
                )
            except Exception:
                self.port_handler = None
                self.packet_handler = None
                raise

    def _configure_motor(
        self,
        capture_open_on_start,
        configured_open_position,
        connection_label,
    ):
        model, result, error = self.packet_handler.ping(
            self.port_handler, self.dxl_id
        )
        self._check(result, error, "ping")

        self._write1(ADDR_TORQUE_ENABLE, TORQUE_DISABLE, "initial torque off")
        self._write1(
            ADDR_OPERATING_MODE,
            EXTENDED_POSITION_MODE,
            "extended position mode",
        )
        present = self.read_position()
        self.open_position = (
            present if bool(capture_open_on_start) else int(configured_open_position)
        )
        self.safe_close_position = (
            self.open_position + self.close_direction * self.safe_move_pulse
        )

        # Prevent a jump when torque is enabled.
        self._write4(ADDR_GOAL_POSITION, present, "initial goal position")
        self._write1(ADDR_TORQUE_ENABLE, TORQUE_ENABLE, "torque on")
        self.set_profile_velocity(self.normal_profile_velocity)
        self._write4(
            ADDR_PROFILE_ACCELERATION,
            self.profile_acceleration,
            "profile acceleration",
        )
        self._write2(ADDR_GOAL_PWM, self.goal_pwm_limit, "goal pwm")
        self.connected = True
        self._log(
            "connected: %s, id=%d, model=%d, OPEN=%d, SAFE_CLOSE=%d"
            % (
                connection_label,
                self.dxl_id,
                model,
                self.open_position,
                self.safe_close_position,
            )
        )

    def close(self, torque_off=True):
        with self._lock:
            if self.port_handler is None:
                return
            try:
                if torque_off:
                    self._write1(ADDR_TORQUE_ENABLE, TORQUE_DISABLE, "torque off")
            except Exception as exc:
                self._log("shutdown warning: %s" % exc, warning=True)
            finally:
                if self._owns_port:
                    self.port_handler.closePort()
                self.port_handler = None
                self.packet_handler = None
                self.connected = False
                self._owns_port = False

    # ------------------------------------------------------------------ status
    def read_position(self):
        position = self._read4(
            ADDR_PRESENT_POSITION,
            "present position",
            signed=True,
            validator=self._position_is_plausible,
            valid_description="a plausible XL430 extended position",
        )
        self._last_position = position
        return position

    def read_load_percent(self):
        raw = self._read2(
            ADDR_PRESENT_LOAD,
            "present load",
            signed=True,
            validator=lambda value: -1000 <= value <= 1000,
            valid_description="-1000..1000 raw (-100..100%)",
        )
        return raw / 10.0

    def read_temperature_c(self):
        return self._read1(
            ADDR_PRESENT_TEMPERATURE,
            "present temperature",
            validator=lambda value: 0 <= value <= 100,
            valid_description="0..100 C",
        )

    def _position_is_plausible(self, position):
        # XL430 extended-position data outside this range is a corrupt packet.
        if not -1048575 <= int(position) <= 1048575:
            return False
        if self.open_position is None or self.safe_close_position is None:
            return True
        margin = 4096
        low = min(self.open_position, self.safe_close_position) - margin
        high = max(self.open_position, self.safe_close_position) + margin
        return low <= int(position) <= high

    def read_status(self):
        position = self.read_position()
        load = self.read_load_percent()
        return {
            "position": position,
            "close_percent": self.close_percent(position),
            "load_percent": load,
            "velocity_raw": self._read4(
                ADDR_PRESENT_VELOCITY,
                "present velocity",
                signed=True,
                validator=lambda value: -32767 <= value <= 32767,
                valid_description="-32767..32767 raw",
            ),
            "pwm_percent": self._read2(
                ADDR_PRESENT_PWM,
                "present pwm",
                signed=True,
                validator=lambda value: -885 <= value <= 885,
                valid_description="-885..885 raw",
            ) / 8.85,
            "voltage_v": self._read2(
                ADDR_PRESENT_INPUT_VOLTAGE,
                "present voltage",
                validator=lambda value: 0 <= value <= 300,
                valid_description="0..30.0 V",
            ) / 10.0,
            "temperature_c": self.read_temperature_c(),
            "moving": bool(
                self._read1(
                    ADDR_MOVING,
                    "moving",
                    validator=lambda value: value in (0, 1),
                    valid_description="0 or 1",
                )
            ),
            "hardware_error": self._read1(
                ADDR_HARDWARE_ERROR_STATUS, "hardware error"
            ),
        }

    def close_percent(self, position=None):
        if self.open_position is None:
            return None
        if position is None:
            position = self.read_position()
        travelled = (int(position) - self.open_position) * self.close_direction
        return 100.0 * travelled / float(self.safe_move_pulse)

    # ---------------------------------------------------------------- movement
    def clamp_position(self, position):
        if self.open_position is None or self.safe_close_position is None:
            raise RuntimeError("centric gripper OPEN reference is not configured")
        low = min(self.open_position, self.safe_close_position)
        high = max(self.open_position, self.safe_close_position)
        return max(low, min(int(position), high))

    def set_profile_velocity(self, value):
        self._write4(ADDR_PROFILE_VELOCITY, int(value), "profile velocity")

    def set_goal_position(self, position):
        goal = self.clamp_position(position)
        self._write4(ADDR_GOAL_POSITION, goal, "goal position")
        return goal

    def ratio_position(self, close_ratio):
        """Return the clamped motor position for a 0.0~1.0 close ratio."""
        ratio = max(0.0, min(float(close_ratio), 1.0))
        return self.clamp_position(
            self.open_position
            + self.close_direction * int(round(self.safe_move_pulse * ratio))
        )

    def command_ratio(self, close_ratio, profile_velocity=None):
        """Start a ratio move without waiting for arrival.

        This is used when an A-arm joint move and gripper pre-close must begin
        together.  Call :meth:`move_ratio` when a blocking move is required.
        """
        velocity = (
            self.normal_profile_velocity
            if profile_velocity is None
            else int(profile_velocity)
        )
        self.set_profile_velocity(velocity)
        return self.set_goal_position(self.ratio_position(close_ratio))

    def hold_current_position(self):
        position = self.read_position()
        self.set_goal_position(position)
        return position

    def emergency_stop(self, reason="requested"):
        try:
            position = self.hold_current_position()
        except Exception as exc:
            if self._last_position is None:
                self._log(
                    "EMERGENCY STOP could not read ID %d position: %s"
                    % (self.dxl_id, exc),
                    warning=True,
                )
                return None
            position = int(self._last_position)
            try:
                self.set_goal_position(position)
            except Exception as hold_exc:
                self._log(
                    "EMERGENCY STOP fallback hold failed at %d: %s"
                    % (position, hold_exc),
                    warning=True,
                )
                return None
        self._log(
            "EMERGENCY STOP: %s; holding position %d" % (reason, position),
            warning=True,
        )
        return position

    def _wait_for_position(
        self,
        goal,
        timeout=None,
        stop_requested: Optional[Callable[[], bool]] = None,
    ):
        deadline = time.monotonic() + (
            self.move_timeout if timeout is None else float(timeout)
        )
        temperature = None
        next_temperature_read = 0.0
        while time.monotonic() < deadline:
            if stop_requested is not None and stop_requested():
                self.emergency_stop("external stop")
                return False
            position = self.read_position()
            load = self.read_load_percent()
            now = time.monotonic()
            if now >= next_temperature_read:
                temperature = self.read_temperature_c()
                next_temperature_read = now + self.temperature_sample_sec
            if abs(load) >= self.emergency_load:
                self.emergency_stop("overload %.1f%%" % load)
                return False
            if temperature is not None and temperature >= self.temperature_stop_c:
                self.emergency_stop("temperature %d C" % temperature)
                return False
            if abs(goal - position) <= self.position_tolerance:
                return True
            time.sleep(0.03)
        self.hold_current_position()
        self._log("position timeout; holding current position", warning=True)
        return False

    def open(self, stop_requested=None):
        self.set_profile_velocity(self.normal_profile_velocity)
        goal = self.set_goal_position(self.open_position)
        return self._wait_for_position(goal, stop_requested=stop_requested)

    def move_ratio(self, close_ratio, stop_requested=None):
        goal = self.command_ratio(close_ratio)
        return self._wait_for_position(goal, stop_requested=stop_requested)

    def auto_grasp(
        self,
        stop_requested: Optional[Callable[[], bool]] = None,
        min_grasp_travel_pulse: Optional[int] = None,
    ) -> GraspResult:
        start_position = self.read_position()
        min_travel = (
            self.min_grasp_travel
            if min_grasp_travel_pulse is None
            else max(0, int(min_grasp_travel_pulse))
        )
        self.set_profile_velocity(self.auto_profile_velocity)
        self.set_goal_position(self.safe_close_position)
        counter = 0
        started = time.monotonic()
        temperature = None
        next_temperature_read = 0.0

        self._log(
            "AUTO GRASP: threshold=%.1f%% x %d, emergency=%.1f%%, timeout=%.1fs"
            % (
                self.auto_load_threshold,
                self.auto_load_count_limit,
                self.emergency_load,
                self.auto_grasp_timeout,
            )
        )
        while True:
            if stop_requested is not None and stop_requested():
                position = self.hold_current_position()
                return GraspResult(False, "external stop", position)

            position = self.read_position()
            load = self.read_load_percent()
            now = time.monotonic()
            if now >= next_temperature_read:
                temperature = self.read_temperature_c()
                next_temperature_read = now + self.temperature_sample_sec
            abs_load = abs(load)
            travelled = abs(position - start_position)
            elapsed = now - started

            if abs_load >= self.emergency_load:
                self.emergency_stop("overload %.1f%%" % load)
                return GraspResult(False, "emergency overload", position, load)
            if temperature is not None and temperature >= self.temperature_stop_c:
                self.emergency_stop("temperature %d C" % temperature)
                return GraspResult(False, "overtemperature", position, load)
            if elapsed >= self.auto_grasp_timeout:
                self.hold_current_position()
                return GraspResult(False, "timeout", position, load)

            if travelled >= min_travel:
                counter = counter + 1 if abs_load >= self.auto_load_threshold else 0

            if counter >= self.auto_load_count_limit:
                hold = self.clamp_position(
                    position + self.close_direction * self.hold_extra_pulse
                )
                self.set_goal_position(hold)
                self._log(
                    "GRASP SUCCESS: position=%d, load=%.1f%%, hold=%d"
                    % (position, load, hold)
                )
                return GraspResult(True, "contact detected", hold, load)

            close_progress = (
                position - self.open_position
            ) * self.close_direction
            if close_progress >= self.safe_move_pulse - self.position_tolerance:
                self.hold_current_position()
                return GraspResult(False, "no object", position, load)

            time.sleep(self.auto_sample_sec)

    def _log(self, text, warning=False):
        if self.logger is not None:
            # Keep INFO and WARNING on separate call sites.  rclpy identifies
            # a logging call by source location and rejects severity changes
            # from the same line ("Logger severity cannot be changed").
            if warning:
                self.logger.warning(str(text))
            else:
                self.logger.info(str(text))
        else:
            print("[CENTRIC] " + str(text))


def main(args=None):
    """Standalone ROS 2 wrapper for manual gripper tests."""
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String

    class CentricGripperNode(Node):
        def __init__(self):
            super().__init__("centric_gripper_5axis")
            d = self.declare_parameter
            d("port", "/dev/ttyUSB1")
            d("baud", 2_000_000)
            d("dxl_id", 8)
            d("capture_open_on_start", True)
            d("configured_open_position", 506)
            d("planet_max_angle_deg", 170.0)
            d("auto_load_threshold_percent", 20.0)
            d("auto_grasp_timeout_sec", 10.0)
            d("status_hz", 2.0)

            self.gripper = CentricGripper(
                port=str(self.get_parameter("port").value),
                baud=int(self.get_parameter("baud").value),
                dxl_id=int(self.get_parameter("dxl_id").value),
                logger=self.get_logger(),
                planet_max_angle_deg=float(
                    self.get_parameter("planet_max_angle_deg").value
                ),
                auto_load_threshold_percent=float(
                    self.get_parameter("auto_load_threshold_percent").value
                ),
                auto_grasp_timeout_sec=float(
                    self.get_parameter("auto_grasp_timeout_sec").value
                ),
            )
            self.gripper.connect(
                capture_open_on_start=bool(
                    self.get_parameter("capture_open_on_start").value
                ),
                configured_open_position=int(
                    self.get_parameter("configured_open_position").value
                ),
            )
            self.get_logger().warning(
                "현재 위치가 OPEN 기준입니다. 실행 전에 완전 개방 상태였는지 확인하세요."
            )
            self.publisher = self.create_publisher(
                String, "/centric_gripper/state", 10
            )
            self.create_subscription(
                String, "/centric_gripper/command", self._on_command, 10
            )
            hz = max(0.2, float(self.get_parameter("status_hz").value))
            self.create_timer(1.0 / hz, self._publish_status)
            self._busy_lock = threading.Lock()
            self._stop_event = threading.Event()

        def _on_command(self, msg):
            parts = msg.data.strip().lower().split()
            if not parts:
                return
            if parts[0] in ("stop", "s"):
                self._stop_event.set()
                self.gripper.emergency_stop("ROS command")
                return
            if not self._busy_lock.acquire(blocking=False):
                self.get_logger().warning("gripper is busy")
                return
            threading.Thread(
                target=self._run_command, args=(parts,), daemon=True
            ).start()

        def _run_command(self, parts):
            try:
                self._stop_event.clear()
                command = parts[0]
                if command in ("open", "o"):
                    self.gripper.open(self._stop_event.is_set)
                elif command in ("grasp", "g"):
                    result = self.gripper.auto_grasp(self._stop_event.is_set)
                    self.get_logger().info("grasp result: %s" % result)
                elif command in ("close", "ratio") and len(parts) == 2:
                    percent = float(parts[1])
                    if percent > 1.0:
                        percent /= 100.0
                    self.gripper.move_ratio(percent, self._stop_event.is_set)
                elif command == "status":
                    self.get_logger().info(str(self.gripper.read_status()))
                else:
                    self.get_logger().warning(
                        "commands: open | grasp | close <0..100> | status | stop"
                    )
            except Exception as exc:
                self.get_logger().error("centric command failed: %s" % exc)
            finally:
                self._busy_lock.release()

        def _publish_status(self):
            try:
                msg = String()
                msg.data = json.dumps(self.gripper.read_status(), ensure_ascii=False)
                self.publisher.publish(msg)
            except Exception as exc:
                self.get_logger().error("centric status read failed: %s" % exc)

        def destroy_node(self):
            self._stop_event.set()
            self.gripper.close(torque_off=True)
            super().destroy_node()

    rclpy.init(args=args)
    node = None
    try:
        node = CentricGripperNode()
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

