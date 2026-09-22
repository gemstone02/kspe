"""
vision_pkg / rail_serial_5axis.py
=================================

rail_stepmotor.ino(v2)와 ROS2 사이의 시리얼 어댑터.

Arduino 명령
------------
  z           : 현재 위치를 0 mm 원점으로 설정
  p <mm>      : 절대 위치 이동
  m <mm>      : 상대 위치 이동
  ?           : 상태 조회
  s / 0       : 부드러운 정지 / 즉시 정지
  von / voff  : 솔레노이드 흡착 ON / OFF
  v?          : 흡착 상태 조회

Arduino 출력 예
---------------
  [완료] 현재 위치를 원점(0.00 mm)으로 설정했습니다.
  원점 설정 여부 : O (설정됨)
  현재 위치      : 20.00 mm (1600 steps)
  [동작 중...] CW (시계방향) 1600 steps 이동
  [완료] 현재 위치: 20.00 mm (1600 steps)

ROS process_sequence 쪽 명령은 그대로 유지한다.
  A5> rz       -> Arduino "z"
  A5> r 50     -> Arduino "p 50"
  A5> r?       -> Arduino "?"
  A5> von      -> Arduino "von"
  A5> voff     -> Arduino "voff"
"""

import re
import threading
import time

PULSES_PER_REV = 1600
LEAD_MM = 10.0
PULSES_PER_MM = PULSES_PER_REV / LEAD_MM
STEPS_PER_M = int(PULSES_PER_MM * 1000)
M_PER_STEP = 1.0 / STEPS_PER_M

RAIL_MIN_M = 0.0
RAIL_MAX_M = 0.9


class RailSerial:
    def __init__(self, port='/dev/ttyACM0', baud=115200, logger=None):
        self.port_name = port
        self.baud = int(baud)
        self.logger = logger
        self.ser = None

        self._pulse = 0
        self._zero_set = False
        self._moving = False
        self._suction = False

        self._lock = threading.Lock()
        self._thread = None
        self._running = False

        self.last_rx = 0.0
        self.last_event = ''
        self.last_line = ''
        self._suction_warned = False

    # ------------------------------------------------------------
    # 연결
    # ------------------------------------------------------------
    def connect(self):
        import serial

        self.ser = serial.Serial(
            self.port_name,
            self.baud,
            timeout=0.05
        )

        # Arduino가 포트 open 시 리셋될 수 있으므로 대기
        time.sleep(2.0)
        self.ser.reset_input_buffer()

        self._running = True
        self._thread = threading.Thread(
            target=self._reader,
            daemon=True
        )
        self._thread.start()

        self.query()
        self._log(
            '레일 시리얼 연결: %s @ %d'
            % (self.port_name, self.baud)
        )
        return True

    def close(self):
        if self.ser is not None:
            try:
                # 종료 시 솔레노이드를 끄고 레일 펄스를 즉시 정지한다.
                self.send('voff')
                self.send('0')
                time.sleep(0.05)
            except Exception:
                pass

        self._running = False

        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

        if self.ser is not None:
            try:
                self.ser.close()
            except Exception:
                pass
            self.ser = None

        self._log('레일 시리얼 닫음')

    # ------------------------------------------------------------
    # 수신
    # ------------------------------------------------------------
    def _reader(self):
        buf = b''

        while self._running:
            try:
                data = self.ser.read(256)
            except Exception:
                break

            if not data:
                time.sleep(0.005)
                continue

            buf += data

            while b'\n' in buf:
                raw, buf = buf.split(b'\n', 1)

                # 현재 Arduino는 한글 메시지를 출력하므로 UTF-8로 읽는다.
                line = raw.decode(
                    'utf-8',
                    errors='ignore'
                ).strip()

                if line:
                    self._parse(line)

    def _parse(self, line):
        now = time.time()

        with self._lock:
            self.last_line = line

        # ========================================================
        # 1. 예전 @P/@D/@Z 프로토콜도 호환
        # ========================================================
        if line.startswith('@') and len(line) >= 2:
            tag = line[1]
            parts = line[2:].split()

            if not parts:
                return

            if tag == 'V':
                with self._lock:
                    self._suction = (parts[0] == '1')
                    self.last_rx = now
                    self.last_event = 'V'
                return

            try:
                pulse = int(parts[0])
            except ValueError:
                return

            zero = (
                len(parts) > 1
                and parts[1] == '1'
            )

            with self._lock:
                self._pulse = pulse
                self._zero_set = zero
                self.last_rx = now
                self.last_event = tag

                if tag in ('D', 'S', 'L', 'Z'):
                    self._moving = False
                elif tag == 'P':
                    self._moving = True

            return

        # ========================================================
        # 2. 현재 Arduino 한글 텍스트 출력
        # ========================================================

        # zero 명령 완료
        if (
            '현재 위치를 원점' in line
            and (
                '설정했습니다' in line
                or '[완료]' in line
            )
        ):
            with self._lock:
                self._pulse = 0
                self._zero_set = True
                self._moving = False
                self.last_rx = now
                self.last_event = 'Z'

            self._log('레일 원점 설정 확인: 0.0 mm')
            return

        # status의 원점 설정 여부
        if '원점 설정 여부' in line:
            is_set = (
                ('O' in line or '설정됨' in line)
                and '미설정' not in line
            )

            with self._lock:
                self._zero_set = bool(is_set)
                self.last_rx = now
                self.last_event = 'Q'
            return

        # 이동 시작
        if '[동작 중' in line or '[동작중' in line:
            with self._lock:
                self._moving = True
                self.last_rx = now
                self.last_event = 'P'
            return

        # 위치 출력
        # 예:
        # 현재 위치      : 20.00 mm (1600 steps)
        # [완료] 현재 위치: 20.00 mm (1600 steps)
        match = re.search(
            r'현재\s*위치\s*:\s*'
            r'([-+]?\d+(?:\.\d+)?)\s*mm'
            r'(?:\s*\(\s*([-+]?\d+)\s*steps?\s*\))?',
            line,
            flags=re.IGNORECASE
        )

        if match:
            mm = float(match.group(1))
            step_text = match.group(2)

            if step_text is None:
                pulse = int(
                    round(mm * PULSES_PER_MM)
                )
            else:
                pulse = int(step_text)

            completed = '[완료]' in line

            with self._lock:
                self._pulse = pulse
                self.last_rx = now

                if completed:
                    self._moving = False
                    self.last_event = 'D'
                else:
                    self.last_event = 'Q'

            if completed:
                self._log(
                    '레일 이동 완료: %.1f mm'
                    % mm
                )
            return

        # 정지/전원 해제 완료
        if (
            '[완료]' in line
            and (
                '정지' in line
                or '전원 해제' in line
                or '모터 전원' in line
            )
        ):
            with self._lock:
                self._moving = False
                self.last_rx = now
                self.last_event = 'S'
            return

        # 오류
        if '[오류]' in line:
            with self._lock:
                self._moving = False
                self.last_rx = now
                self.last_event = 'E'

            self._log(
                'Arduino 응답: %s'
                % line
            )

    # ------------------------------------------------------------
    # 상태
    # ------------------------------------------------------------
    def poll(self):
        return self.position_m

    @property
    def steps(self):
        with self._lock:
            return self._pulse

    @property
    def position_m(self):
        with self._lock:
            return self._pulse * M_PER_STEP

    @property
    def position_mm(self):
        with self._lock:
            return self._pulse / PULSES_PER_MM

    @property
    def suction_on(self):
        with self._lock:
            return self._suction

    @property
    def zero_set(self):
        with self._lock:
            return self._zero_set

    @property
    def moving(self):
        with self._lock:
            return self._moving

    @property
    def connected(self):
        return (
            self.ser is not None
            and self._running
        )

    def stale(self, sec=3.0):
        with self._lock:
            if self.last_rx == 0.0:
                return True

            return (
                time.time() - self.last_rx
            ) > float(sec)

    # ------------------------------------------------------------
    # 명령 송신
    # ------------------------------------------------------------
    def send(self, cmd):
        if self.ser is None:
            return False

        try:
            text = str(cmd).strip()

            self.ser.write(
                (text + '\n').encode('ascii')
            )
            self.ser.flush()
            return True

        except Exception as e:
            self._log(
                '레일 명령 전송 실패: %s'
                % e
            )
            return False

    def zero(self, wait=False, timeout=1.5):
        """
        현재 위치를 0 mm로 설정.
        Arduino 명령: z
        """
        sent_at = time.time()

        with self._lock:
            self.last_event = ''

        if not self.send('z'):
            return False

        if not wait:
            return True

        deadline = (
            time.time() + float(timeout)
        )

        while time.time() < deadline:
            with self._lock:
                confirmed = (
                    self._zero_set
                    and self.last_event == 'Z'
                    and self.last_rx >= sent_at
                )

            if confirmed:
                return True

            time.sleep(0.01)

        return False

    def move_abs_mm(self, mm):
        """
        절대 위치(mm) 이동.
        Arduino 명령: p <mm>
        """
        mm = float(mm)

        if not (
            RAIL_MIN_M * 1000.0
            <= mm
            <= RAIL_MAX_M * 1000.0
        ):
            self._log(
                '절대이동 거부: %.1f mm가 레일 범위 밖입니다.'
                % mm
            )
            return False

        with self._lock:
            if not self._zero_set:
                self._log(
                    '절대이동 거부: 원점이 설정되지 않았습니다.'
                )
                return False

            self._moving = True
            self.last_event = 'P'

        ok = self.send(
            'p %.3f' % mm
        )

        if not ok:
            with self._lock:
                self._moving = False

        return ok

    def move_abs_m(self, m):
        return self.move_abs_mm(
            float(m) * 1000.0
        )

    def move_rel_mm(self, mm):
        """
        상대 위치(mm) 이동.
        Arduino 명령: m <mm>
        """
        delta = float(mm)

        with self._lock:
            current_mm = (
                self._pulse / PULSES_PER_MM
            )
            zero_set = self._zero_set

        if not zero_set:
            self._log(
                '상대이동 거부: 원점이 설정되지 않았습니다.'
            )
            return False

        target_mm = current_mm + delta

        if not (
            RAIL_MIN_M * 1000.0
            <= target_mm
            <= RAIL_MAX_M * 1000.0
        ):
            self._log(
                '상대이동 거부: 예상 목표 %.1f mm가 레일 범위 밖입니다.'
                % target_mm
            )
            return False

        with self._lock:
            self._moving = True
            self.last_event = 'P'

        ok = self.send(
            'm %.3f' % delta
        )

        if not ok:
            with self._lock:
                self._moving = False

        return ok

    def move_rel_m(self, m):
        return self.move_rel_mm(
            float(m) * 1000.0
        )

    def to_end(self):
        return self.move_abs_m(
            RAIL_MAX_M
        )

    def to_origin(self):
        return self.move_abs_m(
            RAIL_MIN_M
        )

    def smooth_stop(self):
        with self._lock:
            self._moving = False
        return self.send('s')

    def estop(self):
        with self._lock:
            self._moving = False
        return self.send('0')

    def stop(self):
        return self.estop()

    def enable(self):
        # rail_stepmotor.ino(v2)는 별도 enable 명령을 사용하지 않는다.
        return self.connected

    def query(self):
        return self.send('?')

    def suction(self, on, wait=False, timeout=1.0):
        """솔레노이드 흡착을 켜거나 끄고, 선택적으로 @V 응답을 기다린다."""
        desired = bool(on)
        sent_at = time.time()
        with self._lock:
            self.last_event = ''

        if not self.send('von' if desired else 'voff'):
            return False
        if not wait:
            return True

        deadline = time.time() + float(timeout)
        while time.time() < deadline:
            with self._lock:
                confirmed = (
                    self.last_event == 'V'
                    and self.last_rx >= sent_at
                    and self._suction == desired
                )
            if confirmed:
                return True
            time.sleep(0.01)
        self._log('흡착 %s 응답 시간초과' % ('ON' if desired else 'OFF'))
        return False

    def suction_query(self):
        return self.send('v?')

    # 구버전 호환
    def cw(self):
        return self.to_end()

    def ccw(self):
        return self.to_origin()

    def _log(self, msg):
        if self.logger is not None:
            try:
                self.logger.info(msg)
                return
            except Exception:
                pass

        print('[RAIL] ' + msg)


# setup.py의 console_scripts 항목 호환용
def main(args=None):
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import Float64, String

    class RailNode(Node):
        def __init__(self):
            super().__init__('rail_serial_5axis')

            self.declare_parameter(
                'port',
                '/dev/ttyACM0'
            )
            self.declare_parameter(
                'baud',
                115200
            )
            self.declare_parameter(
                'auto_zero',
                False
            )

            self.rail = RailSerial(
                port=str(
                    self.get_parameter(
                        'port'
                    ).value
                ),
                baud=int(
                    self.get_parameter(
                        'baud'
                    ).value
                ),
                logger=self.get_logger()
            )

            self.rail.connect()

            if bool(
                self.get_parameter(
                    'auto_zero'
                ).value
            ):
                ok = self.rail.zero(
                    wait=True,
                    timeout=2.0
                )

                if ok:
                    self.get_logger().info(
                        '레일 자동 원점 설정 완료'
                    )
                else:
                    self.get_logger().error(
                        '레일 자동 원점 설정 실패'
                    )

            self.pub = self.create_publisher(
                Float64,
                '/rail/position_mm',
                10
            )

            self.sub = self.create_subscription(
                String,
                '/rail/command',
                self._on_cmd,
                10
            )

            self.create_timer(
                0.1,
                self._tick
            )

        def _on_cmd(self, msg):
            parts = msg.data.strip().split()

            if not parts:
                return

            cmd = parts[0].lower()

            try:
                if cmd in ('zero', 'z'):
                    self.rail.zero()

                elif cmd in ('status', '?'):
                    self.rail.query()

                elif (
                    cmd in ('m', 'abs')
                    and len(parts) == 2
                ):
                    self.rail.move_abs_mm(
                        float(parts[1])
                    )

                elif (
                    cmd in ('r', 'rel')
                    and len(parts) == 2
                ):
                    self.rail.move_rel_mm(
                        float(parts[1])
                    )

                elif cmd == 'stop':
                    self.rail.estop()

                elif cmd == 'en':
                    self.rail.enable()

                else:
                    self.get_logger().warn(
                        '알 수 없는 레일 명령: %s'
                        % msg.data
                    )

            except Exception as e:
                self.get_logger().error(
                    '레일 명령 처리 실패: %s'
                    % e
                )

        def _tick(self):
            msg = Float64()
            msg.data = float(
                self.rail.position_mm
            )
            self.pub.publish(msg)

        def destroy_node(self):
            try:
                self.rail.close()
            except Exception:
                pass

            super().destroy_node()

    rclpy.init(args=args)
    node = RailNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
