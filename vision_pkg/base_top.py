#!/usr/bin/env python3
"""Baseplate upper defect inspection. Hole node exclusively owns pick XY."""

import json
import os
import sys
import time
from pathlib import Path

CUSTOM_PYTHON_PKGS = os.environ.get(
    'VISION_PYTHON_PKGS', '/media/wonseok/새 볼륨/python_pkgs')
if CUSTOM_PYTHON_PKGS and os.path.isdir(CUSTOM_PYTHON_PKGS):
    if CUSTOM_PYTHON_PKGS not in sys.path:
        sys.path.insert(0, CUSTOM_PYTHON_PKGS)

import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray, String
from cv_bridge import CvBridge

try:
    from ultralytics import YOLO
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        'ultralytics를 찾을 수 없습니다. VISION_PYTHON_PKGS를 확인하세요.'
    ) from exc

try:
    from ament_index_python.packages import get_package_share_directory
except ImportError:
    get_package_share_directory = None


WINDOW_NAME = 'BasePlate Top Inspection - Wonseok'


def find_default_model_path():
    specified = os.environ.get('BASEPLATE_MODEL_PATH', '').strip()
    if specified:
        return str(Path(specified).expanduser())
    names = ('Baseplate_top.pt', 'baseplate_top.pt', 'baseplate.pt')
    candidates = []
    if get_package_share_directory is not None:
        try:
            share = Path(get_package_share_directory('vision_pkg'))
            candidates.extend(share / 'pt' / name for name in names)
            candidates.extend(share / 'models' / name for name in names)
        except Exception:
            pass
    script_dir = Path(__file__).resolve().parent
    candidates.extend(script_dir / 'pt' / name for name in names)
    candidates.extend(script_dir / name for name in names)
    candidates.extend(script_dir.parent / 'models' / name for name in names)
    return str(next((p for p in candidates if p.is_file()), candidates[0]))


class BaseplateTopInspector(Node):
    def __init__(self):
        super().__init__('baseplate_top')
        self.declare_parameter('model_path', find_default_model_path())
        self.declare_parameter('image_topic', '/image_raw')
        self.declare_parameter('conf', 0.15)
        self.declare_parameter('iou', 0.45)
        self.declare_parameter('inspection_sec', 3.0)
        self.declare_parameter('defect_hold_sec', 2.0)

        model_path = str(self.get_parameter('model_path').value)
        if not Path(model_path).expanduser().is_file():
            raise FileNotFoundError('BasePlate YOLO 모델을 찾을 수 없습니다: ' + model_path)
        self.model = YOLO(model_path)
        self.conf = float(self.get_parameter('conf').value)
        self.iou = float(self.get_parameter('iou').value)
        self.inspection_sec = float(self.get_parameter('inspection_sec').value)
        self.defect_hold_sec = float(self.get_parameter('defect_hold_sec').value)
        if self.inspection_sec <= 0 or self.defect_hold_sec <= 0:
            raise ValueError('inspection_sec and defect_hold_sec must be positive')

        self.bridge = CvBridge()
        self.create_subscription(
            Image, str(self.get_parameter('image_topic').value),
            self.listener_callback, qos_profile_sensor_data)
        self.create_subscription(String, '/cell/product', self.on_product, 10)
        self.create_subscription(String, '/cell/ready', self.on_ready, 10)
        self.create_subscription(Float64MultiArray, '/cell/corners', self.on_corners, 10)
        self.inspection_pub = self.create_publisher(String, '/inspection_state', 10)
        self.cell_result_pub = self.create_publisher(String, '/cell/top_result', 10)
        self.stop_pub = self.create_publisher(String, '/centric_sequence/command', 10)

        self.active = False
        self.window_open = False
        self.corners_ready = False
        self.inspection_start = None
        self.ng_start = None
        self.final_judge = None
        self.final_reason = ''
        self.get_logger().info('Baseplate TOP defect-only model: %s; classes=%s'
                               % (model_path, self.model.names))

    @staticmethod
    def normalize_label(label):
        return ' '.join(str(label).lower().replace('_', ' ').replace('-', ' ').split())

    def reset_decision(self):
        self.inspection_start = None
        self.ng_start = None
        self.final_judge = None
        self.final_reason = ''

    def on_corners(self, msg):
        # Calibration and coordinate conversion remain in top.py and holes.py.
        self.corners_ready = len(msg.data) == 8
        if not self.corners_ready:
            self.reset_decision()

    def on_product(self, msg):
        enabled = msg.data.strip().lower() == 'baseplate'
        if enabled and not self.window_open:
            self.reset_decision()
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            self.window_open = True
        elif not enabled and self.window_open:
            cv2.destroyWindow(WINDOW_NAME)
            self.window_open = False
        self.active = enabled

    def on_ready(self, msg):
        if msg.data.strip().upper() == 'READY':
            self.on_product(String(data='idle'))
            self.reset_decision()

    def _publish_state(self, has_baseplate, dark_count, short_count, judge):
        msg = String()
        msg.data = json.dumps({
            'has_baseplate': has_baseplate,
            'has_dark_spot': bool(dark_count),
            'dark_spot_count': dark_count,
            'has_short_shot': bool(short_count),
            'short_shot_count': short_count,
            'judge': judge,
            'stamp': time.time(),
        }, ensure_ascii=False)
        self.inspection_pub.publish(msg)

    def _finish(self, judge, reason):
        self.final_judge = judge
        self.final_reason = reason
        msg = String()
        msg.data = json.dumps({
            'product': 'baseplate', 'judge': judge, 'stamp': time.time(),
        }, ensure_ascii=False)
        self.cell_result_pub.publish(msg)
        self.get_logger().info('BasePlate upper inspection: %s (%s); '
                               'pick XY will come from the hole node'
                               % (judge, reason))

    def listener_callback(self, data):
        if not self.active:
            return
        try:
            frame = self.bridge.imgmsg_to_cv2(data, 'bgr8')
        except Exception as exc:
            self.get_logger().error('Top image conversion failed: %s' % exc)
            return

        if not self.corners_ready:
            status, color = 'WAITING FOR A3 CALIBRATION', (0, 255, 255)
        elif self.final_judge is not None:
            # Stop running this YOLO model once judged so the hole scan can
            # use the same camera frames and processing resources promptly.
            status = self.final_judge + ' - BASEPLATE'
            if self.final_reason:
                status += ' (' + self.final_reason + ')'
            color = (0, 0, 255) if self.final_judge == 'NG' else (0, 255, 0)
        else:
            try:
                result = self.model(frame, conf=self.conf, iou=self.iou,
                                    verbose=False)[0]
            except Exception as exc:
                self.get_logger().error('BasePlate inference failed: %s' % exc)
                return
            has_baseplate = False
            dark_count = 0
            short_count = 0
            for box in result.boxes if result.boxes is not None else ():
                label = self.normalize_label(self.model.names[int(box.cls[0])])
                if label in ('baseplate', 'base plate'):
                    has_baseplate = True
                elif label in ('dark spot', 'darkspot'):
                    dark_count += 1
                elif label in ('short shot', 'shortshot'):
                    short_count += 1

            now = time.monotonic()
            if not has_baseplate:
                self.inspection_start = None
                self.ng_start = None
                judge = 'NO_OBJECT'
                status, color = 'NO BASEPLATE', (0, 255, 255)
            else:
                judge = 'NG' if dark_count or short_count else 'OK'
                if self.inspection_start is None:
                    self.inspection_start = now
                if judge == 'NG':
                    if self.ng_start is None:
                        self.ng_start = now
                else:
                    self.ng_start = None
                elapsed = now - self.inspection_start
                reason = ('DARK SPOT' if dark_count else 'SHORT SHOT') if judge == 'NG' else ''
                status = ('NG - ' + reason) if judge == 'NG' else 'OK - BASEPLATE'
                color = (0, 0, 255) if judge == 'NG' else (0, 255, 0)
                if judge == 'NG' and now - self.ng_start >= self.defect_hold_sec:
                    self._finish('NG', reason)
                elif judge == 'OK' and elapsed >= self.inspection_sec:
                    self._finish('OK', '')
            self._publish_state(has_baseplate, dark_count, short_count, judge)

        cv2.putText(frame, status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    0.85, color, 2, cv2.LINE_AA)
        cv2.imshow(WINDOW_NAME, frame)
        key = cv2.waitKey(1) & 255
        if key == ord('r'):
            self.reset_decision()
        elif key in (ord('q'), 27):
            self.stop_pub.publish(String(data='stop'))
            if rclpy.ok():
                rclpy.shutdown()

    def destroy_node(self):
        if self.window_open:
            cv2.destroyWindow(WINDOW_NAME)
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = BaseplateTopInspector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
