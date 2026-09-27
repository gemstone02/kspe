"""One B camera; Baseplate loads a separate model for each view."""
import json
import os
import sys
import time

import numpy as np

EXTRA_PYTHON = os.environ.get('VISION_PYTHON_PKGS',
                             '/media/wonseok/새 볼륨/python_pkgs')
if os.path.isdir(EXTRA_PYTHON) and EXTRA_PYTHON not in sys.path:
    sys.path.insert(0, EXTRA_PYTHON)

import cv2
import rclpy
from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import String
from ultralytics import YOLO


def default_model(name):
    try:
        share = get_package_share_directory('vision_pkg')
    except Exception:
        share = os.path.dirname(__file__)
    variants = {'Carpart(2).pt': ('Carpart(2).pt', 'Carpart_side.pt', 'Carpart.pt')}
    names = variants.get(name, (name,))
    candidates = [os.path.join(root, candidate) for root in
                  (os.path.join(share, 'pt'), os.path.join(share, 'models'),
                   os.path.join(os.path.dirname(__file__), 'pt'),
                   os.path.dirname(__file__)) for candidate in names]
    return next((p for p in candidates if os.path.isfile(p)), candidates[0])


def norm(name):
    return ''.join(c for c in str(name).lower() if c.isalnum())


class Side(Node):
    def __init__(self):
        super().__init__('side')
        for key, value in (
            ('image_topic', '/b_camera/image_raw'),
            ('car_model', default_model('Carpart(2).pt')),
            ('bottom_model', default_model('bottom.pt')),
            ('back_model', default_model('side.pt')),
            ('flash_model', default_model('flash.pt')),
            ('conf', .25), ('base_conf', .15),
            ('duration_sec', 3.0), ('defect_hold_sec', 2.0),
        ):
            self.declare_parameter(key, value)
        g = lambda key: self.get_parameter(key).value
        self.paths = {name: os.path.expanduser(str(g(key))) for name, key in
                      (('carpart', 'car_model'),
                       ('bottom', 'bottom_model'),
                       ('back', 'back_model'),
                       ('flash', 'flash_model'))}
        self.models = {}
        self.bridge = CvBridge()
        self.conf = float(g('conf'))
        self.base_conf = float(g('base_conf'))
        self.duration = float(g('duration_sec'))
        self.hold = float(g('defect_hold_sec'))
        self.product = None
        self.session = None
        self.defect_since = None
        self.max_hold = 0.0
        self.frames = 0
        self.last_result = None
        self.last_view = ''
        self.display_key = None
        self.window_name = 'B Arm'
        self.seen_car_part = False
        self.seen_dark_spot = False
        self.max_dark_spot_count = 0
        self.result_pub = self.create_publisher(String, '/cell/side_result', 10)
        self.car_state_pub = self.create_publisher(String, '/b_inspection_state', 10)
        self.car_result_pub = self.create_publisher(String, '/b_inspection_result', 10)
        self.base_state_pub = self.create_publisher(String, '/inspection_state_side', 10)
        self.base_result_pub = self.create_publisher(String, '/baseplate_side/view_result', 10)
        self.stop_pub = self.create_publisher(String, '/centric_sequence/command', 10)
        self.create_subscription(String, '/cell/product', self.on_product, 10)
        self.create_subscription(String, '/cell/side_command', self.on_command, 10)
        self.create_subscription(Image, str(g('image_topic')), self.on_image,
                                 qos_profile_sensor_data)
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

    def on_product(self, msg):
        product = msg.data.strip().lower()
        if product not in ('carpart', 'baseplate'):
            return
        self.product = product
        wanted_window = ('B Arm Carpart Inspection' if product == 'carpart'
                         else 'BasePlate Side Inspection')
        if self.window_name != wanted_window:
            cv2.destroyWindow(self.window_name)
            self.window_name = wanted_window
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        self.last_result = None
        self.last_view = ''
        self.display_key = 'carpart' if product == 'carpart' else 'bottom'
        # Preload every model for the product, outside the 3-second inspection.
        for key in (('carpart',) if product == 'carpart'
                    else ('bottom', 'back', 'flash')):
            try:
                self._load_model(key)
            except Exception as exc:
                self.get_logger().error('Side model %s: %s' % (key, exc))

    @staticmethod
    def model_key(product, view):
        if product == 'carpart':
            return 'carpart' if view in ('BOTTOM', 'BACK', 'FRONT', 'LEFT', 'RIGHT') else None
        if product == 'baseplate':
            if view == 'BOTTOM_1':
                return 'bottom'
            if view.startswith('BACK_') and view[5:] in tuple(str(n) for n in range(1, 9)):
                return 'back'
            if view.startswith('FLASH_') and view[6:] in tuple(str(n) for n in range(1, 9)):
                return 'flash'
        return None

    def _load_model(self, key):
        if key not in self.models:
            path = self.paths[key]
            if not os.path.isfile(path):
                raise FileNotFoundError(path)
            self.models[key] = YOLO(path)
            self.get_logger().info('B camera %s model: %s' % (key, path))
        return self.models[key]

    def on_command(self, msg):
        try:
            data = json.loads(msg.data)
        except ValueError:
            return
        if data.get('command') == 'STOP':
            self.session = None
            return
        if data.get('command') != 'START':
            return
        if not data.get('cycle_id') or not data.get('request_id') or not data.get('view'):
            return
        self.session = dict(data, start=time.monotonic())
        self.last_result = None
        self.last_view = str(data['view'])
        self.defect_since = None
        self.max_hold = 0.0
        self.frames = 0
        self.seen_car_part = False
        self.seen_dark_spot = False
        self.max_dark_spot_count = 0
        product = str(data.get('product', '')).lower()
        key = self.model_key(product, str(data['view']).upper())
        if product != self.product or key is None:
            self.finish('ERROR', 'invalid product/view: %s / %s' %
                        (product, data['view']))
            return
        try:
            self._load_model(key)
        except Exception as exc:
            self.finish('ERROR', 'model unavailable: %s' % exc)
            return
        self.session['model_key'] = key
        self.display_key = key

    def finish(self, judge, reason):
        s = self.session
        if s is None:
            return
        result = dict(product=self.product, cycle_id=s['cycle_id'],
                      request_id=s['request_id'], view=s['view'], judge=judge,
                      reason=reason, frames=self.frames, stamp=time.time())
        self.result_pub.publish(String(data=json.dumps(result)))
        if self.product == 'carpart':
            legacy = dict(camera='b_arm', view=s['view'], result=judge,
                          reason=reason, inspection_sec=round(time.monotonic()-s['start'], 3),
                          defect_hold_threshold_sec=self.hold,
                          max_dark_hold_sec=round(self.max_hold, 3),
                          seen_car_part=self.seen_car_part,
                          seen_dark_spot=self.seen_dark_spot,
                          max_dark_spot_count=self.max_dark_spot_count,
                          stamp=time.time())
            self.car_result_pub.publish(String(data=json.dumps(legacy)))
        elif self.product == 'baseplate':
            self.base_result_pub.publish(String(data=json.dumps(result)))
        self.last_result = judge
        self.last_view = str(s['view'])
        self.get_logger().info('%s: %s' % (s['view'], judge))
        self.session = None

    def on_image(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        display = frame.copy()
        product = self.product
        current_session = self.session
        key = (current_session.get('model_key') if current_session else
               self.display_key)
        bad = False
        has_car_part = False
        kinds = {'darkspot': 0, 'flash': 0, 'shortshot': 0}
        if key in self.models:
            try:
                model = self.models[key]
                result = model.predict(frame, conf=self.conf if key == 'carpart'
                                       else self.base_conf,
                    iou=.45, verbose=False)[0]
                names = model.names
                defects = ({'darkspot'} if product == 'carpart'
                           else {'darkspot', 'flash', 'shortshot'})
                if result.boxes is not None:
                    polygons = result.masks.xy if result.masks is not None else ()
                    for index, box in enumerate(result.boxes):
                        tag = norm(names[int(box.cls[0])])
                        if product == 'carpart' and tag == 'carpart':
                            has_car_part = True
                        if tag not in defects:
                            continue
                        bad = True
                        kinds[tag] += 1
                        a, b, c, d = map(int, box.xyxy[0].tolist())
                        color = ((255, 255, 0) if tag == 'flash' else
                                 (0, 165, 255) if tag == 'shortshot' else (0, 0, 255))
                        if tag in ('flash', 'shortshot') and index < len(polygons):
                            poly = np.asarray(polygons[index], dtype=np.int32).reshape((-1, 1, 2))
                            if len(poly) >= 3:
                                overlay = display.copy()
                                cv2.fillPoly(overlay, [poly], color)
                                cv2.addWeighted(overlay, .30, display, .70, 0, display)
                                cv2.polylines(display, [poly], True, color, 1,
                                              cv2.LINE_AA)
                            else:
                                cv2.rectangle(display, (a, b), (c, d), color, 1)
                        else:
                            cv2.rectangle(display, (a, b), (c, d), color, 1)
                        label_name = ('Dark spot' if tag == 'darkspot' else
                                      'Flash' if tag == 'flash' else 'Short shot')
                        cv2.putText(display,
                                    '%s %.2f' % (label_name, float(box.conf[0])),
                                    (a, max(20, b - 6)), cv2.FONT_HERSHEY_SIMPLEX,
                                    .45, color, 1, cv2.LINE_AA)
                if current_session is not None:
                    self.frames += 1
                    self.seen_car_part |= has_car_part
                    self.seen_dark_spot |= kinds['darkspot'] > 0
                    self.max_dark_spot_count = max(self.max_dark_spot_count,
                                                   kinds['darkspot'])
                    now = time.monotonic()
                    if bad:
                        if self.defect_since is None:
                            self.defect_since = now
                        self.max_hold = max(self.max_hold, now - self.defect_since)
                    else:
                        self.defect_since = None
                    if self.max_hold >= self.hold:
                        self.finish('NG', 'defect held %.2fs' % self.max_hold)
                    elif now - current_session['start'] >= self.duration:
                        self.finish('OK', 'no sustained defect')
            except Exception as exc:
                self.get_logger().error('Side inference: %s' % exc)
                if current_session is not None:
                    self.finish('ERROR', str(exc))
        if product == 'carpart':
            hold_sec = (time.monotonic() - self.defect_since
                        if self.defect_since is not None else 0.0)
            live = dict(camera='b_arm', view=self.last_view,
                        active=bool(self.session), has_car_part=has_car_part,
                        has_dark_spot=bool(kinds['darkspot']),
                        dark_spot_count=kinds['darkspot'],
                        dark_hold_sec=round(hold_sec, 3),
                        max_dark_hold_sec=round(self.max_hold, 3),
                        judge=self.last_result or ('CHECKING' if self.session else 'IDLE'),
                        stamp=time.time())
            self.car_state_pub.publish(String(data=json.dumps(live)))
        elif product == 'baseplate':
            live = dict(camera='baseplate_side',
                        has_dark_spot=bool(kinds['darkspot']),
                        dark_spot_count=kinds['darkspot'],
                        has_flash=bool(kinds['flash']), flash_count=kinds['flash'],
                        has_short_shot=bool(kinds['shortshot']),
                        short_shot_count=kinds['shortshot'],
                        judge='NG' if bad else 'OK')
            self.base_state_pub.publish(String(data=json.dumps(live)))
        if product == 'carpart':
            if self.last_result:
                status = '%s : %s' % (self.last_view, self.last_result)
                color = (0, 0, 255) if self.last_result == 'NG' else (0, 255, 0)
            elif self.session:
                streak = (time.monotonic() - self.defect_since
                          if self.defect_since is not None else 0.0)
                status = ('%s CHECKING | Dark hold %.2f/%.2fs' %
                          (self.last_view, streak, self.hold))
                color = (0, 255, 255)
            else:
                status = 'IDLE - waiting inspection START'
                color = (200, 200, 200)
            cv2.putText(display, status, (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, .65, color, 1, cv2.LINE_AA)
            cv2.putText(display, 'Dark spot >= %.1fs continuously -> NG' % self.hold,
                        (20, 62), cv2.FONT_HERSHEY_SIMPLEX, .45,
                        (255, 255, 255), 1, cv2.LINE_AA)
        elif product == 'baseplate':
            parts = [('%s:%d' % (name, kinds[tag])) for tag, name in
                     (('darkspot', 'DARK SPOT'), ('flash', 'FLASH'),
                      ('shortshot', 'SHORT SHOT')) if kinds[tag]]
            status = 'NG - ' + ' / '.join(parts) if parts else 'OK'
            cv2.putText(display, status, (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, .65,
                        (0, 0, 255) if parts else (0, 255, 0), 1, cv2.LINE_AA)
        cv2.imshow(self.window_name, display)
        key = cv2.waitKey(1) & 255
        if key in (ord('q'), 27):
            self.get_logger().info('B arm camera quit requested')
            self.stop_pub.publish(String(data='stop'))
            if rclpy.ok():
                rclpy.shutdown()

    def destroy_node(self):
        cv2.destroyWindow(self.window_name)
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = Side()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
