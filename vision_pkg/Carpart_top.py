#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time
import json
from collections import deque

CUSTOM_PYTHON_PKGS = os.environ.get(
    "VISION_PYTHON_PKGS",
    "/media/wonseok/새 볼륨/python_pkgs",
)
if CUSTOM_PYTHON_PKGS and os.path.isdir(CUSTOM_PYTHON_PKGS):
    if CUSTOM_PYTHON_PKGS not in sys.path:
        sys.path.insert(0, CUSTOM_PYTHON_PKGS)

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
from ament_index_python.packages import get_package_share_directory

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "ultralytics를 불러오지 못했습니다. 기본 검색 경로는 "
        "/media/wonseok/새 볼륨/python_pkgs 입니다. "
        "경로가 다르면 VISION_PYTHON_PKGS 환경변수로 지정하세요."
    ) from exc


A3_W_MM = 420.0
A3_H_MM = 297.0
BUSY_FLAG_PATH = "/tmp/robot_auto_busy.flag"

PACKAGE_NAME = "vision_pkg"
MODEL_FILENAME = "Carpart.pt"

# ============================================================
# Display style - Baseplate_top.py와 동일한 기준
# OpenCV color = BGR
# ============================================================
CAR_PART_COLOR = (255, 0, 0)          # 파랑
DARK_SPOT_COLOR = (0, 0, 255)         # 빨강
SUCTION_TARGET_COLOR = (255, 255, 0)  # 시안
HAND_COLOR = (255, 255, 255)          # 흰색
CENTER_COLOR = (255, 255, 0)          # 시안
CAL_POINT_COLOR = (0, 255, 255)       # 노랑
OK_COLOR = (0, 255, 0)                # 초록
WARN_COLOR = (0, 165, 255)            # 주황
NO_OBJECT_COLOR = (0, 0, 255)         # 빨강
AUX_TEXT_COLOR = (255, 255, 255)      # 흰색
BUSY_COLOR = (255, 0, 255)            # 자홍
PUBLISHED_COLOR = (255, 200, 0)

BOX_THICKNESS = 1
TEXT_THICKNESS = 1
FONT = cv2.FONT_HERSHEY_SIMPLEX


def default_model_path():
    # 배포본의 share/pt와 --symlink-install 개발 환경의 vision_pkg/pt를 모두 지원.
    candidates = []
    try:
        candidates.append(os.path.join(
            get_package_share_directory(PACKAGE_NAME), "pt", MODEL_FILENAME
        ))
    except Exception:
        pass
    module_dir = os.path.dirname(os.path.abspath(__file__))
    candidates.extend([
        os.path.join(module_dir, "pt", MODEL_FILENAME),
        os.path.join(module_dir, MODEL_FILENAME),
    ])
    return next((path for path in candidates if os.path.isfile(path)),
                candidates[-2])


class IntegratedDetector(Node):

    def __init__(self):
        super().__init__("Carpart_top")
        self.bridge = CvBridge()

        self.declare_parameter("model_path", default_model_path())
        self.declare_parameter("image_topic", "/image_raw")
        self.declare_parameter("conf", 0.25)
        self.declare_parameter("iou", 0.45)

        # 빠른 안정화 파라미터
        self.declare_parameter("center_window", 8)
        self.declare_parameter("stable_radius_px", 2.0)
        self.declare_parameter("stable_confirm_frames", 5)
        self.declare_parameter("move_reset_px", 8.0)
        self.declare_parameter("move_confirm_frames", 2)
        self.declare_parameter("display_ema_new_weight", 0.65)

        self.model_path = os.path.expanduser(
            str(self.get_parameter("model_path").value)
        )
        self.image_topic = str(self.get_parameter("image_topic").value)
        self.conf = float(self.get_parameter("conf").value)
        self.iou = float(self.get_parameter("iou").value)

        self.center_window = max(
            3, int(self.get_parameter("center_window").value)
        )
        self.stable_radius_px = max(
            0.1, float(self.get_parameter("stable_radius_px").value)
        )
        self.stable_confirm_frames = max(
            1, int(self.get_parameter("stable_confirm_frames").value)
        )
        self.move_reset_px = max(
            1.0, float(self.get_parameter("move_reset_px").value)
        )
        self.move_confirm_frames = max(
            1, int(self.get_parameter("move_confirm_frames").value)
        )
        self.display_ema_new_weight = float(
            np.clip(
                self.get_parameter("display_ema_new_weight").value,
                0.05,
                1.0,
            )
        )

        if not os.path.isfile(self.model_path):
            raise FileNotFoundError(
                "Carpart YOLO 모델을 찾을 수 없습니다: " + self.model_path
            )

        self.model = YOLO(self.model_path)

        self.subscription = self.create_subscription(
            Image,
            self.image_topic,
            self.listener_callback,
            10,
        )
        self.det_pub = self.create_publisher(
            String,
            "/detected_object",
            10,
        )
        self.inspection_pub = self.create_publisher(
            String,
            "/inspection_state",
            10,
        )

        self.clicked_pts = []
        self.H = None
        self.h_ready = False

        self.center_history = deque(maxlen=self.center_window)
        self.display_cx = None
        self.display_cy = None
        self.stable_confirm_count = 0

        self.sent_flag = False
        self.published_px = None
        self.move_confirm_count = 0

        self.win_name = "Real-time Inspection"
        cv2.namedWindow(self.win_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.win_name, self.on_mouse)

        self.get_logger().info(f"Model path: {self.model_path}")
        self.get_logger().info(f"model classes: {self.model.names}")
        self.get_logger().info(
            "빠른 안정화 사용: "
            f"window={self.center_window}, "
            f"radius={self.stable_radius_px:.1f}px, "
            f"confirm={self.stable_confirm_frames}"
        )
        self.get_logger().info(
            "A3 좌상 -> 우상 -> 우하 -> 좌하 순서로 클릭하세요."
        )

    def on_mouse(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN or self.h_ready:
            return

        self.clicked_pts.append((int(x), int(y)))
        self.get_logger().info(
            f"클릭: ({x}, {y}) - {len(self.clicked_pts)}/4"
        )

        if len(self.clicked_pts) == 4:
            src = np.asarray(self.clicked_pts, dtype=np.float32)
            dst = np.asarray(
                [
                    [0.0, 0.0],
                    [A3_W_MM, 0.0],
                    [A3_W_MM, A3_H_MM],
                    [0.0, A3_H_MM],
                ],
                dtype=np.float32,
            )
            self.H = cv2.getPerspectiveTransform(src, dst)
            self.h_ready = True
            self.get_logger().info("★ A3 pixel -> mm 변환 준비 완료")

    def pixel_to_mm(self, px, py):
        if not self.h_ready:
            return None, None

        p = np.asarray([float(px), float(py), 1.0], dtype=np.float64)
        q = np.asarray(self.H, dtype=np.float64) @ p

        if abs(q[2]) < 1e-12:
            return None, None

        return float(q[0] / q[2]), float(q[1] / q[2])

    @staticmethod
    def is_inside_workspace_mm(x_mm, y_mm):
        if x_mm is None or y_mm is None:
            return False
        return 0.0 <= x_mm <= A3_W_MM and 0.0 <= y_mm <= A3_H_MM

    def robot_is_busy(self):
        return os.path.exists(BUSY_FLAG_PATH)

    def reset_tracking(self, clear_publish=True):
        self.center_history.clear()
        self.display_cx = None
        self.display_cy = None
        self.stable_confirm_count = 0
        self.move_confirm_count = 0

        if clear_publish:
            self.sent_flag = False
            self.published_px = None

    def robust_center(self):
        if not self.center_history:
            return None, None

        arr = np.asarray(self.center_history, dtype=np.float64)
        mx = float(np.median(arr[:, 0]))
        my = float(np.median(arr[:, 1]))

        d = np.hypot(arr[:, 0] - mx, arr[:, 1] - my)
        return (mx, my), float(np.max(d))

    def update_display_ema(self, cx, cy):
        if self.display_cx is None:
            self.display_cx = float(cx)
            self.display_cy = float(cy)
            return self.display_cx, self.display_cy

        w = self.display_ema_new_weight
        self.display_cx = (1.0 - w) * self.display_cx + w * float(cx)
        self.display_cy = (1.0 - w) * self.display_cy + w * float(cy)
        return self.display_cx, self.display_cy

    def update_stability(self, raw_cx, raw_cy):
        raw_cx = float(raw_cx)
        raw_cy = float(raw_cy)

        if self.sent_flag and self.published_px is not None:
            move_dist = float(
                np.hypot(
                    raw_cx - self.published_px[0],
                    raw_cy - self.published_px[1],
                )
            )

            if move_dist >= self.move_reset_px:
                self.move_confirm_count += 1
            else:
                self.move_confirm_count = 0

            if self.move_confirm_count >= self.move_confirm_frames:
                self.sent_flag = False
                self.published_px = None
                self.move_confirm_count = 0
                self.center_history.clear()
                self.stable_confirm_count = 0

        self.center_history.append((raw_cx, raw_cy))

        center, spread = self.robust_center()
        if center is None:
            return None, False, 0.0

        enough = len(self.center_history) >= self.center_window
        stable_now = enough and spread <= self.stable_radius_px

        if stable_now:
            self.stable_confirm_count += 1
        else:
            self.stable_confirm_count = 0

        stable_ready = (
            stable_now
            and self.stable_confirm_count >= self.stable_confirm_frames
        )

        return center, stable_ready, spread

    def publish_inspection_state(
        self,
        has_car_part=False,
        has_dark_spot=False,
        dark_spot_count=0,
    ):
        if has_car_part and has_dark_spot:
            judge = "NG"
        elif has_car_part:
            judge = "OK"
        else:
            judge = "NO_OBJECT"

        payload = {
            "has_car_part": bool(has_car_part),
            "has_dark_spot": bool(has_dark_spot),
            "dark_spot_count": int(dark_spot_count),
            "judge": judge,
            "stamp": time.time(),
        }

        msg = String()
        msg.data = json.dumps(payload, ensure_ascii=False)
        self.inspection_pub.publish(msg)
        return judge

    @staticmethod
    def draw_bounding_box(frame, box, color, label, confidence):
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            color,
            BOX_THICKNESS,
        )
        cv2.putText(
            frame,
            f"{label} {confidence:.2f}",
            (x1, max(20, y1 - 6)),
            FONT,
            0.45,
            color,
            TEXT_THICKNESS,
            cv2.LINE_AA,
        )

    def draw_calibration(self, frame):
        for i, pt in enumerate(self.clicked_pts):
            cv2.circle(frame, pt, 4, CAL_POINT_COLOR, -1)
            cv2.putText(
                frame,
                str(i + 1),
                (pt[0] + 7, pt[1] - 7),
                FONT,
                0.4,
                CAL_POINT_COLOR,
                TEXT_THICKNESS,
                cv2.LINE_AA,
            )

        txt = (
            "A3 READY"
            if self.h_ready
            else f"A3 {len(self.clicked_pts)}/4 [TL->TR->BR->BL]"
        )
        cv2.putText(
            frame,
            txt,
            (20, 95),
            FONT,
            0.55,
            OK_COLOR if self.h_ready else CAL_POINT_COLOR,
            TEXT_THICKNESS,
            cv2.LINE_AA,
        )

    def process_key(self):
        key = cv2.waitKey(1) & 0xFF

        if key == ord("r"):
            self.clicked_pts = []
            self.H = None
            self.h_ready = False
            self.reset_tracking(clear_publish=True)
            self.get_logger().info("A3 calibration / tracking reset")

        elif key in (ord("q"), 27):
            if rclpy.ok():
                rclpy.shutdown()

    def listener_callback(self, data):
        try:
            frame = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except Exception as exc:
            self.get_logger().error(f"cv_bridge error: {exc}")
            return

        try:
            result = self.model(
                frame,
                conf=self.conf,
                iou=self.iou,
                verbose=False,
            )[0]
        except Exception as exc:
            self.get_logger().error(f"YOLO inference error: {exc}")
            return

        annotated_frame = frame.copy()
        boxes = result.boxes

        if boxes is None or len(boxes) == 0:
            self.publish_inspection_state(False, False, 0)
            self.reset_tracking(clear_publish=True)

            cv2.putText(
                annotated_frame,
                "NO OBJECT",
                (20, 35),
                FONT,
                0.65,
                NO_OBJECT_COLOR,
                TEXT_THICKNESS,
                cv2.LINE_AA,
            )
            self.draw_calibration(annotated_frame)
            cv2.imshow(self.win_name, annotated_frame)
            self.process_key()
            return

        has_car_part = False
        has_suction_target = False
        has_dark_spot = False
        dark_spot_count = 0

        best_suction_target_box = None
        best_suction_target_conf = -1.0

        for box in boxes:
            class_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = str(self.model.names[class_id]).strip()
            label_l = label.lower()

            if label_l == "car_part":
                has_car_part = True
                self.draw_bounding_box(
                    annotated_frame,
                    box,
                    CAR_PART_COLOR,
                    "CarPart",
                    conf,
                )

            elif label_l == "suction_target":
                has_suction_target = True
                self.draw_bounding_box(
                    annotated_frame,
                    box,
                    SUCTION_TARGET_COLOR,
                    "suction_target",
                    conf,
                )
                if conf > best_suction_target_conf:
                    best_suction_target_conf = conf
                    best_suction_target_box = box

            elif label_l == "dark spot":
                has_dark_spot = True
                dark_spot_count += 1
                self.draw_bounding_box(
                    annotated_frame,
                    box,
                    DARK_SPOT_COLOR,
                    "DarkSpot",
                    conf,
                )

            elif label_l == "hand":
                self.draw_bounding_box(
                    annotated_frame,
                    box,
                    HAND_COLOR,
                    "hand",
                    conf,
                )

        judge = self.publish_inspection_state(
            has_car_part or has_suction_target,
            has_dark_spot,
            dark_spot_count,
        )

        if best_suction_target_box is None:
            # 한 프레임 미검출로 publish 재arm하지는 않음
            self.center_history.clear()
            self.stable_confirm_count = 0

            cv2.putText(
                annotated_frame,
                f"NO SUCTION TARGET / Dark spot:{dark_spot_count}",
                (20, 35),
                FONT,
                0.65,
                NO_OBJECT_COLOR,
                TEXT_THICKNESS,
                cv2.LINE_AA,
            )
            self.draw_calibration(annotated_frame)
            cv2.imshow(self.win_name, annotated_frame)
            self.process_key()
            return

        # float bbox center: 기존 int + //2 양자화 제거
        x1, y1, x2, y2 = best_suction_target_box.xyxy[0].tolist()
        raw_cx = (float(x1) + float(x2)) / 2.0
        raw_cy = (float(y1) + float(y2)) / 2.0

        center, stable_ready, spread_px = self.update_stability(
            raw_cx,
            raw_cy,
        )

        if center is None:
            robust_cx, robust_cy = raw_cx, raw_cy
        else:
            robust_cx, robust_cy = center

        display_cx, display_cy = self.update_display_ema(
            robust_cx,
            robust_cy,
        )

        cv2.circle(
            annotated_frame,
            (int(round(raw_cx)), int(round(raw_cy))),
            3,
            BUSY_COLOR,
            -1,
        )
        cv2.drawMarker(
            annotated_frame,
            (int(round(robust_cx)), int(round(robust_cy))),
            CENTER_COLOR,
            cv2.MARKER_CROSS,
            12,
            1,
        )
        cv2.circle(
            annotated_frame,
            (int(round(display_cx)), int(round(display_cy))),
            4,
            CENTER_COLOR,
            1,
            cv2.LINE_AA,
        )

        status_text = "A3 CALIBRATION REQUIRED"
        status_color = CAL_POINT_COLOR

        if self.h_ready:
            # 최종 좌표는 느린 EMA가 아니라 median center로 변환
            mx, my = self.pixel_to_mm(robust_cx, robust_cy)

            inside = self.is_inside_workspace_mm(mx, my)
            busy = self.robot_is_busy()

            publish_ready = (
                stable_ready
                and not self.sent_flag
                and judge in {"OK", "NG"}
                and inside
                and not busy
            )

            if publish_ready:
                msg = String()
                msg.data = (
                    # YOLO class는 suction_target이지만 로봇 제어 노드는
                    # /detected_object 메시지의 첫 필드로 'dot'을 요구한다.
                    "dot,"
                    f"{mx:.1f},"
                    f"{my:.1f},"
                    "0.0,0,"
                    f"{judge}"
                )
                self.det_pub.publish(msg)

                self.sent_flag = True
                self.published_px = (robust_cx, robust_cy)
                self.move_confirm_count = 0

                self.get_logger().info(
                    "★ 빠른 안정화 dot 좌표 전송: "
                    f"{msg.data} "
                    f"(spread={spread_px:.2f}px)"
                )

            if busy:
                status_text = "ROBOT BUSY"
                status_color = BUSY_COLOR
            elif not inside:
                status_text = "OUTSIDE WORKSPACE"
                status_color = NO_OBJECT_COLOR
            elif self.sent_flag:
                status_text = "PUBLISHED"
                status_color = PUBLISHED_COLOR
            elif stable_ready:
                status_text = "READY"
                status_color = OK_COLOR
            else:
                status_text = (
                    f"STABILIZING "
                    f"{len(self.center_history)}/{self.center_window} "
                    f"spread:{spread_px:.2f}px "
                    f"confirm:{self.stable_confirm_count}/"
                    f"{self.stable_confirm_frames}"
                )
                status_color = CAL_POINT_COLOR

            cv2.putText(
                annotated_frame,
                f"{mx:.1f}, {my:.1f} mm",
                (int(round(x1)), int(round(y2)) + 25),
                FONT,
                0.45,
                status_color,
                TEXT_THICKNESS,
                cv2.LINE_AA,
            )

        cv2.putText(
            annotated_frame,
            f"{status_text} | suction_target conf:{best_suction_target_conf:.2f}",
            (20, 40),
            FONT,
            0.65,
            status_color,
            TEXT_THICKNESS,
            cv2.LINE_AA,
        )

        cv2.putText(
            annotated_frame,
            (
                f"judge:{judge} / dark spot:{dark_spot_count}"
                if has_dark_spot
                else "judge:OK"
            ),
            (20, 68),
            FONT,
            0.45,
            DARK_SPOT_COLOR if has_dark_spot else OK_COLOR,
            TEXT_THICKNESS,
            cv2.LINE_AA,
        )

        self.draw_calibration(annotated_frame)
        cv2.imshow(self.win_name, annotated_frame)
        self.process_key()

    def destroy_node(self):
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = IntegratedDetector()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

