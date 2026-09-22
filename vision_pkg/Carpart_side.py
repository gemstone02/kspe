#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
B-arm side camera inspection node

핵심 동작
- /b_camera/image_raw 구독
- Car_part / Dark spot 검출
- 검사 자세별 START/END 세션 관리
- Dark spot이 연속 defect_hold_sec(기본 2.0초) 이상 검출되면 즉시 NG(불량품) 확정/전송
- END 시점까지 NG가 확정되지 않았다면 물체 검출 여부와 관계없이 OK 전송
- 결과는 /b_inspection_result 에 자세별 1회 publish
- 실시간 상태는 /b_inspection_state 에 계속 publish

검사 명령 토픽: /b_inspection_cmd (std_msgs/String)
예)
  START,B_BOTTOM_VIEW
  END
  START,B_BACK_VIEW_1
  END
  RESET

ROS2 package share/vision_pkg/pt 에서 모델을 자동 탐색한다.
"""

import os
import sys
import json
import time

# ============================================================
# 사용자 노트북의 별도 Python 패키지 폴더
# 기존 export PYTHONPATH=... 환경을 ros2 run에서도 재현
# ============================================================
CUSTOM_PYTHON_PKGS = os.environ.get(
    "VISION_PYTHON_PKGS",
    "/media/wonseok/새 볼륨/python_pkgs",
)

if CUSTOM_PYTHON_PKGS and os.path.isdir(CUSTOM_PYTHON_PKGS):
    if CUSTOM_PYTHON_PKGS not in sys.path:
        sys.path.insert(0, CUSTOM_PYTHON_PKGS)

import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
from ament_index_python.packages import get_package_share_directory

try:
    from ultralytics import YOLO
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "ultralytics를 불러오지 못했습니다. 기본 검색 경로는 "
        "'/media/wonseok/새 볼륨/python_pkgs' 입니다. "
        "경로가 다르면 VISION_PYTHON_PKGS 환경변수로 지정하세요."
    ) from exc


PACKAGE_NAME = "vision_pkg"
MODEL_CANDIDATES = (
    "Carpart(2).pt",
    "Carpart.pt",
    "Carpart_side.pt",
)

# ============================================================
# Display style - Baseplate_top.py와 동일한 기준
# OpenCV color = BGR
# ============================================================
CAR_PART_COLOR = (255, 0, 0)       # 파랑
DARK_SPOT_COLOR = (0, 0, 255)      # 빨강
OK_COLOR = (0, 255, 0)             # 초록
WARN_COLOR = (0, 165, 255)         # 주황
CAL_POINT_COLOR = (0, 255, 255)    # 노랑
AUX_TEXT_COLOR = (255, 255, 255)   # 흰색
IDLE_COLOR = (200, 200, 200)

BOX_THICKNESS = 1
TEXT_THICKNESS = 1
FONT = cv2.FONT_HERSHEY_SIMPLEX


def find_default_model_path():
    share_dir = get_package_share_directory(PACKAGE_NAME)
    pt_dir = os.path.join(share_dir, "pt")

    for filename in MODEL_CANDIDATES:
        path = os.path.join(pt_dir, filename)
        if os.path.isfile(path):
            return path

    # 오류 메시지에 사용할 첫 후보
    return os.path.join(pt_dir, MODEL_CANDIDATES[0])


class BArmInspectionCamera(Node):
    def __init__(self):
        super().__init__("Carpart_side")

        # ========================================================
        # Parameters
        # ========================================================
        self.declare_parameter("model_path", find_default_model_path())
        self.declare_parameter("image_topic", "/b_camera/image_raw")
        self.declare_parameter("command_topic", "/b_inspection_cmd")
        self.declare_parameter("state_topic", "/b_inspection_state")
        self.declare_parameter("result_topic", "/b_inspection_result")
        self.declare_parameter("window_name", "B Arm Carpart Inspection")
        self.declare_parameter("conf", 0.25)
        self.declare_parameter("iou", 0.45)

        # 불량이 '연속' 이 시간 이상 검출되면 NG 확정
        self.declare_parameter("defect_hold_sec", 2.0)

        self.model_path = os.path.expanduser(
            str(self.get_parameter("model_path").value)
        )
        self.image_topic = str(self.get_parameter("image_topic").value)
        self.command_topic = str(self.get_parameter("command_topic").value)
        self.state_topic = str(self.get_parameter("state_topic").value)
        self.result_topic = str(self.get_parameter("result_topic").value)
        self.window_name = str(self.get_parameter("window_name").value)
        self.conf = float(self.get_parameter("conf").value)
        self.iou = float(self.get_parameter("iou").value)
        self.defect_hold_sec = max(
            0.1, float(self.get_parameter("defect_hold_sec").value)
        )

        if not os.path.isfile(self.model_path):
            raise FileNotFoundError(
                "Carpart YOLO 모델을 찾을 수 없습니다: " + self.model_path
            )

        self.bridge = CvBridge()
        self.model = YOLO(self.model_path)

        # ========================================================
        # ROS I/O
        # ========================================================
        self.subscription = self.create_subscription(
            Image,
            self.image_topic,
            self.listener_callback,
            10,
        )

        self.command_sub = self.create_subscription(
            String,
            self.command_topic,
            self.command_callback,
            10,
        )

        # 실시간 프레임 상태
        self.state_pub = self.create_publisher(
            String,
            self.state_topic,
            10,
        )

        # 검사 자세 1회당 최종 결과 1회
        self.result_pub = self.create_publisher(
            String,
            self.result_topic,
            10,
        )

        # ========================================================
        # Inspection session state
        # ========================================================
        self.active = False
        self.current_view = ""
        self.view_started_at = None

        self.dark_start_at = None
        self.current_dark_hold_sec = 0.0
        self.max_dark_hold_sec = 0.0

        self.seen_car_part = False
        self.seen_dark_spot = False
        self.max_dark_spot_count = 0

        self.final_sent = False
        self.final_result = None

        # 최근 프레임 상태
        self.last_has_car_part = False
        self.last_has_dark_spot = False
        self.last_dark_spot_count = 0

        # ========================================================
        # OpenCV
        # ========================================================
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

        self.get_logger().info("B-arm Carpart inspection camera started")
        self.get_logger().info(f"image_topic   : {self.image_topic}")
        self.get_logger().info(f"command_topic : {self.command_topic}")
        self.get_logger().info(f"state_topic   : {self.state_topic}")
        self.get_logger().info(f"result_topic  : {self.result_topic}")
        self.get_logger().info(f"model_path    : {self.model_path}")
        self.get_logger().info(f"model classes : {self.model.names}")
        self.get_logger().info(
            f"NG rule       : Dark spot continuous >= {self.defect_hold_sec:.2f} sec"
        )
        self.get_logger().info(
            "command example: START,B_BOTTOM_VIEW -> END"
        )

    # ============================================================
    # Utility
    # ============================================================
    @staticmethod
    def normalize_label(label):
        return (
            str(label)
            .strip()
            .lower()
            .replace("_", " ")
            .replace("-", " ")
        )

    def reset_session(self):
        self.active = False
        self.current_view = ""
        self.view_started_at = None

        self.dark_start_at = None
        self.current_dark_hold_sec = 0.0
        self.max_dark_hold_sec = 0.0

        self.seen_car_part = False
        self.seen_dark_spot = False
        self.max_dark_spot_count = 0

        self.final_sent = False
        self.final_result = None

    def start_view(self, view_name):
        # 이전 세션이 active 상태인데 새 START가 오면 이전 검사를 먼저 끝낸다.
        if self.active and not self.final_sent:
            self.get_logger().warning(
                f"새 START 수신. 이전 view '{self.current_view}'를 먼저 종료 처리합니다."
            )
            self.finish_view(reason="new_start")

        self.reset_session()

        self.active = True
        self.current_view = str(view_name).strip() or "UNKNOWN_VIEW"
        self.view_started_at = time.monotonic()

        self.get_logger().info(
            "========================================"
        )
        self.get_logger().info(
            f"[INSPECTION START] {self.current_view}"
        )
        self.get_logger().info(
            "========================================"
        )

    def finish_view(self, reason="end"):
        if not self.active:
            self.get_logger().warning("END 수신했지만 진행 중인 검사가 없습니다.")
            return

        # 이미 2초 연속 Dark spot으로 NG 확정했다면 중복 publish하지 않는다.
        if self.final_sent:
            self.get_logger().info(
                f"[INSPECTION END] {self.current_view} -> already {self.final_result}"
            )
            self.active = False
            return

        # 다면검사는 물체 유무와 무관하게 불량 검출 여부만 판정한다.
        self.publish_final_result(
            "OK",
            "불량 미검출",
            reason=f"{reason}: no_confirmed_dark_spot",
        )

        self.active = False

    # ============================================================
    # Inspection command
    # ============================================================
    def command_callback(self, msg):
        raw = str(msg.data).strip()
        if not raw:
            return

        # 쉼표/공백 모두 허용
        normalized = raw.replace(",", " ")
        parts = normalized.split()
        key = parts[0].upper()

        if key == "START":
            if len(parts) < 2:
                self.get_logger().error(
                    "START 명령 형식: START,B_BOTTOM_VIEW"
                )
                return
            self.start_view(" ".join(parts[1:]))
            return

        if key == "END":
            self.finish_view(reason="end")
            return

        if key == "RESET":
            self.reset_session()
            self.get_logger().info("[INSPECTION RESET]")
            return

        # START를 생략하고 view 이름만 보내도 새 검사 시작으로 처리
        self.start_view(raw)

    # ============================================================
    # Final result publish
    # ============================================================
    def publish_final_result(self, result, result_ko, reason):
        if self.final_sent:
            return

        now = time.monotonic()
        elapsed = (
            now - self.view_started_at
            if self.view_started_at is not None
            else 0.0
        )

        payload = {
            "camera": "b_arm",
            "view": self.current_view,
            "result": result,
            "result_ko": result_ko,
            "reason": reason,
            "inspection_sec": round(float(elapsed), 3),
            "defect_hold_threshold_sec": round(self.defect_hold_sec, 3),
            "max_dark_hold_sec": round(self.max_dark_hold_sec, 3),
            "seen_car_part": bool(self.seen_car_part),
            "seen_dark_spot": bool(self.seen_dark_spot),
            "max_dark_spot_count": int(self.max_dark_spot_count),
            "stamp": time.time(),
        }

        out = String()
        out.data = json.dumps(payload, ensure_ascii=False)
        self.result_pub.publish(out)

        self.final_sent = True
        self.final_result = result

        self.get_logger().info(
            "========================================"
        )
        self.get_logger().info(
            f"[INSPECTION RESULT] view={self.current_view} -> {result} ({result_ko})"
        )
        self.get_logger().info(
            f"reason={reason} | max_dark_hold={self.max_dark_hold_sec:.2f}s"
        )
        self.get_logger().info(
            "========================================"
        )

    # ============================================================
    # Live state publish
    # ============================================================
    def publish_live_state(
        self,
        has_car_part,
        has_dark_spot,
        dark_spot_count,
    ):
        now = time.monotonic()
        elapsed = (
            now - self.view_started_at
            if self.active and self.view_started_at is not None
            else 0.0
        )

        if self.final_sent:
            live_judge = self.final_result
        elif self.active:
            live_judge = "CHECKING"
        else:
            live_judge = "IDLE"

        payload = {
            "camera": "b_arm",
            "view": self.current_view,
            "active": bool(self.active),
            "has_car_part": bool(has_car_part),
            "has_dark_spot": bool(has_dark_spot),
            "dark_spot_count": int(dark_spot_count),
            "dark_hold_sec": round(self.current_dark_hold_sec, 3),
            "max_dark_hold_sec": round(self.max_dark_hold_sec, 3),
            "inspection_sec": round(float(elapsed), 3),
            "judge": live_judge,
            "stamp": time.time(),
        }

        out = String()
        out.data = json.dumps(payload, ensure_ascii=False)
        self.state_pub.publish(out)

    # ============================================================
    # Image callback
    # ============================================================
    def listener_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as exc:
            self.get_logger().error(f"cv_bridge error: {exc}")
            return

        try:
            results = self.model(
                frame,
                conf=self.conf,
                iou=self.iou,
                verbose=False,
            )
        except Exception as exc:
            self.get_logger().error(f"YOLO inference error: {exc}")
            return

        result = results[0]
        boxes = result.boxes

        has_car_part = False
        has_dark_spot = False
        dark_spot_count = 0

        view = frame.copy()

        if boxes is not None:
            for box in boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                label = self.model.names[class_id]
                label_lower = self.normalize_label(label)

                if label_lower in {"car part", "carpart"}:
                    # Car part는 판정용으로만 검출하고
                    # 웹캠 화면에는 bbox / label을 표시하지 않는다.
                    has_car_part = True

                elif label_lower in {"dark spot", "darkspot"}:
                    has_dark_spot = True
                    dark_spot_count += 1

                    x1, y1, x2, y2 = map(
                        int, box.xyxy[0].tolist()
                    )
                    cv2.rectangle(
                        view, (x1, y1), (x2, y2), DARK_SPOT_COLOR, BOX_THICKNESS
                    )
                    cv2.putText(
                        view,
                        f"Dark spot {confidence:.2f}",
                        (x1, max(25, y1 - 8)),
                        FONT,
                        0.45,
                        DARK_SPOT_COLOR,
                        TEXT_THICKNESS,
                        cv2.LINE_AA,
                    )

        self.last_has_car_part = has_car_part
        self.last_has_dark_spot = has_dark_spot
        self.last_dark_spot_count = dark_spot_count

        # ========================================================
        # 검사 자세가 활성화됐을 때만 판정 시간 누적
        # ========================================================
        if self.active:
            now = time.monotonic()

            if has_car_part:
                self.seen_car_part = True

            if has_dark_spot:
                self.seen_dark_spot = True
                self.max_dark_spot_count = max(
                    self.max_dark_spot_count,
                    dark_spot_count,
                )

                if self.dark_start_at is None:
                    self.dark_start_at = now

                self.current_dark_hold_sec = now - self.dark_start_at
                self.max_dark_hold_sec = max(
                    self.max_dark_hold_sec,
                    self.current_dark_hold_sec,
                )

                # 2초 이상 연속 검출 -> 바로 NG 확정/전송
                if (
                    not self.final_sent
                    and self.current_dark_hold_sec >= self.defect_hold_sec
                ):
                    self.publish_final_result(
                        "NG",
                        "불량품",
                        reason=(
                            "dark_spot_continuous_"
                            f"{self.current_dark_hold_sec:.2f}s"
                        ),
                    )
            else:
                # 연속 검출 조건이므로 한 프레임이라도 끊기면 다시 0부터 측정
                self.dark_start_at = None
                self.current_dark_hold_sec = 0.0

        else:
            self.dark_start_at = None
            self.current_dark_hold_sec = 0.0

        self.publish_live_state(
            has_car_part,
            has_dark_spot,
            dark_spot_count,
        )

        # ========================================================
        # 화면 상태
        # ========================================================
        if self.final_sent:
            if self.final_result == "NG":
                status = f"{self.current_view} : NG / DEFECT"
                color = DARK_SPOT_COLOR
            elif self.final_result == "OK":
                status = f"{self.current_view} : OK / NO DEFECT"
                color = OK_COLOR
            else:
                status = f"{self.current_view} : OK / NO DEFECT"
                color = OK_COLOR
        elif self.active:
            status = (
                f"{self.current_view} CHECKING | "
                f"Dark hold {self.current_dark_hold_sec:.2f}/"
                f"{self.defect_hold_sec:.2f}s"
            )
            color = CAL_POINT_COLOR
        else:
            status = "IDLE - waiting inspection START"
            color = IDLE_COLOR

        cv2.putText(
            view,
            status,
            (20, 35),
            FONT,
            0.65,
            color,
            TEXT_THICKNESS,
            cv2.LINE_AA,
        )

        cv2.putText(
            view,
            "Dark spot >= %.1fs continuously -> NG" % self.defect_hold_sec,
            (20, 62),
            FONT,
            0.45,
            AUX_TEXT_COLOR,
            TEXT_THICKNESS,
            cv2.LINE_AA,
        )

        cv2.imshow(self.window_name, view)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            self.get_logger().info("사용자 종료 요청")
            if rclpy.ok():
                rclpy.shutdown()

    def destroy_node(self):
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = BArmInspectionCamera()

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


