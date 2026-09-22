# Dual-Arm Vision-Guided Robot Process

ROS2 Humble 기반의 비전 인식, 5축 A 로봇팔, 6축 B 로봇팔, 리니어 가이드 레일 및 센트릭 그리퍼 통합 제어 패키지입니다.

PyBullet 시뮬레이션과 실제 Dynamixel 로봇 제어를 지원하며, 자동차 부품 검사 공정과 베이스플레이트 조립 공정을 포함합니다.

## 개발 환경

* Ubuntu 22.04
* ROS2 Humble
* Python 3.10
* PyBullet
* OpenCV
* YOLOv8
* Dynamixel SDK
* Arduino Serial Communication

## 주요 ROS2 노드

| 파일                          | 기능                                                       |
| --------------------------- | -------------------------------------------------------- |
| `process_sequence_5axis.py` | 흡착 그리퍼를 사용하는 자동차 부품 이송·검사·분류 공정의 메인 노드                   |
| `centric_sequence_5axis.py` | 센트릭 그리퍼를 사용하는 샤프트·베어링·기어 조립 공정의 메인 노드                    |
| `Carpart_top.py`            | 상부 카메라로 자동차 부품 위치와 상부 검사 결과를 검출하여 `/detected_object`로 발행 |
| `Carpart_side.py`           | B 로봇팔의 다면검사 위치에서 부품 상태를 판정하고 검사 결과를 발행                   |
| `baseplate_hole_wonseok.py` | 베이스플레이트의 MID·SIDE 홀 좌표를 검출하여 `/detected_holes`로 발행       |

## 로봇 제어 모듈

| 파일                             | 기능                                |
| ------------------------------ | --------------------------------- |
| `dxl_bridge_suction_5axis.py`  | 흡착 엔드이펙터가 장착된 A 로봇팔의 Dynamixel 제어 |
| `dxl_bridge_centric_5axis.py`  | 센트릭 그리퍼가 장착된 A 로봇팔의 Dynamixel 제어  |
| `centric_gripper_5axis.py`     | 센트릭 그리퍼 개폐, 자동 파지 및 부하 감지         |
| `B_arm_v2_tolerance.py`        | 6축 B 로봇팔의 Dynamixel 제어와 관절각 변환    |
| `rail_serial_5axis.py`         | 흡착 공정용 리니어 가이드 레일과 솔레노이드 제어       |
| `rail_serial_centric_5axis.py` | 센트릭 그리퍼 공정용 리니어 가이드 레일 제어         |

## 경로계획 및 운동학 모듈

| 파일                                 | 기능                            |
| ---------------------------------- | ----------------------------- |
| `kinematics_ee_5axis.py`           | 5축 A 로봇팔의 순기구학 및 역기구학 계산      |
| `trajectory_5axis.py`              | 관절공간 이동 경로 생성                 |
| `vertical_path_5axis.py`           | 엔드이펙터 자세를 유지하는 수직 하강·상승 경로 생성 |
| `path_collision_guard_ee_5axis.py` | 이동 경로의 충돌 검사                  |

## 자동차 부품 검사 공정

메인 노드: `process_sequence_5axis`

1. `Carpart_top.py`가 상부 카메라 영상에서 부품을 검출합니다.
2. 부품 중심 좌표와 상부 검사 결과를 `/detected_object`로 발행합니다.
3. A 로봇팔이 검출 좌표 상부로 이동합니다.
4. 엔드이펙터 자세를 유지하면서 수직 하강합니다.
5. 솔레노이드 흡착을 작동하고 같은 경로로 상승합니다.
6. 상부 검사 결과가 NG이면 불량 배출 위치로 이동해 부품을 배출합니다.
7. 상부 검사 결과가 OK이면 BOTTOM, BACK, FRONT, LEFT, RIGHT 순서로 다면검사를 진행합니다.
8. 검사 중 NG가 발생하면 즉시 불량 배출 공정으로 전환합니다.
9. 모든 검사가 OK이면 정상 배출 위치로 이동해 부품을 배출합니다.
10. A 로봇팔, B 로봇팔과 레일이 초기 위치로 복귀합니다.

주요 토픽:

* `/detected_object`: 자동차 부품 좌표 및 상부 검사 결과
* `/b_inspection_cmd`: 측면 검사 시작 명령
* `/b_inspection_result`: 측면 검사 결과
* `/process_cmd`: 공정 제어 명령
* `/process_state`: 현재 공정 상태
* `/rail_position`: 레일 위치

## 베이스플레이트 조립 공정

메인 노드: `centric_sequence_5axis`

1. `baseplate_hole_wonseok.py`가 MID와 SIDE 홀 좌표를 검출합니다.
2. 검출 좌표를 `/detected_holes`로 발행합니다.
3. A 로봇팔과 레일이 부품 공급 위치로 이동합니다.
4. 센트릭 그리퍼가 부품을 자동으로 파지합니다.
5. A 로봇팔이 부품을 들어 올린 후 검출된 홀 위치로 이동합니다.
6. 엔드이펙터 자세를 유지하며 수직으로 부품을 삽입합니다.
7. 아래 순서로 총 6회의 픽앤플레이스를 수행합니다.

| 순서 | 부품        |  레일 위치 | 조립 위치 |
| -: | --------- | -----: | ----- |
|  1 | Shaft 1   | 663 mm | MID   |
|  2 | Shaft 2   | 718 mm | SIDE  |
|  3 | Bearing 1 | 842 mm | MID   |
|  4 | Bearing 2 | 913 mm | SIDE  |
|  5 | Gear 1    | 663 mm | MID   |
|  6 | Gear 2    | 718 mm | SIDE  |

8. 조립 완료 후 A 로봇팔은 HOME, 레일은 0 mm 위치로 복귀합니다.

주요 토픽:

* `/detected_holes`: MID·SIDE 홀 좌표
* `/centric_sequence/command`: 센트릭 공정 명령
* `/centric_sequence/state`: 센트릭 공정 상태

## 저장소 구조

```text
kspe/
├── meshes/                 # PyBullet용 STL 모델
├── resource/               # ROS2 패키지 리소스
├── urdf/                   # 로봇팔과 레일 URDF
├── vision_pkg/             # ROS2 Python 노드와 제어 모듈
├── package.xml
├── setup.cfg
└── setup.py
```

## 설치 방법

ROS2 워크스페이스의 `src` 폴더에 저장소를 복제합니다.

```bash
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws/src
git clone https://github.com/gemstone02/kspe.git
```

Python 라이브러리를 설치합니다.

```bash
python3 -m pip install numpy pybullet pyserial dynamixel-sdk
```

비전 노드를 실행하려면 다음 라이브러리도 필요합니다.

```bash
python3 -m pip install opencv-python ultralytics
```

ROS2 패키지를 빌드합니다.

```bash
cd ~/ros2_ws
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install
source install/setup.bash
```

## 시뮬레이션 실행

자동차 부품 공정:

```bash
ros2 run vision_pkg process_sequence_5axis --ros-args \
  -p use_robot:=false \
  -p use_b_robot:=false \
  -p use_rail:=false
```

센트릭 조립 공정:

```bash
ros2 run vision_pkg centric_sequence_5axis --ros-args \
  -p use_robot:=false \
  -p use_rail:=false \
  -p use_centric_gripper:=false \
  -p use_simulation:=true
```

## 실제 장비 실행 예시

```bash
ros2 run vision_pkg centric_sequence_5axis --ros-args \
  -p use_robot:=true \
  -p use_rail:=true \
  -p use_centric_gripper:=true \
  -p use_simulation:=true \
  -p dxl_port:=/dev/ttyUSB0 \
  -p dxl_baud:=2000000 \
  -p centric_id:=8 \
  -p rail_port:=/dev/ttyACM0 \
  -p rail_baud:=115200 \
  -p rail_auto_zero:=true \
  -p auto_start_on_vision:=false \
  -p check_collision:=true \
  -p full_grasp_mode:=auto
```

## 주의사항

* 실제 장비 실행 전 반드시 PyBullet 시뮬레이션으로 이동 경로를 확인해야 합니다.
* 로봇의 초기 자세와 Dynamixel 영점이 코드 설정과 일치해야 합니다.
* 장치에 따라 `/dev/ttyUSB0`, `/dev/ttyUSB1`, `/dev/ttyACM0` 번호가 달라질 수 있습니다.
* A 로봇팔과 센트릭 그리퍼는 동일한 Dynamixel 통신 포트를 사용할 수 있습니다.
* A 로봇팔과 B 로봇팔을 동시에 사용할 경우 코드 설정에 따라 별도의 U2D2 포트가 필요할 수 있습니다.
* YOLO 비전 노드를 사용하려면 학습된 `.pt` 모델 파일을 별도로 준비해야 합니다.

근데 pt파일은 노션에서 드릴게요

* 실제 장비를 처음 실행할 때 비상정지가 파이불렛 화면 옆에 뜰 텐데 계속 유심히 보다가 혹여나 로봇 이상하면 그냥 u2d2를 끄거나 비상정지 누르셔야  합니다.
