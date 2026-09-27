# ROS2 Dual-Arm Vision Cell

ROS2 Humble 기반의 비전 인식 및 듀얼 로봇팔 자동화 공정 패키지입니다.

상부 카메라로 공정품을 인식하고, A 로봇팔이 공정품을 파지·이송합니다. 이후 B 로봇팔과 여러 비전 노드를 이용해 다면 검사를 수행합니다. 필요에 따라 리니어 가이드 레일과 센트릭 그리퍼도 함께 제어합니다.

## 1. 개발 환경

- Ubuntu 22.04
- ROS2 Humble
- Python 3.10
- OpenCV
- YOLO
- Dynamixel SDK
- PyBullet
- USB 카메라 2대
- Dynamixel 기반 A/B 로봇팔
- 센트릭 그리퍼
- 리니어 가이드 레일

> 이 프로젝트는 실제 로봇팔, 카메라, 시리얼 장치에 맞춰 제작되었습니다. 다른 컴퓨터에서 실행하려면 동일한 ROS2 환경과 필요한 장치 또는 코드에 맞는 설정 변경이 필요합니다.

## 2. 주요 파일 설명

| 파일 | 역할 |
|---|---|
| `launch/cell.launch.py` | 카메라, 비전 노드 및 통합 공정 노드를 함께 실행 |
| `vision_pkg/cell.py` | 전체 공정 순서와 A/B 로봇팔 동작 관리 |
| `vision_pkg/cell_cli.py` | 터미널 명령을 이용한 통합 공정 제어 |
| `vision_pkg/cell_rail_serial.py` | 리니어 가이드 레일 시리얼 통신 |
| `vision_pkg/top.py` | 공정품 상부 검사 |
| `vision_pkg/base_top.py` | 베이스플레이트 상부 검사 |
| `vision_pkg/side.py` | 공정품 측면 검사 |
| `vision_pkg/holes.py` | 베이스플레이트 홀 검사 |
| `vision_pkg/centric_sequence_5axis.py` | A 로봇팔과 센트릭 그리퍼의 동작 순서 관리 |
| `vision_pkg/centric_gripper_5axis.py` | 센트릭 그리퍼 제어 |
| `vision_pkg/dxl_bridge_centric_5axis.py` | Dynamixel 모터 통신 |
| `vision_pkg/kinematics_ee_5axis.py` | A 로봇팔 정기구학 및 역기구학 계산 |
| `vision_pkg/trajectory_5axis.py` | 로봇팔 관절 궤적 생성 |
| `vision_pkg/vertical_path_5axis.py` | 엔드이펙터의 수직 이동 경로 생성 |
| `vision_pkg/path_collision_guard_ee_5axis.py` | 이동 경로의 충돌 검사 |
| `pt/` | YOLO 학습 모델 파일 |
| `urdf/` | 로봇 모델 파일 |
| `meshes/` | PyBullet 및 URDF에서 사용하는 3D 모델 |

## 3. 전체 공정 흐름

1. 상부 카메라로 공정품을 촬영합니다.
2. YOLO 모델이 공정품 종류와 위치를 검출합니다.
3. 검출 결과를 ROS2 토픽으로 전달합니다.
4. A 로봇팔이 검출 좌표로 이동합니다.
5. 센트릭 그리퍼가 공정품을 파지합니다.
6. A 로봇팔이 공정품을 검사 위치로 이동시킵니다.
7. B 로봇팔과 비전 노드가 상부·하부·측면 등 여러 방향에서 검사합니다.
8. 검사 결과에 따라 다음 공정을 진행합니다.
9. 필요하면 리니어 가이드 레일을 이용해 공정품 또는 로봇 위치를 변경합니다.

## 4. 저장소 내려받기

ROS2 워크스페이스의 `src` 폴더에서 실행합니다.

```bash
source /opt/ros/humble/setup.bash
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws/src
git clone https://github.com/gemstone02/kspe.git
```

이미 저장소를 내려받았다면 다음 명령으로 최신화할 수 있습니다.

```bash
cd ~/ros2_ws/src/kspe
git pull
```

## 5. 의존성 설치

ROS2 패키지에 등록된 의존성을 설치합니다.

```bash
cd ~/ros2_ws
rosdep update
rosdep install --from-paths src --ignore-src -r -y
```

추가 Python 패키지가 설치되어 있지 않다면 다음과 같이 설치합니다.

```bash
pip3 install ultralytics pyserial dynamixel-sdk pybullet
```

USB 카메라 노드가 없다면 설치합니다.

```bash
sudo apt update
sudo apt install ros-humble-v4l2-camera python3-opencv
```

## 6. 장치 연결 확인

실행 전에 카메라와 시리얼 장치가 연결되어 있는지 확인합니다.

```bash
ls /dev/video*
ls /dev/ttyUSB* /dev/ttyACM* 2>/dev/null
```

기본 장치 설정은 다음과 같습니다.

| 장치 | 기본 포트 |
|---|---|
| 상부 카메라 | `/dev/video0` |
| 로봇팔 측 카메라 | `/dev/video2` |
| A 로봇팔 및 센트릭 그리퍼 | `/dev/ttyUSB0` |
| B 로봇팔 | `/dev/ttyUSB1` |
| 리니어 가이드 레일 | `/dev/ttyACM0` |

컴퓨터 연결 순서에 따라 장치 번호가 달라질 수 있습니다. 실제 포트가 다르면 `cell.launch.py` 또는 관련 파라미터를 수정해야 합니다.

## 7. 빌드 방법

```bash
source /opt/ros/humble/setup.bash
cd ~/ros2_ws
colcon build --packages-select vision_pkg --symlink-install
source ~/ros2_ws/install/setup.bash
```

빌드가 끝난 후 새 터미널을 열었다면 아래 두 줄을 다시 실행해야 합니다.

```bash
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash
```

## 8. 통합 시스템 실행

### 터미널 1 — 전체 노드 실행

```bash
source /opt/ros/humble/setup.bash
cd ~/ros2_ws
source ~/ros2_ws/install/setup.bash

ros2 launch vision_pkg cell.launch.py \
  use_robot:=true \
  use_b_robot:=true \
  use_rail:=true
```

### 터미널 2 — 공정 제어 CLI 실행

```bash
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash
ros2 run vision_pkg cell_cli
```

두 터미널을 모두 실행해야 통합 공정과 터미널 제어 기능을 함께 사용할 수 있습니다.

## 9. 실행 옵션

실제 장치를 사용하지 않을 경우 필요한 옵션을 `false`로 변경할 수 있습니다.

```bash
ros2 launch vision_pkg cell.launch.py \
  use_robot:=false \
  use_b_robot:=false \
  use_rail:=false
```

단, 시뮬레이션 또는 장치 비활성화 동작은 각 노드의 구현 상태에 따라 제한될 수 있습니다.

## 10. 실행 전 주의사항

- 로봇 주변에 사람이나 장애물이 없는지 확인합니다.
- 비상 정지 방법을 미리 확인합니다.
- A/B 로봇팔의 USB 포트가 올바른지 확인합니다.
- 카메라 번호가 `/dev/video0`, `/dev/video2`와 일치하는지 확인합니다.
- 리니어 가이드의 이동 범위와 원점 상태를 확인합니다.
- 모델 파일이 `pt/` 폴더에 있는지 확인합니다.
- 실제 로봇을 처음 실행할 때는 낮은 속도로 시험합니다.
- 로봇팔에 전원을 인가하기 전에 초기 자세와 충돌 가능성을 확인합니다.

## 11. 코드 수정 후 다시 업로드하기

수정한 코드를 저장소 폴더에 반영한 후 다음 명령을 실행합니다.

```bash
cd ~/kspe
git status
git add -A
git commit -m "Update ROS2 cell process"
git push
```

업로드가 끝난 후 다음 결과가 나오면 로컬 파일과 GitHub가 동일한 상태입니다.

```text
Your branch is up to date with 'origin/main'.
nothing to commit, working tree clean
```

## 12. 참고 사항

GitHub에서 코드를 내려받는 것만으로 모든 컴퓨터에서 즉시 동일하게 작동하는 것은 아닙니다. 다음 조건이 맞아야 실제 공정을 실행할 수 있습니다.

- ROS2 및 Python 버전
- Python·ROS2 의존성 패키지
- YOLO 모델 파일
- 카메라 및 시리얼 장치 포트
- Dynamixel ID와 통신 속도
- 로봇의 기구학 파라미터
- 실제 로봇팔과 레일의 초기 위치
- 장치 접근 권한

동일한 하드웨어가 없는 컴퓨터에서는 코드를 확인하고 일부 시뮬레이션을 실행할 수 있지만, 실제 로봇 공정을 그대로 재현하려면 장치 구성과 파라미터를 맞춰야 합니다.
