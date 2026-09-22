"""
robot_description / path_collision_guard_ee_5axis.py
----------------------------------------------------
엔드이펙터 포함 로봇의 경로 충돌 검사(별도 DIRECT 서버).
 - 바닥 충돌(로봇 vs plane)
 - 비인접(부모-자식 아님) 링크끼리 충돌  (인접 링크는 관절 한계각으로 제한)
 - home 에서 이미 닿은 쌍은 무시(형상 근사 오탐 방지)
위치 IK 는 q1~q4, joint5(그리퍼)는 0 으로 두고 검사한다.
"""
import os, re
import numpy as np
import pybullet as p
import pybullet_data

try:
    import rclpy
    from rclpy.node import Node
    from ament_index_python.packages import get_package_share_directory
except Exception:
    rclpy = None
    Node = None
    get_package_share_directory = None


class CollisionChecker:
    def __init__(self, urdf_src, pkg_dir, n_pos_joints=4, mesh_dir=None):
        self.mesh_dir = mesh_dir
        self.n_pos = n_pos_joints
        self.cid = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.cid)
        self.plane = p.loadURDF('plane.urdf', physicsClientId=self.cid)
        self.robot = p.loadURDF(self._prep(urdf_src, pkg_dir), [0, 0, 0], useFixedBase=True,
                                flags=p.URDF_USE_SELF_COLLISION, physicsClientId=self.cid)
        self.n_links = p.getNumJoints(self.robot, physicsClientId=self.cid)
        self.rev = [j for j in range(self.n_links)
                    if p.getJointInfo(self.robot, j, physicsClientId=self.cid)[2] == p.JOINT_REVOLUTE]
        for a in range(-1, self.n_links):
            for b in range(a + 1, self.n_links):
                p.setCollisionFilterPair(self.robot, self.robot, a, b,
                                         1 if abs(a - b) > 1 else 0, physicsClientId=self.cid)
        self._set_q(np.zeros(self.n_pos))
        self.ignore = self._selfpairs()

    def _prep(self, src, pkg_dir):
        with open(src, encoding='utf-8') as f:
            txt = f.read()

        # 사용자의 노트북에서는 STL이 slider/mashes에 있으므로,
        # URDF에 적힌 기존 package/상대경로와 관계없이 같은 파일명의 mesh를 연결한다.
        if self.mesh_dir:
            mesh_dir = os.path.abspath(self.mesh_dir)

            def mesh_repl(m):
                quote = m.group(1)
                raw = m.group(2)
                if not raw.lower().endswith(('.stl', '.dae', '.obj')):
                    return m.group(0)
                base = os.path.basename(raw.replace('\\', '/'))
                candidate = os.path.join(mesh_dir, base)
                if not os.path.isfile(candidate):
                    for root, _dirs, files in os.walk(mesh_dir):
                        if base in files:
                            candidate = os.path.join(root, base)
                            break
                if os.path.isfile(candidate):
                    return 'filename=%s%s%s' % (quote, candidate, quote)
                fallback = raw.replace('package://robot_description/', pkg_dir.rstrip('/') + '/')
                return 'filename=%s%s%s' % (quote, fallback, quote)

            txt = re.sub(r'filename=(["\'])(.*?)(?:\1)', mesh_repl, txt)

        txt = txt.replace('package://robot_description/', pkg_dir.rstrip('/') + '/')
        txt = txt.replace('package://vision_pkg/', pkg_dir.rstrip('/') + '/')

        def addc(m):
            inner = re.search(r'<visual>(.*?)</visual>', m.group(0), re.S).group(1)
            return "<visual>" + inner + "</visual>\n<collision>" + inner + "</collision>"

        txt = re.sub(r'<visual>.*?</visual>', addc, txt, flags=re.S)
        out = '/tmp/ee_collision.urdf'
        with open(out, 'w', encoding='utf-8') as f:
            f.write(txt)
        return out

    def _set_q(self, q):
        # q1~q4 세팅, 나머지 회전관절(q5=그리퍼)은 0
        full = list(q) + [0.0] * (len(self.rev) - self.n_pos)
        for k, jid in enumerate(self.rev):
            p.resetJointState(self.robot, jid, float(full[k]), physicsClientId=self.cid)
        p.performCollisionDetection(physicsClientId=self.cid)

    def _selfpairs(self):
        s = set()
        for cp in p.getContactPoints(self.robot, self.robot, physicsClientId=self.cid):
            if cp[3] != cp[4] and abs(cp[3] - cp[4]) > 1:
                s.add(tuple(sorted((cp[3], cp[4]))))
        return s

    def _floor(self):
        return len(p.getContactPoints(self.robot, self.plane, physicsClientId=self.cid)) > 0

    def check_config(self, q):
        self._set_q(q)
        if self._floor():
            return False, 'floor'
        new = self._selfpairs() - self.ignore
        if new:
            return False, 'self:%s' % new
        return True, ''

    def check_path(self, traj, max_samples=None):
        """
        경로 충돌 검사.

        max_samples=None : 전체 샘플 검사
        max_samples=N    : 시작/끝을 포함해 최대 N개 지점만 균등 검사
                           (수동 teaching 명령의 지연 감소용)
        """
        traj = np.asarray(traj, float)
        n = len(traj)
        if n == 0:
            return True, -1, ''

        if max_samples is not None and int(max_samples) > 0 and n > int(max_samples):
            idxs = np.unique(
                np.linspace(0, n - 1, int(max_samples), dtype=int)
            )
        else:
            idxs = range(n)

        for i in idxs:
            ok, reason = self.check_config(traj[int(i)])
            if not ok:
                return False, int(i), reason
        return True, -1, ''

    def close(self):
        p.disconnect(physicsClientId=self.cid)


def _find_ros_resource(base_dir, subdir, filename):
    candidates = [
        os.path.join(base_dir, subdir, filename),
        os.path.join(base_dir, filename),
    ]
    for pth in candidates:
        if os.path.isfile(pth):
            return pth
    return candidates[0]


def main(args=None):
    """
    ros2 run vision_pkg path_collision_guard_ee_5axis

    기본적으로 vision_pkg/share 아래의
      urdf/robot_arm_5axis_suction.urdf
      meshes/
    를 사용해 q=[0,0,0,0] 자세를 진단한다.

    예:
      ros2 run vision_pkg path_collision_guard_ee_5axis --ros-args \
        -p urdf_name:=robot_arm_5axis_suction.urdf \
        -p q_deg:="[0.0, 20.0, -30.0, -80.0]"
    """
    if rclpy is None or Node is None or get_package_share_directory is None:
        raise RuntimeError("ROS2/ament_index_python 환경을 찾을 수 없습니다.")

    rclpy.init(args=args)
    node = Node("path_collision_guard_ee_5axis")
    checker = None
    try:
        share_dir = get_package_share_directory("vision_pkg")

        node.declare_parameter("urdf_name", "robot_arm_5axis_suction.urdf")
        node.declare_parameter("mesh_dir", os.path.join(share_dir, "meshes"))
        node.declare_parameter("q_deg", [0.0, 0.0, 0.0, 0.0])

        urdf_name = str(node.get_parameter("urdf_name").value)
        mesh_dir = str(node.get_parameter("mesh_dir").value)
        q_deg = [float(v) for v in node.get_parameter("q_deg").value]

        if not os.path.isdir(mesh_dir):
            alt = os.path.join(share_dir, "mashes")
            if os.path.isdir(alt):
                mesh_dir = alt

        urdf_src = _find_ros_resource(share_dir, "urdf", urdf_name)
        if not os.path.isfile(urdf_src):
            raise FileNotFoundError("URDF not found: %s" % urdf_src)

        checker = CollisionChecker(
            urdf_src=urdf_src,
            pkg_dir=share_dir,
            n_pos_joints=4,
            mesh_dir=mesh_dir,
        )
        ok, reason = checker.check_config(np.radians(q_deg[:4]))
        if ok:
            node.get_logger().info("collision check: SAFE q_deg=%s" % q_deg[:4])
        else:
            node.get_logger().error(
                "collision check: COLLISION q_deg=%s reason=%s" % (q_deg[:4], reason)
            )
    finally:
        if checker is not None:
            checker.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
