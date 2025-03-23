from scipy.spatial.transform import Rotation

import numpy as np
from maya import cmds

from ..util import get_dag_path
from .base import RetargetableObject


class JointObject(RetargetableObject):
    """ジョイントオブジェクト用の実装."""

    def __init__(self, joint_path: str) -> None:
        if isinstance(joint_path, str):
            try:
                self.dag_path = get_dag_path(joint_path)
            except RuntimeError:
                print(f"Invalid joint path: {joint_path}")
                raise ValueError(f"Invalid joint path: {joint_path}")

            if not self.dag_path:
                raise ValueError(f"Invalid joint path: {joint_path}")
        else:
            self.dag_path = joint_path

        self.name = self.dag_path.fullPathName()

    def get_points(self, sampling_stride: int = 1) -> np.ndarray:
        """ジョイントの位置を点として取得."""
        pos = cmds.xform(self.name, query=True, worldSpace=True, translation=True)
        # return np.array([[p.x, p.y, p.z] for p in sparse_points])
        point = [pos[0], pos[1], pos[2]]
        return np.array([point])  # type: ignore

    def get_transforms(self) -> list[dict]:
        """ジョイントの変換情報を取得."""
        # 位置を取得
        pos = self.get_points()

        # 回転をクォータニオンとして取得
        rot = cmds.xform(self.name, query=True, worldSpace=True, rotation=True)
        quat = Rotation.from_euler("xyz", rot, degrees=True).as_quat()

        # スケールを取得
        scale = cmds.getAttr(f"{self.name}.scale")[0]

        return [
            {
                "path": self.name,
                "position": pos,
                "rotation": quat,
                "scale": np.array(scale),
            },
        ]

    def get_joint_hierarchy(self) -> list[str]:
        """ジョイント階層を取得."""
        joints = cmds.listRelatives(self.name, allDescendents=True, type="joint", fullPath=True) or []
        joints.insert(0, self.name)  # ルートジョイントを追加
        return joints

    def duplicate(self, suffix: str = "_retarget") -> "JointObject":
        """ジョイント階層を複製."""
        # ルートジョイントを特定
        current = self.name
        short_name = cmds.ls(current, shortNames=True)[0]
        if cmds.objExists(f"{short_name}{suffix}"):
            if len(cmds.ls(f"{short_name}{suffix}", long=True)) > 1:
                mes = f"Duplicated joint name found: {short_name}{suffix}"
                raise ValueError(mes)

            return self.__class__.create_from_path(f"{short_name}{suffix}")

        while True:
            parent = cmds.listRelatives(current, parent=True, type="joint")
            if not parent:
                break
            current = parent[0]

        # Store the original joint hierarchy names
        original_joint_hierarchy = cmds.listRelatives(current, allDescendents=True, type="joint", fullPath=True) or []
        original_joint_hierarchy.insert(0, current)

        # 階層ごと複製
        duped = cmds.duplicate(current, renameChildren=True)[0]

        # すべての複製されたジョイントにサフィックスを追加
        joint_hierarchy = cmds.listRelatives(duped, allDescendents=True, type="joint", fullPath=True) or []
        joint_hierarchy.insert(0, duped)

        # ループで処理中名前変更を伴うため、パスでの参照をUUIDでの参照にする
        joint_uuids = [cmds.ls(joint, uuid=True)[0] for joint in joint_hierarchy]

        for i, joint_uuid in enumerate(joint_uuids):
            joint = cmds.ls(joint_uuid, long=True)[0]
            if not cmds.objExists(joint):
                raise ValueError("Joint does not exist")

            name = f"""{original_joint_hierarchy[i].split("|")[-1]}{suffix}"""
            cmds.rename(joint, name)

        if cmds.objExists(f"{short_name}{suffix}"):
            if len(cmds.ls(f"{short_name}{suffix}", long=True)) > 1:
                mes = f"Duplicated joint name found: {short_name}{suffix}"
                raise ValueError(mes)

            return self.__class__.create_from_path(f"{short_name}{suffix}")

    def apply_transforms(self, transform_data: list[dict]) -> None:
        """変換情報をジョイントに適用."""
        data = transform_data[0]["position"]  # matrix
        pos = data.flatten().tolist()[0]

        # 位置を適用
        cmds.xform(self.name, worldSpace=True, translation=pos)

        # 回転を適用
        # euler = Rotation.from_quat(data["rotation"]).as_euler('xyz', degrees=True)
        # cmds.xform(self.name, worldSpace=True, rotation=euler)

        # スケールを適用
        # for i, axis in enumerate(['x', 'y', 'z']):
        #     cmds.setAttr(f"{self.name}.scale{axis.upper()}", data["scale"][i])

    def calculate_threshold_distance(self, coefficient: float) -> float:
        """しきい値距離の計算."""
        # ジョイント階層のバウンディングボックスを使用
        hierarchy = self.get_joint_hierarchy()
        bbox = cmds.exactWorldBoundingBox(hierarchy)
        diag = np.sqrt(sum((np.array(bbox[3:6]) - np.array(bbox[0:3])) ** 2))
        return diag * coefficient

    def get_children(self, type_filter: str = "joint") -> list["JointObject"]:
        """子ジョイントを取得."""
        children = cmds.listRelatives(self.name, children=True, type=type_filter, fullPath=True) or []
        return [self.__class__.create_from_path(child) for child in children]

    @staticmethod
    def create_from_path(path: str) -> "JointObject":
        """パスからインスタンスを作成."""
        return JointObject(path)
