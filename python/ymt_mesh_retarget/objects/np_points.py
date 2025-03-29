import numpy as np
from scipy.spatial.transform import Rotation

from .base import RetargetableObject


class NumpyPointsObject(RetargetableObject):
    """Implementation of RetargetableObject for numpy points (np.ndarray)."""

    def __init__(self, points: np.ndarray) -> None:
        """Initialize the JointObject instance."""
        if not isinstance(points, np.ndarray):
            raise ValueError("NumpyPointsObject only accepts np.ndarray as input")

        self.points = points

    def get_points(self, sampling_stride: int = 1) -> np.ndarray:  # noqa: ARG002
        """Get the points."""
        return self.points

    def get_transforms(self) -> list[dict]:
        """ジョイントの変換情報を取得."""
        # 位置を取得
        pos = self.get_points()

        return [
            {
                "path": None,
                "position": pos,
                "rotation": None,
                "scale": None,
            },
        ]

    def duplicate(self, suffix: str = "_retarget") -> "NumpyPointsObject":
        """Dummy method to duplicate the object.

        Numpy Points object does not have a duplicate method.
        """
        return self

    def apply_transforms(self, transform_data: list[dict]) -> None:
        """Apply the transform data to the object.

        Numpy Points object does not have a transform method.
        """
        pass

    def calculate_threshold_distance(self, coefficient: float) -> float:
        """Calculate the threshold distance.

        Numpy Points object does not have a threshold distance method.
        """
        pass

    def get_children(self, type_filter: str = None) -> list["RetargetableObject"]:
        """Get the children objects.

        Numpy Points object does not have children objects.
        """
        return []

    @staticmethod
    def create_from_path(path: str) -> "NumpyPointsObject":
        """Create an instance from the path."""
        raise NotImplementedError("NumpyPointsObject does not support create_from_path method")
