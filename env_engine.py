"""
Stage 1: 环境建模引擎
核心任务：建立物理空间基准，像素到米 (m/px) 的转换。
- 透视矫正/比例尺：基于 1m*1m 校准块计算 homography_matrix
- 展品位置空间：在摄像机视图中定义展品的 2D 投影（供 Stage 3 射线检测）
"""
from typing import List, Optional, Tuple

import cv2
import numpy as np

from models.hub import ExhibitVolume3D, ModelingDataset


class EnvEngine:
    """环境建模：单应性变换与比例尺"""

    def __init__(
        self,
        map_scale_m_per_px: float = 0.01,
        homography_matrix: Optional[np.ndarray] = None,
        cam_pose: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        cam_orientation: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    ):
        self.map_scale_m_per_px = map_scale_m_per_px
        self.homography_matrix = homography_matrix  # 3x3
        self.cam_pose = cam_pose
        self.cam_orientation = cam_orientation
        self.exhibits: List[ExhibitVolume3D] = []

    def set_scale_from_calibration(
        self,
        pts_pixel: List[Tuple[float, float]],
        side_meters: float = 1.0,
    ) -> None:
        """
        通过 1m*1m 校准块（四边形像素坐标）计算比例尺。
        pts_pixel: 四边形四个顶点像素坐标，顺序为 [左上, 右上, 右下, 左下]
        假设该四边形对应真实世界 side_meters x side_meters 的正方形。
        """
        if len(pts_pixel) != 4:
            return
        src = np.array(pts_pixel, dtype=np.float32)
        # 目标：正方形，边长 side_meters 对应像素时先按边长 1 算，再乘 scale
        w = side_meters
        dst = np.array([[0, 0], [w, 0], [w, w], [0, w]], dtype=np.float32)
        H, _ = cv2.findHomography(src, dst)
        self.homography_matrix = H
        # 简化：用四边形中心处的像素尺寸估算 m/px
        cx = np.mean([p[0] for p in pts_pixel])
        cy = np.mean([p[1] for p in pts_pixel])
        pt_center = np.array([[[cx, cy]]], dtype=np.float32)
        pt_world = cv2.perspectiveTransform(pt_center, H)
        # 取一个边长的像素距离
        dx_px = np.linalg.norm(np.array(pts_pixel[1]) - np.array(pts_pixel[0]))
        if dx_px > 1e-6:
            self.map_scale_m_per_px = side_meters / dx_px
        else:
            self.map_scale_m_per_px = 0.01

    def pixel_to_world(
        self,
        x_px: float,
        y_px: float,
    ) -> Tuple[float, float]:
        """像素坐标转世界坐标 (米)。若无单应性矩阵则仅用比例尺。"""
        if self.homography_matrix is not None:
            pt = np.array([[[x_px, y_px]]], dtype=np.float32)
            pt_w = cv2.perspectiveTransform(pt, self.homography_matrix)
            return float(pt_w[0, 0, 0]), float(pt_w[0, 0, 1])
        # 线性比例尺：以图像中心为原点
        return x_px * self.map_scale_m_per_px, y_px * self.map_scale_m_per_px

    def bbox_ground_to_world(
        self,
        bbox_px: Tuple[float, float, float, float],
    ) -> Tuple[float, float, float, float]:
        """ bbox (x1,y1,x2,y2) 像素 -> 世界坐标 (米)，返回中心点 + 宽高 或 四角。PRD 要求输出 [x_real, y_real]，这里返回中心。"""
        x1, y1, x2, y2 = bbox_px
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        x_real, y_real = self.pixel_to_world(cx, cy)
        return x_real, y_real, (x2 - x1) * self.map_scale_m_per_px, (y2 - y1) * self.map_scale_m_per_px

    def to_modeling_dataset(self) -> ModelingDataset:
        """导出 Stage 1 建模数据集"""
        H_list = None
        if self.homography_matrix is not None:
            H_list = self.homography_matrix.tolist()
        return ModelingDataset(
            map_scale_m_per_px=self.map_scale_m_per_px,
            homography_matrix=H_list,
            cam_pose=self.cam_pose,
            cam_orientation=self.cam_orientation,
            exhibit_volume_3d=self.exhibits,
        )

    def load_exhibits_from_rois(self, rois: List[dict]) -> None:
        """
        从现有 config.json 的 rois 格式加载为展品 2D 投影多边形。
        rois: [{"id", "name", "vertices": [x1,y1,x2,y2]}, ...]
        """
        self.exhibits = []
        for r in rois:
            v = r.get("vertices", [])
            if len(v) >= 4:
                x1, y1, x2, y2 = v[0], v[1], v[2], v[3]
                polygon_2d = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
                self.exhibits.append(
                    ExhibitVolume3D(
                        exhibit_id=r.get("name", f"exhibit_{r.get('id', 0)}"),
                        polygon_2d=polygon_2d,
                    )
                )
