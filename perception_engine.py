"""
Stage 2: 感知计算引擎
实时视频流解析：人体位置、追踪、世界坐标、群体聚类。
P0: YOLOv8 + ByteTrack -> bbox_ground, [x_real, y_real]
P2: DBSCAN -> candidate_group_id, group_density
"""
import json
import math
import queue
import threading
import time
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from sklearn.cluster import DBSCAN
from ultralytics import YOLO

from env_engine import EnvEngine
from models.hub import CentralDataHub, PerceptionData

# 兼容原有 ROI 配置
def _load_rois(config_path: str = "config.json") -> List[dict]:
    with open(config_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("rois", [])


class PerceptionEngine:
    """感知计算：YOLO+ByteTrack + 世界坐标 + DBSCAN"""

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        confidence: float = 0.25,
        dbscan_eps: float = 50.0,
        dbscan_min_samples: int = 3,
        config_path: str = "config.json",
    ):
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.dbscan_eps = dbscan_eps
        self.dbscan_min_samples = dbscan_min_samples
        self.rois = _load_rois(config_path)
        self.env = EnvEngine(map_scale_m_per_px=0.01)
        self.env.load_exhibits_from_rois(self.rois)
        self._frame_id = 0
        self._track_velocity: Dict[int, Tuple[float, float]] = {}

    def set_hub(self, hub: CentralDataHub) -> None:
        self.hub = hub

    def process_frame(
        self,
        frame: np.ndarray,
        hub: CentralDataHub,
        ts: Optional[datetime] = None,
    ) -> Tuple[np.ndarray, Dict]:
        """
        处理单帧：检测、追踪、世界坐标、DBSCAN，并写入 hub。
        返回 (可视化帧, 统计 info dict)。
        """
        if ts is None:
            ts = datetime.now()
        self._frame_id += 1
        t_sec = ts.timestamp()

        results = self.model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=self.confidence,
            verbose=False,
        )
        boxes = results[0].boxes if results and len(results) else None
        centroids: List[Tuple[float, float]] = []
        ids: List[int] = []
        if boxes is not None and boxes.id is not None:
            for box, tid in zip(boxes.xyxy.cpu().numpy(), boxes.id.cpu().numpy()):
                x1, y1, x2, y2 = box
                cx = float((x1 + x2) / 2)
                cy = float((y1 + y2) / 2)
                centroids.append((cx, cy))
                ids.append(int(tid))

        # DBSCAN
        cluster_ids: Dict[int, int] = {}
        group_density_map: Dict[int, float] = {}
        if centroids:
            pts = np.array(centroids)
            db = DBSCAN(eps=self.dbscan_eps, min_samples=self.dbscan_min_samples).fit(pts)
            for pid, label in zip(ids, db.labels_):
                cluster_ids[pid] = int(label)
            for pid in ids:
                cid = cluster_ids.get(pid, -1)
                if cid >= 0:
                    cnt = sum(1 for k in ids if cluster_ids.get(k) == cid)
                    group_density_map[pid] = float(cnt) / max(self.dbscan_min_samples, 1)
                else:
                    group_density_map[pid] = 0.0

        # 速度（简化：用上一帧位置，这里用当前帧与上一帧需在外部维护；此处用 0）
        for pid in ids:
            if pid not in self._track_velocity:
                self._track_velocity[pid] = (0.0, 0.0)

        # 写入 hub：每个 track 一条 PerceptionData
        for (cx, cy), track_id in zip(centroids, ids):
            x_real, y_real = self.env.pixel_to_world(cx, cy)
            trace_id = str(uuid.uuid4())
            bbox_ground = (float(cx - 20), float(cy - 40), float(cx + 20), float(cy + 40))  # 近似脚底框
            pd = PerceptionData(
                trace_id=trace_id,
                timestamp=t_sec,
                frame_id=self._frame_id,
                track_id=track_id,
                bbox_ground=bbox_ground,
                x_real=x_real,
                y_real=y_real,
                velocity_vector=self._track_velocity.get(track_id, (0.0, 0.0)),
                candidate_group_id=cluster_ids.get(track_id) if cluster_ids.get(track_id, -1) >= 0 else None,
                group_density=group_density_map.get(track_id, 0.0),
            )
            hub.push_perception(pd)

        # 可视化
        vis = frame.copy()
        for r in self.rois:
            v = r.get("vertices", [])
            if len(v) >= 4:
                x1, y1, x2, y2 = int(v[0]), int(v[1]), int(v[2]), int(v[3])
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        for (cx, cy), pid in zip(centroids, ids):
            cv2.circle(vis, (int(cx), int(cy)), 4, (0, 0, 255), -1)
            if cluster_ids.get(pid, -1) >= 0:
                cv2.circle(vis, (int(cx), int(cy)), int(self.dbscan_eps), (255, 255, 0), 2)
            cv2.putText(vis, f"ID{pid}", (int(cx) + 5, int(cy) + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        info = {
            "timestamp": ts.isoformat(),
            "frame_id": self._frame_id,
            "total_people": len(ids),
            "track_ids": ids,
            "roi_counts": {},
        }
        return vis, info
