"""
智慧展厅全链路流水线：Stage 1 -> 2 -> 3 -> 4
视频循环中串联 感知 -> 逻辑判定 -> 标签化，并写入 CentralDataHub。
"""
import math
import queue
import threading
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from env_engine import EnvEngine
from logic_engine import LogicEngine
from models.hub import CentralDataHub, ModelingDataset
from perception_engine import PerceptionEngine
from stage4_tags import update_all_profiles


class FullChainPipeline(threading.Thread):
    """全链路流水线线程：读视频 -> Perception -> Logic -> Stage4 标签 -> 输出帧与统计。frame_queue 需有 put(frame, info) 与 get_latest()。"""

    def __init__(
        self,
        video_path: str,
        frame_queue,  # FrameQueue 或兼容 put(vis, info) 的对象
        config_path: str = "config.json",
        model_path: str = "yolov8n.pt",
        confidence: float = 0.25,
        dbscan_eps: float = 50.0,
        dbscan_min_samples: int = 3,
    ):
        super().__init__(daemon=True)
        self.video_path = video_path
        self.frame_queue = frame_queue
        self._stop = threading.Event()
        self.hub = CentralDataHub()
        self.env = EnvEngine(map_scale_m_per_px=0.01)
        self._load_rois(config_path)
        self.perception = PerceptionEngine(
            model_path=model_path,
            confidence=confidence,
            dbscan_eps=dbscan_eps,
            dbscan_min_samples=dbscan_min_samples,
            config_path=config_path,
        )
        self.logic = LogicEngine(self.hub, dwell_window_seconds=30.0)
        self.perception.env = self.env
        self.perception.rois = self.rois

    def _load_rois(self, config_path: str) -> None:
        import json
        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.rois = data.get("rois", [])

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise RuntimeError(f"无法打开视频: {self.video_path}")
        # Stage 1 写入 hub
        self.env.load_exhibits_from_rois(self.rois)
        self.hub.set_modeling(self.env.to_modeling_dataset(), trace_id="pipeline_init")
        while not self._stop.is_set():
            ok, frame = cap.read()
            if not ok:
                break
            ts = datetime.now()
            vis, info = self.perception.process_frame(frame, self.hub, ts)
            # 展品 2D 多边形（像素）
            exhibits_polygons: List[Tuple[str, List[Tuple[float, float]]]] = []
            for ex in self.env.exhibits:
                exhibits_polygons.append((ex.exhibit_id, ex.polygon_2d))
            # 仅对当前帧出现的 track 做逻辑判定
            current_track_ids = info.get("track_ids", [])
            for track_id in current_track_ids:
                hist = self.hub.get_history(track_id, window=1)
                if not hist:
                    continue
                last = hist[-1]
                # bbox_ground 为 (x1,y1,x2,y2) 像素
                cx = (last.bbox_ground[0] + last.bbox_ground[2]) / 2.0
                cy = (last.bbox_ground[1] + last.bbox_ground[3]) / 2.0
                ex_id = None
                for r in self.rois:
                    vert = r.get("vertices", [])
                    if len(vert) < 4:
                        continue
                    x1, y1, x2, y2 = vert[0], vert[1], vert[2], vert[3]
                    if x1 <= cx <= x2 and y1 <= cy <= y2:
                        ex_id = r.get("name", "")
                        break
                # 朝向：用 velocity 或简单用 (1,0)
                vx, vy = last.velocity_vector[0], last.velocity_vector[1]
                if abs(vx) < 1e-6 and abs(vy) < 1e-6:
                    gaze_2d = (1.0, 0.0)
                else:
                    L = math.hypot(vx, vy)
                    gaze_2d = (vx / L, vy / L)
                origin_px = (cx, cy)
                metrics = self.logic.run_logic_for_track(
                    track_id,
                    origin_px,
                    gaze_2d,
                    exhibits_polygons,
                    last.timestamp,
                    ex_id,
                    last.group_density,
                    last.trace_id,
                )
                self.hub.set_logic(track_id, metrics, last.trace_id)
            update_all_profiles(self.hub)
            info["visitor_labels"] = {tid: p.label for tid, p in self.hub.stage4_assets.items()}
            try:
                self.frame_queue.put(vis, info)
            except Exception:
                pass
        cap.release()
