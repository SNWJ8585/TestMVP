import json
import math
import queue
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from sklearn.cluster import DBSCAN
from ultralytics import YOLO

from database import insert_raw_event, insert_stay_record
from log_manager import EventType, LogManager
# 导入 Stage 4 逻辑
from stage4_tags import rule_engine

@dataclass
class ROIConfig:
    id: int
    name: str
    vertices: Tuple[int, int, int, int]

@dataclass
class TrackState:
    history: List[Tuple[float, float]] = field(default_factory=list)
    in_roi_id: Optional[int] = None
    enter_time: Optional[datetime] = None
    dwell_frames: int = 0
    last_label: str = "普通访客"  # 新增：记录当前标签

def load_rois(config_path: str = "config.json") -> List[ROIConfig]:
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        rois: List[ROIConfig] = []
        for item in data.get("rois", []):
            rois.append(
                ROIConfig(
                    id=item["id"],
                    name=item["name"],
                    vertices=tuple(item["vertices"]),
                )
            )
        return rois
    except Exception as e:
        print(f"加载ROI配置失败: {e}")
        return []

def point_in_rect(px, py, rect):
    x1, y1, x2, y2 = rect
    return x1 <= px <= x2 and y1 <= py <= y2

class Processor(threading.Thread):
    def __init__(
        self,
        video_path: str,
        frame_queue,
        model_path: str = "yolov8n.pt",
        confidence: float = 0.25,
        dbscan_eps: float = 50.0,
        dbscan_min_samples: int = 3,
        log_manager: Optional[LogManager] = None,
    ):
        super().__init__(daemon=True)
        self.video_path = video_path
        self.frame_queue = frame_queue
        self.model = YOLO(model_path)
        self.conf = confidence
        self.eps = dbscan_eps
        self.min_samples = dbscan_min_samples
        self.log_manager = log_manager
        
        self.rois = load_rois()
        self.tracks: Dict[int, TrackState] = {}
        self.fps = 30.0
        self.current_frame_id = 0
        self.running = False
        
        # 优化：数据库写入计数器
        self.db_write_interval = 10
        self.frame_counter = 0

    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            return
        self.fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.running = True

        while self.running:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            self.current_frame_id += 1
            self.frame_counter += 1
            results = self.model.track(frame, persist=True, conf=self.conf, verbose=False)
            
            vis_frame = frame.copy()
            ts = datetime.now()
            
            current_in_roi_counts = {roi.id: 0 for roi in self.rois}
            
            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                ids = results[0].boxes.id.cpu().numpy().astype(int)
                
                # 聚类分析
                centers = []
                for box in boxes:
                    centers.append([(box[0]+box[2])/2, (box[1]+box[3])/2])
                
                cluster_ids = {}
                if len(centers) >= self.min_samples:
                    clustering = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit(centers)
                    for i, tid in enumerate(ids):
                        cluster_ids[tid] = clustering.labels_[i]
                else:
                    cluster_ids = {tid: -1 for tid in ids}

                for i, track_id in enumerate(ids):
                    x1, y1, x2, y2 = boxes[i]
                    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                    
                    if track_id not in self.tracks:
                        self.tracks[track_id] = TrackState()
                    
                    state = self.tracks[track_id]
                    state.history.append((cx, cy))
                    if len(state.history) > 30: state.history.pop(0)

                    # ROI 判定
                    matched_roi_id = None
                    for roi in self.rois:
                        if point_in_rect(cx, cy, roi.vertices):
                            matched_roi_id = roi.id
                            current_in_roi_counts[roi.id] += 1
                            break
                    
                    # 停留时间逻辑
                    if matched_roi_id != state.in_roi_id:
                        if state.in_roi_id is not None and state.enter_time:
                            duration = (ts - state.enter_time).total_seconds()
                            insert_stay_record(track_id, state.in_roi_id, state.enter_time.isoformat(), ts.isoformat(), duration)
                        state.in_roi_id = matched_roi_id
                        state.enter_time = ts if matched_roi_id else None
                        state.dwell_frames = 0
                    else:
                        if state.in_roi_id:
                            state.dwell_frames += 1

                    # --- 核心改进：计算 Stage 4 标签 ---
                    dwell_sec = state.dwell_frames / self.fps
                    # 模拟一个专注度（基础模式下简化处理，如果有位移则认为在观察）
                    fake_focus = 10.0 if state.in_roi_id else 50.0 
                    state.last_label = rule_engine.generate_label(dwell_sec, fake_focus)

                    # 数据库写入：每10帧写一次
                    if self.frame_counter % self.db_write_interval == 0:
                        insert_raw_event(ts.isoformat(), ts.timestamp(), track_id, cx, cy, matched_roi_id)

                    # 绘制
                    color = (0, 255, 0) if cluster_ids[track_id] != -1 else (255, 0, 0)
                    cv2.rectangle(vis_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    # 画面上显示标签
                    display_text = f"ID:{track_id} {state.last_label}"
                    cv2.putText(vis_frame, display_text, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # 组装输出信息
            visitor_labels = {tid: s.last_label for tid, s in self.tracks.items() if tid in (ids if results[0].boxes.id is not None else [])}
            
            info = {
                "timestamp": ts.strftime("%H:%M:%S"),
                "total_people": len(visitor_labels),
                "visitor_labels": visitor_labels,
                "roi_counts": current_in_roi_counts,
            }

            self.frame_queue.put(vis_frame, info)
            time.sleep(0.005)

        cap.release()

    def stop(self):
        self.running = False