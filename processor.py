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


def load_rois(config_path: str = "config.json") -> List[ROIConfig]:
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


def point_in_roi(x: float, y: float, roi: ROIConfig) -> bool:
  x1, y1, x2, y2 = roi.vertices
  return x1 <= x <= x2 and y1 <= y <= y2


class FrameQueue:
  """简单的线程安全队列，用于和 Streamlit 前端通信。"""

  def __init__(self, maxsize: int = 5):
    self.q: "queue.Queue[Tuple[np.ndarray, dict]]" = queue.Queue(maxsize=maxsize)

  def put(self, frame: np.ndarray, info: dict):
    if self.q.full():
      try:
        self.q.get_nowait()
      except queue.Empty:
        pass
    self.q.put((frame, info))

  def get_latest(self) -> Optional[Tuple[np.ndarray, dict]]:
    latest = None
    while not self.q.empty():
      try:
        latest = self.q.get_nowait()
      except queue.Empty:
        break
    return latest


class Processor(threading.Thread):
  """视频检测与人流统计处理线程。"""

  def __init__(
    self,
    video_path: str,
    frame_queue: FrameQueue,
    config_path: str = "config.json",
    model_path: str = "yolov8n.pt",
    confidence: float = 0.25,
    dbscan_eps: float = 50.0,
    dbscan_min_samples: int = 3,
    min_dwell_time: float = 0.0,
    log_manager: Optional[LogManager] = None,
    tag_rules: Optional[dict] = None,
  ):
    super().__init__(daemon=True)
    self.video_path = video_path
    self.frame_queue = frame_queue
    self.config_path = config_path
    self._stop_event = threading.Event()
    self.model = YOLO(model_path)
    self.rois = load_rois(config_path)
    self.tracks: Dict[int, TrackState] = {}
    self.confidence = confidence
    self.dbscan_eps = dbscan_eps
    self.dbscan_min_samples = dbscan_min_samples
    self.min_dwell_time = min_dwell_time
    self.current_frame_id = 0
    self.log_manager = log_manager
    self.fps = 25.0  # 默认 FPS，会在 run() 中更新
    self.tag_rules = tag_rules or {
      "expert_min_dwell": 60,
      "expert_max_focus": 15,
      "normal_min_dwell": 20,
      "normal_max_focus": 30,
      "brief_min_dwell": 5,
      "casual_max_dwell": 3
    }

  def stop(self):
    self._stop_event.set()

  def run(self):
    cap = cv2.VideoCapture(self.video_path)
    if not cap.isOpened():
      raise RuntimeError(f"无法打开视频: {self.video_path}")

    self.fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    
    # 如果提供了 log_manager，开始新会话
    if self.log_manager:
      self.log_manager.start_session()
    
    # 如果提供了 log_manager，开始新会话
    if self.log_manager:
      self.log_manager.start_session()

    while not self._stop_event.is_set():
      ok, frame = cap.read()
      if not ok:
        break

      ts = datetime.now()
      self.current_frame_id += 1

      # 使用 YOLOv8 跟踪模式，保持 ID 一致，使用可调置信度
      results = self.model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        conf=self.confidence,
        verbose=False,
      )

      boxes = results[0].boxes if results and len(results) else None
      centroids = []
      ids = []

      if boxes is not None and boxes.id is not None:
        for box, track_id in zip(boxes.xyxy.cpu().numpy(), boxes.id.cpu().numpy()):
          x1, y1, x2, y2 = box
          cx = float((x1 + x2) / 2)
          cy = float((y1 + y2) / 2)
          track_id = int(track_id)

          centroids.append((cx, cy))
          ids.append(track_id)

          state = self.tracks.setdefault(track_id, TrackState())
          state.history.append((cx, cy))
          if len(state.history) > 50:
            state.history.pop(0)

      # DBSCAN 聚集计算（使用可调参数）
      cluster_ids = {}
      if centroids:
        pts = np.array(centroids)
        db = DBSCAN(eps=self.dbscan_eps, min_samples=self.dbscan_min_samples).fit(pts)
        for pid, label in zip(ids, db.labels_):
          cluster_ids[pid] = int(label)

      # 处理每个目标：heading, dwell, roi, 记录数据库
      current_in_roi_counts: Dict[int, int] = {roi.id: 0 for roi in self.rois}

      for (cx, cy), person_id in zip(centroids, ids):
        state = self.tracks[person_id]

        # heading 计算：θ = atan2(y_t - y_{t-1}, x_t - x_{t-1}) * 180/π
        # 结果范围：-180 到 180，转换为 0-360 度
        heading = None
        if len(state.history) >= 2:
          x_t, y_t = state.history[-1]
          x_prev, y_prev = state.history[-2]
          dx, dy = x_t - x_prev, y_t - y_prev
          if abs(dx) > 1e-3 or abs(dy) > 1e-3:
            heading_deg = math.degrees(math.atan2(dy, dx))
            # 转换为 0-360 度范围
            heading = heading_deg if heading_deg >= 0 else heading_deg + 360.0

        # ROI 停留时间计算
        current_roi_id = None
        area_name = None
        for roi in self.rois:
          if point_in_roi(cx, cy, roi):
            current_roi_id = roi.id
            area_name = roi.name
            current_in_roi_counts[roi.id] += 1
            break
        
        # 判断是否属于聚类
        is_in_cluster = cluster_ids.get(person_id, -1) != -1
        
        # 判断事件类型
        event_type = EventType.STAY
        
        if current_roi_id is not None:
          # 进入或持续停留
          if state.in_roi_id is None:
            state.in_roi_id = current_roi_id
            state.enter_time = ts
            state.dwell_frames = 1
            event_type = EventType.ENTER
          elif state.in_roi_id == current_roi_id:
            state.dwell_frames += 1
            dwell_time_sec = state.dwell_frames / self.fps
            # 检查是否超时
            if dwell_time_sec >= self.min_dwell_time and dwell_time_sec > 0:
              event_type = EventType.OVERTIME
            else:
              event_type = EventType.STAY
          else:
            # 从一个区域切换到另一个区域，先结算前一个
            if state.enter_time is not None:
              total_time = state.dwell_frames / self.fps
              insert_stay_record(
                person_id=person_id,
                roi_id=state.in_roi_id,
                enter_time=state.enter_time,
                leave_time=ts,
                total_time=total_time,
              )
            state.in_roi_id = current_roi_id
            state.enter_time = ts
            state.dwell_frames = 1
            event_type = EventType.ENTER
        else:
          # 离开所有区域，若之前在区域中则结算
          if state.in_roi_id is not None and state.enter_time is not None:
            total_time = state.dwell_frames / self.fps
            if total_time >= self.min_dwell_time:
              insert_stay_record(
                person_id=person_id,
                roi_id=state.in_roi_id,
                enter_time=state.enter_time,
                leave_time=ts,
                total_time=total_time,
              )
            event_type = EventType.LEAVE
          state.in_roi_id = None
          state.enter_time = None
          state.dwell_frames = 0
        
        # 检查是否拥挤（聚类且人数 >= MinPts）
        if is_in_cluster:
          cluster_size = sum(1 for cid in cluster_ids.values() if cid == cluster_ids.get(person_id))
          if cluster_size >= self.dbscan_min_samples:
            event_type = EventType.CROWDED
        
        # 计算当前停留时间
        current_dwell_time = state.dwell_frames / self.fps if state.in_roi_id is not None else 0.0
        
        # 记录到日志管理器
        if self.log_manager:
          self.log_manager.add_record(
            frame_id=self.current_frame_id,
            visitor_id=person_id,
            position_x=cx,
            position_y=cy,
            heading_angle=heading,
            area_id=area_name,
            dwell_time=current_dwell_time,
            is_cluster=is_in_cluster,
            event_type=event_type,
            timestamp=ts,
          )

        # 原始事件写入 raw_events（包含 frame_id）
        insert_raw_event(
          ts=ts,
          person_id=person_id,
          x=cx,
          y=cy,
          roi_id=current_roi_id,
          heading=heading,
          cluster_id=cluster_ids.get(person_id),
          frame_id=self.current_frame_id,
        )

      # 画 ROI、框、朝向箭头
      vis_frame = frame.copy()
      for roi in self.rois:
        x1, y1, x2, y2 = roi.vertices
        cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        text = f"{roi.name}"
        cv2.putText(
          vis_frame,
          text,
          (x1, max(0, y1 - 5)),
          cv2.FONT_HERSHEY_SIMPLEX,
          0.6,
          (0, 255, 0),
          2,
          cv2.LINE_AA,
        )

      for (cx, cy), person_id in zip(centroids, ids):
        cv2.circle(vis_frame, (int(cx), int(cy)), 4, (0, 0, 255), -1)
        state = self.tracks[person_id]
        heading = None
        if len(state.history) >= 2:
          x_t, y_t = state.history[-1]
          x_prev, y_prev = state.history[-2]
          dx, dy = x_t - x_prev, y_t - y_prev
          if abs(dx) > 1e-3 or abs(dy) > 1e-3:
            heading_deg = math.degrees(math.atan2(dy, dx))
            # 转换为 0-360 度范围
            heading = heading_deg if heading_deg >= 0 else heading_deg + 360.0
        
        # 绘制 DBSCAN 聚类圆圈
        if person_id in cluster_ids and cluster_ids[person_id] != -1:
          cv2.circle(vis_frame, (int(cx), int(cy)), int(self.dbscan_eps), (255, 255, 0), 2)

        if heading is not None:
          length = 30
          rad = math.radians(heading)
          x2 = int(cx + length * math.cos(rad))
          y2 = int(cy + length * math.sin(rad))
          cv2.arrowedLine(
            vis_frame,
            (int(cx), int(cy)),
            (x2, y2),
            (255, 0, 0),
            2,
            tipLength=0.3,
          )
        cv2.putText(
          vis_frame,
          f"ID {person_id}",
          (int(cx) + 5, int(cy) + 5),
          cv2.FONT_HERSHEY_SIMPLEX,
          0.5,
          (255, 255, 0),
          1,
          cv2.LINE_AA,
        )

      # 统计信息
      total_people = len(set(ids))
      # 计算各区域平均停留时间
      roi_avg_dwell = {}
      for roi in self.rois:
        roi_tracks = [
          t for t in self.tracks.values()
          if t.in_roi_id == roi.id and t.enter_time is not None
        ]
        if roi_tracks:
          avg_dwell = sum(t.dwell_frames / self.fps for t in roi_tracks) / len(roi_tracks)
          roi_avg_dwell[roi.id] = avg_dwell
      
      info = {
        "timestamp": ts.isoformat(timespec="seconds"),
        "frame_id": self.current_frame_id,
        "total_people": total_people,
        "roi_counts": current_in_roi_counts,
        "roi_avg_dwell": roi_avg_dwell,
        "clusters": len([c for c in cluster_ids.values() if c != -1]),
      }

      self.frame_queue.put(vis_frame, info)

      # 控制速度，避免 CPU 跑满
      time.sleep(0.001)

    cap.release()
    
    # 检测结束时，自动导出日志（如果提供了 log_manager）
    if self.log_manager and self.log_manager.session_data:
      try:
        json_path, csv_path = self.log_manager.export_session()
        print(f"✅ 检测完成，日志已自动导出：{json_path}, {csv_path}")
      except Exception as e:
        print(f"⚠️ 日志导出失败: {e}")
    
    # 检测结束时，自动导出日志（如果提供了 log_manager）
    if self.log_manager and self.log_manager.session_data:
      try:
        json_path, csv_path = self.log_manager.export_session()
        print(f"✅ 检测完成，日志已自动导出：{json_path}, {csv_path}")
      except Exception as e:
        print(f"⚠️ 日志导出失败: {e}")

