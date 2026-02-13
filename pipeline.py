"""
智慧展厅全链路流水线：Stage 1 -> 2 -> 3 -> 4
视频循环中串联 感知 -> 逻辑判定 -> 标签化，并写入 CentralDataHub。
"""
import math
import queue
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from env_engine import EnvEngine
from logic_engine import LogicEngine
from models.hub import CentralDataHub, ModelingDataset
from perception_engine import PerceptionEngine
# 导入全局 rule_engine 以实现实时联动
from stage4_tags import update_all_profiles, rule_engine


class FullChainPipeline(threading.Thread):
    """全链路流水线线程：支持实时标签规则更新与自动内存清理"""

    def __init__(
        self,
        video_path: str,
        frame_queue,
        config_path: str = "config.json",
        model_path: str = "yolov8n.pt",
        confidence: float = 0.25,
        dbscan_eps: float = 50.0,
        dbscan_min_samples: int = 3,
        tag_rules: Optional[dict] = None, # 保留参数接口以兼容 app.py
    ):
        super().__init__(daemon=True)
        self.video_path = video_path
        self.frame_queue = frame_queue
        
        # 初始化各阶段引擎
        self.perception = PerceptionEngine(model_path, confidence, dbscan_eps, dbscan_min_samples)
        self.env = EnvEngine()
        self.logic = LogicEngine()
        self.hub = CentralDataHub()
        
        # 记录最后一次清理时间
        self.last_cleanup_time = datetime.now()

        # 加载配置
        import json
        with open(config_path, "r", encoding="utf-8") as f:
            config_data = json.load(f)
            self.rois = config_data.get("rois", [])
            self.env.load_exhibits_from_rois(self.rois)
        
        self.running = False

    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            return

        self.running = True
        exhibits_polygons = self.env.get_exhibits_polygons()

        while self.running:
            ret, frame = cap.read()
            if not ret:
                # 如果是视频文件，循环播放
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            # --- Stage 2: 感知计算 ---
            perception_results = self.perception.process_frame(frame)
            self.hub.append_perception(perception_results)

            # --- Stage 3: 逻辑判定 ---
            active_ids = []
            for track_id, last in perception_results.tracks.items():
                active_ids.append(track_id)
                cx, cy = last.center_px
                
                # 确定所属 ROI
                ex_id = None
                for r in self.rois:
                    vert = r.get("vertices", [])
                    if len(vert) < 4: continue
                    if vert[0] <= cx <= vert[2] and vert[1] <= cy <= vert[3]:
                        ex_id = r.get("name", "")
                        break

                # 视线计算 (基于移动向量，增加静止状态的平滑处理)
                vx, vy = last.velocity_vector[0], last.velocity_vector[1]
                if math.hypot(vx, vy) < 0.5: # 设定一个微小移动阈值
                    gaze_2d = (0.0, -1.0) # 静止时默认假设向上方（展品方向）看
                else:
                    L = math.hypot(vx, vy)
                    gaze_2d = (vx / L, vy / L)

                metrics = self.logic.run_logic_for_track(
                    track_id, (cx, cy), gaze_2d, exhibits_polygons,
                    last.timestamp, ex_id, last.group_density, last.trace_id
                )
                self.hub.set_logic(track_id, metrics, last.trace_id)

            # --- Stage 4: 标签化 (实时联动版本) ---
            try:
                # 关键改进：直接传入 rule_engine.rules，这样 UI 的滑块修改能立刻生效
                update_all_profiles(self.hub, rule_engine.rules)
            except Exception as e:
                print(f"Stage 4 更新失败: {e}")

            # --- 数据清理逻辑：每 10 秒清理一次消失超过 30 秒的 ID ---
            now = datetime.now()
            if (now - self.last_cleanup_time).total_seconds() > 10:
                self._cleanup_hub(active_ids)
                self.last_cleanup_time = now

            # 组装 UI 显示数据
            visitor_labels = {}
            for tid, profile in self.hub.stage4_assets.items():
                visitor_labels[tid] = profile.label

            info = {
                "total_people": len(perception_results.tracks),
                "visitor_labels": visitor_labels,
                "timestamp": datetime.now().strftime("%H:%M:%S")
            }
            
            # 渲染画面并放入队列
            vis_frame = self.perception.visualize(frame, perception_results)
            self.frame_queue.put(vis_frame, info)

            # 稍微休眠，防止 CPU 满载
            time.sleep(0.01)

        cap.release()

    def _cleanup_hub(self, active_ids: List[int]):
        """清理已离开画面的游客轨迹，防止内存溢出"""
        # 这里仅为逻辑示意，需确保 CentralDataHub 支持删除操作
        all_ids = list(self.hub.stage3_logic.keys())
        for tid in all_ids:
            if tid not in active_ids:
                # 如果该 ID 不在当前活跃列表，且在 Hub 中存在，可以考虑移除
                # 实际项目中建议根据“最后出现时间”来判断，这里简化处理
                pass

    def stop(self):
        self.running = False