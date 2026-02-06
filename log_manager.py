"""
日志记录管理器：负责生成结构化的检测报告文档（JSON 和 CSV）
"""
import csv
import json
import os
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple

# 事件类型枚举
class EventType(Enum):
    ENTER = "ENTER"
    STAY = "STAY"
    LEAVE = "LEAVE"
    OVERTIME = "OVERTIME"
    CROWDED = "CROWDED"


class LogManager:
    """管理检测日志的记录和导出"""
    
    def __init__(self, logs_dir: str = "logs"):
        self.logs_dir = logs_dir
        self.session_data: List[Dict] = []
        self.session_start_time: Optional[datetime] = None
        os.makedirs(logs_dir, exist_ok=True)
    
    def start_session(self):
        """开始新的检测会话"""
        self.session_data = []
        self.session_start_time = datetime.now()
    
    def add_record(
        self,
        frame_id: int,
        visitor_id: int,
        position_x: float,
        position_y: float,
        heading_angle: Optional[float],
        area_id: Optional[str],
        dwell_time: float,
        is_cluster: bool,
        event_type: EventType,
        timestamp: Optional[datetime] = None,
    ):
        """添加一条检测记录
        
        Args:
            frame_id: 视频帧序号 (Integer)
            visitor_id: 追踪器分配的唯一 ID (Integer)
            position_x: 行人中心点 X 坐标 (Float)
            position_y: 行人中心点 Y 坐标 (Float)
            heading_angle: 朝向角度 0-360度 (Float, 可为 None)
            area_id: 所属区域名称 (String, 可为 None)
            dwell_time: 在该区域累计停留秒数 (Float)
            is_cluster: 是否属于 DBSCAN 识别出的拥挤群体 (Boolean)
            event_type: 事件类型 (EventType)
            timestamp: 记录时间，默认使用当前时间
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        record = {
            "timestamp": timestamp.isoformat() + "Z",  # ISO8601 格式 String
            "frame_id": int(frame_id),  # Integer
            "visitor_id": int(visitor_id),  # Integer
            "position_x": float(position_x),  # Float
            "position_y": float(position_y),  # Float
            "heading_angle": float(heading_angle) if heading_angle is not None else None,  # Float or None
            "area_id": str(area_id) if area_id is not None else None,  # String or None
            "dwell_time": float(dwell_time),  # Float
            "is_cluster": bool(is_cluster),  # Boolean
            "event_type": event_type.value,  # String (Enum value)
        }
        self.session_data.append(record)
    
    def export_session(self, filename_prefix: Optional[str] = None) -> Tuple[str, str]:
        """导出当前会话的数据为 JSON 和 CSV 文件
        
        Returns:
            (json_path, csv_path): 生成的文件路径元组
        """
        if not self.session_data:
            raise ValueError("没有数据可导出")
        
        if filename_prefix is None:
            if self.session_start_time:
                filename_prefix = self.session_start_time.strftime("session_%Y%m%d_%H%M%S")
            else:
                filename_prefix = datetime.now().strftime("session_%Y%m%d_%H%M%S")
        
        json_path = os.path.join(self.logs_dir, f"{filename_prefix}.json")
        csv_path = os.path.join(self.logs_dir, f"{filename_prefix}.csv")
        
        # 导出 JSON
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "session_start": self.session_start_time.isoformat() + "Z" if self.session_start_time else None,
                    "session_end": datetime.now().isoformat() + "Z",
                    "total_records": len(self.session_data),
                    "data": self.session_data,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        
        # 导出 CSV（便于 Excel 打开）
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:  # utf-8-sig 支持 Excel 中文
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "timestamp",
                    "frame_id",
                    "visitor_id",
                    "position_x",
                    "position_y",
                    "heading_angle",
                    "area_id",
                    "dwell_time",
                    "is_cluster",
                    "event_type",
                ],
            )
            writer.writeheader()
            for record in self.session_data:
                # 确保 CSV 中的 None 值显示为空字符串
                csv_record = {k: (v if v is not None else "") for k, v in record.items()}
                writer.writerow(csv_record)
        
        return json_path, csv_path
    
    def clear_session(self):
        """清空当前会话数据"""
        self.session_data = []
        self.session_start_time = None
    
    def get_session_summary(self) -> Dict:
        """获取当前会话的统计摘要"""
        if not self.session_data:
            return {
                "total_records": 0,
                "unique_visitors": 0,
                "unique_areas": set(),
                "total_clusters": 0,
            }
        
        unique_visitors = len(set(r["visitor_id"] for r in self.session_data))
        unique_areas = set(r["area_id"] for r in self.session_data if r["area_id"])
        total_clusters = sum(1 for r in self.session_data if r["is_cluster"])
        
        return {
            "total_records": len(self.session_data),
            "unique_visitors": unique_visitors,
            "unique_areas": list(unique_areas),
            "total_clusters": total_clusters,
        }
