"""
数据校验与存储架构 - 严格遵循 PRD 逻辑判定数据集汇总
使用 Pydantic 进行高性能数据验证与 JSON 格式化
"""
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from pydantic import BaseModel, Field

# 允许 NumPy 类型在 Pydantic 中序列化
def np_to_list(v: Any) -> Any:
    if isinstance(v, np.ndarray):
        return v.tolist()
    return v


class ExhibitVolume3D(BaseModel):
    """展品 3D 体积定义"""
    exhibit_id: str
    # 2D 投影多边形顶点 (用于 Shapely 射线检测) [(x,y), ...]
    polygon_2d: List[Tuple[float, float]] = Field(default_factory=list)
    # 可选 3D 顶点
    vertices_3d: Optional[List[Tuple[float, float, float]]] = None

    class Config:
        arbitrary_types_allowed = True


class ModelingDataset(BaseModel):
    """Stage 1 环境建模数据集"""
    map_scale_m_per_px: float = Field(description="全局比例尺 m/px")
    homography_matrix: Optional[List[List[float]]] = Field(None, description="单应性矩阵 3x3")
    cam_pose: Tuple[float, float, float] = Field(default=(0.0, 0.0, 0.0), description="摄像机位置 [x, y, z]")
    cam_orientation: Tuple[float, float, float] = Field(default=(0.0, 0.0, 0.0), description="欧拉角")
    exhibit_volume_3d: List[ExhibitVolume3D] = Field(default_factory=list)
    map_blob: Optional[bytes] = None  # 场馆地图二进制，序列化时跳过或 base64

    class Config:
        arbitrary_types_allowed = True


class PerceptionData(BaseModel):
    """Stage 2 单帧感知数据 - 对应 PRD 感知计算输出"""
    trace_id: str = Field(description="全链路追踪 ID")
    timestamp: float = Field(description="时间戳")
    frame_id: int = Field(description="视频帧序号")
    track_id: int = Field(description="全局 ID / 多目追踪 ID")
    # 人体位置 (P0)
    bbox_ground: Tuple[float, float, float, float] = Field(description="地面投影框 [x1,y1,x2,y2]")
    x_real: float = Field(description="世界坐标 x (米)")
    y_real: float = Field(description="世界坐标 y (米)")
    detection_features: Optional[List[float]] = None  # 特征向量，用于 ReID
    # 骨架 (P1)
    skeleton_kpts_3d: Optional[List[Tuple[float, float, float]]] = None
    action_label: Optional[str] = None
    # 运动
    velocity_vector: Tuple[float, float] = Field(default=(0.0, 0.0), description="运动矢量")
    # 头部朝向 (P1)
    gaze_vector: Optional[Tuple[float, float, float]] = None  # 3D 视线矢量
    # 群体聚类 (P2)
    candidate_group_id: Optional[int] = None
    group_density: float = Field(default=0.0, description="拥挤度/亲密度")

    class Config:
        arbitrary_types_allowed = True


class BehaviorMetrics(BaseModel):
    """Stage 3 逻辑判定数据集汇总 - PRD 核心字段"""
    is_looking_at: bool = Field(default=False, description="是否看向展品")
    focus_index: Optional[float] = Field(None, description="视线偏离值")
    dwell_duration: float = Field(default=0.0, description="停留时长(秒)")
    interaction_type: str = Field(default="none", description="兴趣程度类型")
    social_group_id: Optional[int] = None
    is_communicating: bool = Field(default=False, description="交流判定")
    group_density: float = Field(default=0.0, description="组内亲密度")
    action_pattern: str = Field(default="none", description="特殊行为: 触碰、跨越等")
    is_alert: bool = Field(default=False, description="违规预警")

    class Config:
        arbitrary_types_allowed = True


class VisitorProfile(BaseModel):
    """Stage 4 游客画像 / 标签"""
    track_id: int
    trace_id: str
    label: str = Field(description="自动标签: 资深爱好者、走马观花者等")
    dwell_duration_avg: float = 0.0
    interaction_type: str = "none"
    is_alert: bool = False


class CentralDataHub:
    """
    分阶段存储 Data Hub。
    Stage 3 判定时可通过 get_history(track_id, window=30) 查询 Stage 2 历史骨架和位置。
    每阶段输入输出记录 TraceID，支持全链路查找。
    """

    def __init__(self):
        self.stage1_modeling: Optional[ModelingDataset] = None  # map_blob, homography_matrix
        self.stage2_perception: Dict[int, List[PerceptionData]] = {}  # track_id -> 历史感知数据
        self.stage3_logic: Dict[int, BehaviorMetrics] = {}  # track_id -> 当前行为指标
        self.stage4_assets: Dict[int, VisitorProfile] = {}  # 最终游客画像
        self._trace_log: List[Dict] = []  # 每阶段 TraceID 日志
        self._max_history_per_track = 500  # 单 track 保留最大历史条数

    def set_modeling(self, data: ModelingDataset, trace_id: str = "") -> None:
        self.stage1_modeling = data
        self._trace_log.append({"stage": 1, "trace_id": trace_id or "modeling", "action": "set_modeling"})

    def push_perception(self, data: PerceptionData) -> None:
        tid = data.track_id
        if tid not in self.stage2_perception:
            self.stage2_perception[tid] = []
        self.stage2_perception[tid].append(data)
        # 滑动窗口截断
        if len(self.stage2_perception[tid]) > self._max_history_per_track:
            self.stage2_perception[tid] = self.stage2_perception[tid][-self._max_history_per_track:]
        self._trace_log.append({
            "stage": 2, "trace_id": data.trace_id, "track_id": tid, "frame_id": data.frame_id,
        })

    def set_logic(self, track_id: int, metrics: BehaviorMetrics, trace_id: str = "") -> None:
        self.stage3_logic[track_id] = metrics
        self._trace_log.append({"stage": 3, "trace_id": trace_id or str(track_id), "track_id": track_id})

    def set_visitor_profile(self, profile: VisitorProfile) -> None:
        self.stage4_assets[profile.track_id] = profile
        self._trace_log.append({"stage": 4, "trace_id": profile.trace_id, "track_id": profile.track_id})

    def get_history(self, track_id: int, window: int = 30) -> List[PerceptionData]:
        """Stage 3 判定时查询 Stage 2 历史骨架和位置"""
        if track_id not in self.stage2_perception:
            return []
        hist = self.stage2_perception[track_id]
        return hist[-window:] if len(hist) >= window else hist

    def get_trace_by_id(self, trace_id: str) -> List[Dict]:
        """通过 TraceID 查找全链路数据"""
        return [e for e in self._trace_log if e.get("trace_id") == trace_id]

    def clear_perception(self) -> None:
        """清空感知缓存（新会话时调用）"""
        self.stage2_perception.clear()
        self.stage3_logic.clear()
        self.stage4_assets.clear()
        self._trace_log.clear()
