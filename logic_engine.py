"""
Stage 3: 逻辑判定引擎
基于空间几何与时序分析，判定具体业务行为。
- 视线方向 (P0): Shapely 射线-多边形相交 -> is_looking_at, focus_index
- 停留时长 (P1): 滑动窗口 + 平滑 -> dwell_duration, interaction_type
"""
from collections import deque
from typing import List, Optional, Tuple

from models.hub import BehaviorMetrics, CentralDataHub, PerceptionData

try:
    from shapely.geometry import LineString, Point, Polygon
    from shapely.ops import nearest_points
    SHAPELY_AVAILABLE = True
except ImportError:
    SHAPELY_AVAILABLE = False


def _ray_polygon_intersect_2d(
    origin: Tuple[float, float],
    direction: Tuple[float, float],
    polygon_2d: List[Tuple[float, float]],
    length: float = 5000.0,
) -> bool:
    """2D 射线与多边形是否相交。direction 为单位方向或任意方向向量。"""
    if not SHAPELY_AVAILABLE or len(polygon_2d) < 3:
        return False
    dx, dy = direction[0], direction[1]
    end = (origin[0] + length * dx, origin[1] + length * dy)
    line = LineString([origin, end])
    poly = Polygon(polygon_2d)
    return line.intersects(poly)


def _focus_index_2d(
    origin: Tuple[float, float],
    direction: Tuple[float, float],
    polygon_2d: List[Tuple[float, float]],
) -> Optional[float]:
    """视线偏离值：射线到多边形最近点的距离（像素），越小越对准。"""
    if not SHAPELY_AVAILABLE or len(polygon_2d) < 3:
        return None
    dx, dy = direction[0], direction[1]
    end = (origin[0] + 5000.0 * dx, origin[1] + 5000.0 * dy)
    line = LineString([origin, end])
    poly = Polygon(polygon_2d)
    if not line.intersects(poly):
        try:
            p1, p2 = nearest_points(line, poly)
            return float(Point(origin).distance(p1))
        except Exception:
            return None
    return 0.0  # 相交则偏离为 0


class LogicEngine:
    """逻辑判定：视线、停留、兴趣程度"""

    def __init__(self, hub: CentralDataHub, dwell_window_seconds: float = 5.0):
        self.hub = hub
        self.dwell_window_seconds = dwell_window_seconds
        self._dwell_cache: dict = {}  # track_id -> deque of (timestamp, exhibit_id or None)

    def _get_velocity_from_history(self, hist: List[PerceptionData]) -> Tuple[float, float]:
        if len(hist) < 2:
            return (0.0, 0.0)
        a, b = hist[-1], hist[-2]
        dt = a.timestamp - b.timestamp
        if abs(dt) < 1e-6:
            return (a.velocity_vector[0], a.velocity_vector[1])
        vx = (a.x_real - b.x_real) / dt
        vy = (a.y_real - b.y_real) / dt
        return (vx, vy)

    def compute_is_looking_at(
        self,
        track_id: int,
        origin_px: Tuple[float, float],
        gaze_direction_2d: Tuple[float, float],
        exhibits_polygons: List[Tuple[str, List[Tuple[float, float]]]],
    ) -> Tuple[bool, Optional[float]]:
        """
        视线方向判定：射线-多边形相交。
        gaze_direction_2d: 2D 单位方向向量 (dx, dy)，可由 heading 或 gaze_vector 投影得到。
        exhibits_polygons: [(exhibit_id, polygon_2d), ...]
        返回 (is_looking_at, focus_index)。
        """
        if not SHAPELY_AVAILABLE:
            return False, None
        # 单位化
        import math
        L = math.hypot(gaze_direction_2d[0], gaze_direction_2d[1])
        if L < 1e-9:
            return False, None
        dx = gaze_direction_2d[0] / L
        dy = gaze_direction_2d[1] / L
        direction = (dx, dy)
        focus = None
        for exhibit_id, poly in exhibits_polygons:
            if _ray_polygon_intersect_2d(origin_px, direction, poly):
                fi = _focus_index_2d(origin_px, direction, poly)
                if focus is None or (fi is not None and fi < focus):
                    focus = fi if fi is not None else 0.0
                return True, focus
        return False, focus

    def compute_dwell_and_interaction(
        self,
        track_id: int,
        current_ts: float,
        current_exhibit_id: Optional[str],
        window_seconds: Optional[float] = None,
    ) -> Tuple[float, str]:
        """
        停留时长与兴趣类型：时序滑动窗口。
        返回 (dwell_duration, interaction_type)。
        """
        w = window_seconds or self.dwell_window_seconds
        if track_id not in self._dwell_cache:
            self._dwell_cache[track_id] = deque(maxlen=2000)
        q = self._dwell_cache[track_id]
        q.append((current_ts, current_exhibit_id))
        # 只保留 window 内
        while q and current_ts - q[0][0] > w:
            q.popleft()
        # 当前展品上的连续停留
        dwell = 0.0
        if not current_exhibit_id:
            return 0.0, "none"
        t0 = None
        for ts, eid in reversed(q):
            if eid != current_exhibit_id:
                break
            t0 = ts
        if t0 is not None:
            dwell = current_ts - t0
        # 兴趣程度类型
        if dwell >= 60:
            interaction_type = "deep"
        elif dwell >= 15:
            interaction_type = "medium"
        elif dwell >= 3:
            interaction_type = "brief"
        else:
            interaction_type = "passing"
        return dwell, interaction_type

    def run_logic_for_track(
        self,
        track_id: int,
        origin_px: Tuple[float, float],
        gaze_direction_2d: Tuple[float, float],
        exhibits_polygons: List[Tuple[str, List[Tuple[float, float]]]],
        current_ts: float,
        current_exhibit_id: Optional[str],
        group_density: float,
        trace_id: str,
    ) -> BehaviorMetrics:
        """对单 track 执行 Stage 3 判定。"""
        hist = self.hub.get_history(track_id, window=30)
        is_looking_at = False
        focus_index: Optional[float] = None
        if gaze_direction_2d and exhibits_polygons:
            is_looking_at, focus_index = self.compute_is_looking_at(
                track_id, origin_px, gaze_direction_2d, exhibits_polygons
            )
        dwell_duration, interaction_type = self.compute_dwell_and_interaction(
            track_id, current_ts, current_exhibit_id
        )
        return BehaviorMetrics(
            is_looking_at=is_looking_at,
            focus_index=focus_index,
            dwell_duration=dwell_duration,
            interaction_type=interaction_type,
            social_group_id=None,  # P2 可接 NetworkX
            is_communicating=False,
            group_density=group_density,
            action_pattern="none",
            is_alert=False,
        )
