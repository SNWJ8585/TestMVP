"""
Stage 4: 数据资产标签化
根据 dwell_duration 和 interaction_type 自动生成游客属性（如：资深爱好者、走马观花者）
"""
from models.hub import BehaviorMetrics, CentralDataHub, VisitorProfile


def label_visitor(track_id: int, metrics: BehaviorMetrics, trace_id: str) -> VisitorProfile:
    """
    根据逻辑判定结果自动赋予游客属性。
    """
    dwell = metrics.dwell_duration
    if dwell >= 60 or metrics.interaction_type == "deep":
        label = "资深爱好者"
    elif dwell >= 15 or metrics.interaction_type == "medium":
        label = "一般观众"
    elif dwell >= 3 or metrics.interaction_type == "brief":
        label = "短暂驻足"
    else:
        label = "走马观花者"
    if metrics.is_alert:
        label = label + "(违规预警)"
    return VisitorProfile(
        track_id=track_id,
        trace_id=trace_id,
        label=label,
        dwell_duration_avg=dwell,
        interaction_type=metrics.interaction_type,
        is_alert=metrics.is_alert,
    )


def update_all_profiles(hub: CentralDataHub) -> None:
    """根据 Stage 3 当前逻辑结果，刷新 Stage 4 游客画像。"""
    for track_id, metrics in list(hub.stage3_logic.items()):
        trace_id = f"track_{track_id}"
        profile = label_visitor(track_id, metrics, trace_id)
        hub.set_visitor_profile(profile)
