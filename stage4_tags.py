"""
Stage 4: 数据资产标签化
根据 dwell_duration 和 focus_index 自动生成游客属性
支持从UI动态调整阈值
"""
from models.hub import BehaviorMetrics, CentralDataHub, VisitorProfile
import datetime


class TagRuleEngine:
    """标签规则引擎 - 支持动态阈值调整"""

    def __init__(self):
        """初始化默认规则"""
        self.rules = {
            # ===== 资深爱好者规则 =====
            # 要求：长时间停留 + 高度专注
            "expert_min_dwell": 60,  # 最短停留时间（秒）
            "expert_max_focus": 15,  # 最大视线偏离值（非常专注）

            # ===== 一般观众规则 =====
            # 要求：中等停留时间 + 一般专注
            "normal_min_dwell": 20,  # 最短停留时间（秒）
            "normal_max_focus": 30,  # 最大视线偏离值（一般专注）

            # ===== 短暂驻足规则 =====
            # 要求：只看停留时间，不看视线
            "brief_min_dwell": 10,  # 最短停留时间（秒）

            # ===== 走马观花者规则 =====
            # 要求：停留时间极短
            "casual_max_dwell": 3,  # 最长停留时间（秒）
        }
        self.last_updated = None

    def update_rules(self, new_rules: dict):
        """从UI更新规则"""
        for key, value in new_rules.items():
            if key in self.rules:
                self.rules[key] = value
                print(f"规则更新: {key} = {value}")

        self.last_updated = datetime.datetime.now()

    def generate_label(self, dwell_time: float, focus_index: float = None) -> str:
        """
        根据当前规则生成标签
        规则优先级（从上到下判断）：
        1. 资深爱好者（高门槛）
        2. 一般观众（中等门槛）
        3. 短暂驻足（低门槛）
        4. 走马观花者（最低）
        """

        # === 规则1：资深爱好者 ===
        # 条件：停留时间长 AND 非常专注
        if dwell_time >= self.rules["expert_min_dwell"]:
            if focus_index is None or focus_index <= self.rules["expert_max_focus"]:
                return "资深爱好者"

        # === 规则2：一般观众 ===
        # 条件：停留时间中等 AND 一般专注
        if dwell_time >= self.rules["normal_min_dwell"]:
            if focus_index is None or focus_index <= self.rules["normal_max_focus"]:
                return "一般观众"

        # === 规则3：短暂驻足 ===
        # 条件：停留时间较短（不看视线）
        if dwell_time >= self.rules["brief_min_dwell"]:
            return "短暂驻足"

        # === 规则4：走马观花者 ===
        # 条件：停留时间极短
        return "走马观花者"

    def get_rule_summary(self) -> dict:
        """获取当前规则的文字描述"""
        return {
            "expert": f"停留≥{self.rules['expert_min_dwell']}秒 + 视线≤{self.rules['expert_max_focus']}",
            "normal": f"停留≥{self.rules['normal_min_dwell']}秒 + 视线≤{self.rules['normal_max_focus']}",
            "brief": f"停留≥{self.rules['brief_min_dwell']}秒",
            "casual": f"停留≤{self.rules['casual_max_dwell']}秒"
        }


# 创建全局规则引擎实例
rule_engine = TagRuleEngine()


def update_thresholds_from_ui(
        expert_dwell: int = None,
        expert_focus: int = None,
        normal_dwell: int = None,
        normal_focus: int = None,
        brief_dwell: int = None,
        casual_dwell: int = None
) -> dict:
    """
    从UI更新所有阈值
    """
    updates = {}

    # 资深爱好者规则
    if expert_dwell is not None:
        updates["expert_min_dwell"] = expert_dwell
    if expert_focus is not None:
        updates["expert_max_focus"] = expert_focus

    # 一般观众规则
    if normal_dwell is not None:
        updates["normal_min_dwell"] = normal_dwell
    if normal_focus is not None:
        updates["normal_max_focus"] = normal_focus

    # 短暂驻足规则
    if brief_dwell is not None:
        updates["brief_min_dwell"] = brief_dwell

    # 走马观花者规则
    if casual_dwell is not None:
        updates["casual_max_dwell"] = casual_dwell

    if updates:
        rule_engine.update_rules(updates)

    return updates


def label_visitor(track_id: int, metrics: BehaviorMetrics, trace_id: str) -> VisitorProfile:
    """
    根据逻辑判定结果自动赋予游客属性。
    使用 BehaviorMetrics 中的字段：
    - dwell_duration: 停留时间
    - focus_index: 视线偏离值（越小越专注）
    - interaction_type: 互动类型
    - is_alert: 是否违规
    """
    # 获取停留时间
    dwell = metrics.dwell_duration

    # 获取视线偏离值
    focus_index = metrics.focus_index

    # 使用规则引擎生成标签
    label = rule_engine.generate_label(dwell, focus_index)

    # 违规预警处理
    if metrics.is_alert:
        label = label + " (违规)"

    # 根据互动类型添加额外标识
    if metrics.interaction_type == "deep":
        label = label + " (互动程度高)"
    elif metrics.interaction_type == "medium":
        label = label + " (互动程度中等)"
    elif metrics.interaction_type == "brief":
        label = label + " (互动程度低)"

    return VisitorProfile(
        track_id=track_id,
        trace_id=trace_id,
        label=label,
        dwell_duration_avg=dwell,
        interaction_type=metrics.interaction_type,
        is_alert=metrics.is_alert,
    )


def update_all_profiles(hub: CentralDataHub, tag_rules: dict = None) -> None:
    """
    根据 Stage 3 当前逻辑结果，刷新 Stage 4 游客画像。

    参数:
        hub: 中央数据仓库
        tag_rules: 从UI传入的标签规则字典（可选）
            - expert_min_dwell: 资深爱好者最短停留时间
            - expert_max_focus: 资深爱好者最大视线偏离值
            - normal_min_dwell: 一般观众最短停留时间
            - normal_max_focus: 一般观众最大视线偏离值
            - brief_min_dwell: 短暂驻足最短停留时间
            - casual_max_dwell: 走马观花者最长停留时间
    """
    # 1. 如果有传入规则，先更新规则引擎
    if tag_rules:
        valid_updates = {}
        for key, value in tag_rules.items():
            if key in rule_engine.rules and value is not None:
                valid_updates[key] = value

        if valid_updates:
            rule_engine.update_rules(valid_updates)
            print(f"Stage 4 规则已更新: {valid_updates}")

    # 2. 使用更新后的规则引擎生成所有游客画像
    for track_id, metrics in list(hub.stage3_logic.items()):
        trace_id = "unknown"
        if track_id in hub.stage2_perception and hub.stage2_perception[track_id]:
            trace_id = hub.stage2_perception[track_id][-1].trace_id

        profile = label_visitor(track_id, metrics, trace_id)
        hub.set_visitor_profile(profile)
