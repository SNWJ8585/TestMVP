import os
from datetime import datetime, timedelta
from typing import Optional

import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config_manager import define_rois
from database import (
    get_anomalies,
    get_heatmap_data,
    get_historical_events,
    get_today_flow_by_time,
    init_db,
)
from log_manager import LogManager
from pipeline import FullChainPipeline
from processor import FrameQueue, Processor
from stage4_tags import update_thresholds_from_ui, rule_engine

# 页面配置
st.set_page_config(
    page_title="Museum-Flow-AI",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 加载自定义 CSS
def load_css():
    css_path = os.path.join("assets", "style.css")
    if os.path.exists(css_path):
        with open(css_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


load_css()

# 初始化数据库
init_db()

# 初始化 session state
if "processor" not in st.session_state:
    st.session_state.processor = None
    st.session_state.pipeline = None  # 智慧展厅全链路流水线
    st.session_state.frame_queue = FrameQueue()
    st.session_state.mode = "展示模式"  # 或 "调试模式" / "智慧展厅全链路"
    st.session_state.log_manager = LogManager()


def ensure_db():
    init_db()


def run_roi_config(video_path: str, config_path: str = "config.json"):
    define_rois(video_path=video_path, config_path=config_path)


def main():
    st.title("🎨 智慧展厅全链路行为感知系统")

    # 侧边栏：模式选择和基础配置
    st.sidebar.header("⚙️ 系统设置")
    mode = st.sidebar.radio(
        "运行模式",
        ["展示模式", "调试模式", "智慧展厅全链路"],
        index=0,
        help="智慧展厅全链路：Stage1~4 流水线 + Pydantic 校验 + 游客标签",
    )

    st.sidebar.markdown("---")
    st.sidebar.header("📁 基础配置")
    video_path = st.sidebar.text_input("视频路径 / 摄像头索引", value="sample.mp4")
    model_path = st.sidebar.text_input("YOLO 模型路径", value="yolov8n.pt")

    if st.sidebar.button("🎯 定义/编辑观测区域 (ROI)"):
        if not os.path.exists(video_path) and not video_path.isdigit():
            st.error("视频路径不存在，请检查。")
        else:
            if video_path.isdigit():
                st.warning("当前 ROI 工具仅支持本地视频文件，请先录制一小段视频用于标注。")
            else:
                st.info("已在桌面弹出 OpenCV 窗口，请在窗口中用鼠标拖拽矩形，按 S 保存并退出。")
                run_roi_config(video_path)

    # 调试模式：参数调整面板
    if mode == "调试模式":
        st.sidebar.markdown("---")
        st.sidebar.header("🔧 调试参数")
        confidence = st.sidebar.slider("YOLO 置信度", 0.1, 0.9, 0.25, 0.05)
        dbscan_eps = st.sidebar.slider("DBSCAN 邻域半径 (ε)", 10.0, 200.0, 50.0, 5.0)
        dbscan_min_samples = st.sidebar.slider("DBSCAN 最小人数 (MinPts)", 2, 10, 3, 1)
        min_dwell_time = st.sidebar.slider("最小停留时间 (秒)", 0.0, 300.0, 0.0, 5.0)
    else:
        # 展示模式：使用默认参数
        confidence = 0.25
        dbscan_eps = 50.0
        dbscan_min_samples = 3
        min_dwell_time = 0.0

    # ========== Stage 4: 游客标签规则编辑器 ==========
    st.sidebar.markdown("---")
    st.sidebar.header("Stage 4: 游客标签规则")

    # ===== 规则1：资深爱好者 =====
    with st.sidebar.expander("资深爱好者判定规则", expanded=True):
        st.sidebar.caption("条件：长时间停留 + 高度专注")

        col1, col2 = st.sidebar.columns(2)
        with col1:
            expert_min_dwell = st.slider(
                "最短停留时间（秒）",
                min_value=30, max_value=300, value=60, step=5,
                key="expert_dwell",
                help="停留时间超过这个阈值的游客才可能被标记为资深爱好者"
            )
        with col2:
            expert_max_focus = st.slider(
                "最大视线偏离值",
                min_value=5, max_value=30, value=15, step=5,
                key="expert_focus",
                help="视线偏离值小于这个数值才算专注观看（数值越小越专注）"
            )

        st.sidebar.info(
            f"当前规则：停留 ≥ {expert_min_dwell}秒 且 视线偏离 ≤ {expert_max_focus}"
        )

    # ===== 规则2：一般观众 =====
    with st.sidebar.expander("一般观众判定规则", expanded=True):
        st.sidebar.caption("条件：中等停留时间 + 一般专注")

        col1, col2 = st.sidebar.columns(2)
        with col1:
            normal_min_dwell = st.slider(
                "最短停留时间（秒）",
                min_value=10, max_value=120, value=20, step=5,
                key="normal_dwell",
                help="停留时间超过这个阈值的游客才可能被标记为一般观众"
            )
        with col2:
            normal_max_focus = st.slider(
                "最大视线偏离值",
                min_value=10, max_value=50, value=30, step=5,
                key="normal_focus",
                help="一般观众的视线要求比资深爱好者宽松"
            )

        st.sidebar.info(
            f"当前规则：停留 ≥ {normal_min_dwell}秒 且 视线偏离 ≤ {normal_max_focus}"
        )

    # ===== 规则3：短暂驻足 =====
    with st.sidebar.expander("短暂驻足判定规则", expanded=True):
        st.sidebar.caption("条件：停留时间较短，不看视线")

        brief_min_dwell = st.slider(
            "最短停留时间（秒）",
            min_value=3, max_value=30, value=5, step=1,
            key="brief_dwell",
            help="停留超过这个时间但不够一般观众标准的，标记为短暂驻足"
        )

        st.sidebar.info(f"当前规则：停留 ≥ {brief_min_dwell}秒（不看视线）")

    # ===== 规则4：走马观花者 =====
    with st.sidebar.expander("走马观花者判定规则", expanded=True):
        st.sidebar.caption("条件：停留时间极短")

        casual_max_dwell = st.slider(
            "最长停留时间（秒）",
            min_value=1, max_value=20, value=3, step=1,
            key="casual_dwell",
            help="停留时间少于这个阈值的游客，直接标记为走马观花者"
        )

        st.sidebar.info(f"当前规则：停留 ≤ {casual_max_dwell}秒")

    # ===== 规则预览 =====
    st.sidebar.markdown("---")
    st.sidebar.subheader("当前完整规则链")

    # 获取规则摘要
    rules_summary = rule_engine.get_rule_summary()
    st.sidebar.markdown(f"""
       ```
       资深爱好者: {rules_summary['expert']}
       一般观众:   {rules_summary['normal']}  
       短暂驻足:   {rules_summary['brief']}
       走马观花者: {rules_summary['casual']}
       ```
       """)

    # ===== 联动核心：把所有滑块值传给规则引擎 =====
    update_thresholds_from_ui(
        expert_dwell=expert_min_dwell,
        expert_focus=expert_max_focus,
        normal_dwell=normal_min_dwell,
        normal_focus=normal_max_focus,
        brief_dwell=brief_min_dwell,
        casual_dwell=casual_max_dwell
    )

    # 显示规则更新时间
    if rule_engine.last_updated:
        st.sidebar.caption(f"规则生效: {rule_engine.last_updated.strftime('%H:%M:%S')}")

    # ===== 实时标签测试区 =====
    st.sidebar.markdown("---")
    st.sidebar.header("实时标签测试")

    # 创建测试数据
    import pandas as pd
    import numpy as np

    # 生成各种类型的游客测试数据
    test_data = [
        {"游客ID": "#1", "停留(秒)": 120, "视线偏离": 10, "描述": "深度爱好者"},
        {"游客ID": "#2", "停留(秒)": 80, "视线偏离": 12, "描述": "爱好者"},
        {"游客ID": "#3", "停留(秒)": 45, "视线偏离": 18, "描述": "普通观众"},
        {"游客ID": "#4", "停留(秒)": 30, "视线偏离": 25, "描述": "随便看看"},
        {"游客ID": "#5", "停留(秒)": 15, "视线偏离": 35, "描述": "路过"},
        {"游客ID": "#6", "停留(秒)": 8, "视线偏离": 40, "描述": "快步走过"},
        {"游客ID": "#7", "停留(秒)": 2, "视线偏离": 45, "描述": "匆匆一瞥"},
    ]

    # 为每个游客生成标签
    test_results = []
    for data in test_data:
        label = rule_engine.generate_label(
            dwell_time=data["停留(秒)"],
            focus_index=data["视线偏离"]  # 注意：这里是 focus_index，不是 gaze_angle！
        )
        test_results.append({
            "ID": data["游客ID"],
            "停留": f"{data['停留(秒)']}s",
            "视线偏离": f"{data['视线偏离']}",
            "标签": label
        })

    # 显示测试结果表格
    st.sidebar.dataframe(
        pd.DataFrame(test_results),
        use_container_width=True,
        hide_index=True
    )

    # 统计各类标签数量
    df_test = pd.DataFrame(test_results)
    expert_count = len(df_test[df_test["标签"].str.contains("资深爱好者")])
    normal_count = len(df_test[df_test["标签"].str.contains("一般观众")])
    brief_count = len(df_test[df_test["标签"].str.contains("短暂驻足")])
    casual_count = len(df_test[df_test["标签"].str.contains("走马观花者")])

    st.sidebar.markdown(f"""
       **测试结果统计**
       - 资深爱好者: {expert_count}人
       - 一般观众: {normal_count}人  
       - 短暂驻足: {brief_count}人
       - 走马观花者: {casual_count}人
       """)

    # 添加视线偏离值说明
    st.sidebar.caption("""
       **ℹ视线偏离值说明**
       - 0-15: 非常专注
       - 16-30: 一般专注  
       - 31+: 分心/随意
       """)
    # ========== Stage 4 规则编辑器结束 ==========
    st.sidebar.markdown("---")
    st.sidebar.header("🎮 检测控制")

    # 主界面布局
    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.subheader("📹 实时视频流")
        frame_placeholder = st.empty()

        # 视频回溯功能（仅在展示模式）
        if mode == "展示模式":
            st.markdown("---")
            st.subheader("⏱️ 历史回放")
            col_time1, col_time2, col_time3 = st.columns([2, 2, 1])
            with col_time1:
                start_date = st.date_input("开始日期", value=datetime.now().date())
                start_time = st.time_input("开始时间", value=datetime.now().time())
            with col_time2:
                end_date = st.date_input("结束日期", value=datetime.now().date())
                end_time = st.time_input("结束时间", value=datetime.now().time())
            with col_time3:
                if st.button("🔍 查询历史"):
                    start_dt = datetime.combine(start_date, start_time)
                    end_dt = datetime.combine(end_date, end_time)
                    historical_data = get_historical_events(start_dt, end_dt)
                    if historical_data:
                        st.success(f"找到 {len(historical_data)} 条历史记录")
                        # 这里可以实现历史回放逻辑
                    else:
                        st.warning("该时间段内无数据")

    with col_right:
        st.subheader("📊 实时数据面板")
        total_people_placeholder = st.metric("当前馆内总人数", 0)
        roi_text_placeholder = st.empty()
        avg_dwell_placeholder = st.empty()

        # 异常报警面板
        st.markdown("---")
        st.subheader("⚠️ 异常报警")
        anomaly_placeholder = st.empty()
        
        # 导出报告按钮
        st.markdown("---")
        st.subheader("📥 数据导出")
        export_col1, export_col2 = st.columns([2, 1])
        with export_col1:
            if st.button("📊 导出当前检测报告", use_container_width=True):
                if st.session_state.log_manager and st.session_state.log_manager.session_data:
                    try:
                        json_path, csv_path = st.session_state.log_manager.export_session()
                        st.success(f"✅ 报告已导出！")
                        st.info(f"JSON: `{json_path}`\n\nCSV: `{csv_path}`")
                        
                        # 提供下载链接
                        with open(json_path, "rb") as f:
                            st.download_button(
                                "⬇️ 下载 JSON 文件",
                                f.read(),
                                file_name=json_path.split("/")[-1],
                                mime="application/json",
                            )
                        with open(csv_path, "rb") as f:
                            st.download_button(
                                "⬇️ 下载 CSV 文件",
                                f.read(),
                                file_name=csv_path.split("/")[-1],
                                mime="text/csv",
                            )
                    except Exception as e:
                        st.error(f"导出失败: {e}")
                else:
                    st.warning("⚠️ 当前没有检测数据可导出，请先启动检测。")
        
        # 显示会话统计
        if st.session_state.log_manager:
            summary = st.session_state.log_manager.get_session_summary()
            if summary["total_records"] > 0:
                st.markdown("---")
                st.subheader("📈 会话统计")
                st.metric("总记录数", summary["total_records"])
                st.metric("唯一访客数", summary["unique_visitors"])
                st.metric("检测区域数", len(summary["unique_areas"]))
                st.metric("聚类事件数", summary["total_clusters"])

    st.markdown("---")

    # 底部：历史趋势图和热力图
    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        st.subheader("📈 历史趋势图（当日人流曲线）")
        chart_placeholder = st.empty()

    with col_chart2:
        st.subheader("🔥 热力图（过去 5 分钟）")
        heatmap_placeholder = st.empty()

    # 启动/停止检测
    start = st.sidebar.button("▶️ 启动检测")
    stop = st.sidebar.button("⏹️ 停止检测")

    if start and st.session_state.processor is None and st.session_state.pipeline is None:
        if not os.path.exists(video_path) and not video_path.isdigit():
            st.error("视频路径不存在或摄像头索引非法。")
        else:
            # ===== 收集当前的标签规则 =====
            current_tag_rules = {
                "expert_min_dwell": expert_min_dwell,
                "expert_max_focus": expert_max_focus,
                "normal_min_dwell": normal_min_dwell,
                "normal_max_focus": normal_max_focus,
                "brief_min_dwell": brief_min_dwell,
                "casual_max_dwell": casual_max_dwell
            }

            if mode == "智慧展厅全链路":
                st.session_state.pipeline = FullChainPipeline(
                    video_path=video_path,
                    frame_queue=st.session_state.frame_queue,
                    config_path="config.json",
                    model_path=model_path if os.path.exists(model_path) else "yolov8n.pt",
                    confidence=confidence,
                    dbscan_eps=dbscan_eps,
                    dbscan_min_samples=dbscan_min_samples,
                    tag_rules=current_tag_rules
                )
                st.session_state.pipeline.start()
                st.success("智慧展厅全链路已启动（Stage 1~4 + 游客标签）")
                st.info(f"当前标签规则: 资深≥{expert_min_dwell}s/≤{expert_max_focus}°, 一般≥{normal_min_dwell}s/≤{normal_max_focus}°, 驻足≥{brief_min_dwell}s, 走马≤{casual_max_dwell}s")
            else:
                # 启动新的日志会话
                st.session_state.log_manager.start_session()
                if mode == "调试模式":
                    st.session_state.processor = Processor(
                        video_path=video_path,
                        frame_queue=st.session_state.frame_queue,
                        model_path=model_path if os.path.exists(model_path) else "yolov8n.pt",
                        confidence=confidence,
                        dbscan_eps=dbscan_eps,
                        dbscan_min_samples=dbscan_min_samples,
                        min_dwell_time=min_dwell_time,
                        log_manager=st.session_state.log_manager,
                    )
                else:
                    st.session_state.processor = Processor(
                        video_path=video_path,
                        frame_queue=st.session_state.frame_queue,
                        model_path=model_path if os.path.exists(model_path) else "yolov8n.pt",
                        log_manager=st.session_state.log_manager,
                    )
                st.session_state.processor.start()
                st.success("检测已启动")

    if stop and (st.session_state.processor is not None or st.session_state.pipeline is not None):
        if st.session_state.pipeline is not None:
            st.session_state.pipeline.stop()
            st.session_state.pipeline = None
        if st.session_state.processor is not None:
            st.session_state.processor.stop()
            import time
            time.sleep(0.5)
            if st.session_state.log_manager and st.session_state.log_manager.session_data:
                try:
                    json_path, csv_path = st.session_state.log_manager.export_session()
                    st.success(f"✅ 检测已停止，日志已自动导出！")
                    st.info(f"📁 JSON: `{json_path}`\n\n📁 CSV: `{csv_path}`")
                except Exception as e:
                    st.warning(f"⚠️ 日志导出失败: {e}")
            st.session_state.processor = None
        st.info("检测已停止")

    # 主循环：不断从队列拿最新帧并渲染
    latest_frame: Optional[np.ndarray] = None
    latest_info = None

    frame_data = st.session_state.frame_queue.get_latest()
    if frame_data is not None:
        latest_frame, latest_info = frame_data

    if latest_frame is not None:
        # BGR -> RGB
        rgb = cv2.cvtColor(latest_frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(rgb, channels="RGB", use_column_width=True)

    if latest_info is not None:
        total_people = latest_info.get("total_people", 0)
        total_people_placeholder.metric("当前馆内总人数", total_people)

        roi_counts = latest_info.get("roi_counts", {})
        roi_avg_dwell = latest_info.get("roi_avg_dwell", {})
        lines = []
        for roi_id, cnt in roi_counts.items():
            avg_dwell = roi_avg_dwell.get(roi_id, 0.0)
            lines.append(f"**区域 {roi_id}**: {cnt} 人 | 平均停留: {avg_dwell:.1f}秒")
        visitor_labels = latest_info.get("visitor_labels", {})
        if visitor_labels:
            lines.append("**游客标签 (Stage 4)**")
            # 显示当前生效的规则
            rules_summary = rule_engine.get_rule_summary()
            lines.append(f"  *当前规则: {rules_summary['expert']}*")
            # 按标签类型分组显示，更直观
            expert_ids = []
            normal_ids = []
            brief_ids = []
            casual_ids = []
            other_ids = []
            for tid, label in visitor_labels.items():
                if"资深爱好者" in label:
                    expert_ids.append(f"ID {tid}")
                elif "一般观众" in label:
                    normal_ids.append(f"ID {tid}")
                elif "短暂驻足" in label:
                    brief_ids.append(f"ID {tid}")
                elif "走马观花者" in label:
                    casual_ids.append(f"ID {tid}")
                else:
                    other_ids.append(f"ID {tid}")
            if expert_ids:
                lines.append(f"  资深爱好者 ({len(expert_ids)}人): {', '.join(expert_ids[:3])}{'...' if len(expert_ids) > 3 else ''}")
            if normal_ids:
                lines.append(f"  一般观众 ({len(normal_ids)}人): {', '.join(normal_ids[:3])}{'...' if len(normal_ids) > 3 else ''}")
            if brief_ids:
                lines.append(f"  短暂驻足 ({len(brief_ids)}人): {', '.join(brief_ids[:3])}{'...' if len(brief_ids) > 3 else ''}")
            if casual_ids:
                lines.append(f"  走马观花者 ({len(casual_ids)}人): {', '.join(casual_ids[:3])}{'...' if len(casual_ids) > 3 else ''}")
            if other_ids:
                lines.append(f"  其他 ({len(other_ids)}人): {', '.join(other_ids[:3])}{'...' if len(other_ids) > 3 else ''}")
        if lines:
            roi_text_placeholder.markdown("  \n".join(lines))
        else:
            roi_text_placeholder.markdown("暂无区域数据")

        # 异常报警
        if mode == "展示模式":
            long_stays, clusters = get_anomalies(min_dwell_time, dbscan_min_samples)
            anomaly_text = []
            if long_stays:
                anomaly_text.append("**停留超时**:")
                for stay in long_stays[:5]:  # 只显示前5个
                    person_id, roi_id, enter_time, leave_time, total_time, x, y = stay
                    anomaly_text.append(
                        f"  - ID {person_id} 在区域 {roi_id} 停留 {total_time:.1f}秒 (坐标: {x:.0f}, {y:.0f})"
                    )
            if clusters:
                anomaly_text.append("**高密度聚集**:")
                for cluster in clusters[:5]:  # 只显示前5个
                    cluster_id, cnt, avg_x, avg_y, last_ts = cluster
                    anomaly_text.append(
                        f"  - 聚类 {cluster_id}: {cnt} 人 (坐标: {avg_x:.0f}, {avg_y:.0f})"
                    )
            if anomaly_text:
                anomaly_placeholder.markdown("  \n".join(anomaly_text))
            else:
                anomaly_placeholder.markdown("✅ 无异常")

    # 历史趋势图
    times, counts = get_today_flow_by_time()
    if times:
        df_flow = pd.DataFrame({"时间": times, "人数": counts})
        fig_flow = px.line(df_flow, x="时间", y="人数", title="当日人流变化")
        fig_flow.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#ecf0f1",
        )
        chart_placeholder.plotly_chart(fig_flow, use_container_width=True)
    else:
        chart_placeholder.write("暂无今日历史数据")

    # 热力图
    heatmap_data = get_heatmap_data(minutes=5)
    if heatmap_data:
        df_heat = pd.DataFrame(heatmap_data, columns=["x", "y", "roi_id"])
        fig_heat = px.density_heatmap(
            df_heat,
            x="x",
            y="y",
            nbinsx=30,
            nbinsy=30,
            title="人流密度热力图",
            color_continuous_scale="Viridis",
        )
        fig_heat.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#ecf0f1",
        )
        heatmap_placeholder.plotly_chart(fig_heat, use_container_width=True)
    else:
        heatmap_placeholder.write("暂无热力图数据（需要运行检测）")


if __name__ == "__main__":
    main()
