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
# 导入 Stage 4 逻辑
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
    if os.path.exists(css_path) :
        with open(css_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()
init_db()

# 初始化 session state
if "processor" not in st.session_state:
    st.session_state.processor = None
if "pipeline" not in st.session_state:
    st.session_state.pipeline = None

# --- 侧边栏：配置区 ---
st.sidebar.title("⚙️ 系统配置")

# 1. 视频源设置
video_source = st.sidebar.selectbox("选择视频源", ["demo.mp4", "摄像头 (0)"])
video_path = 0 if "摄像头" in video_source else video_source

# 2. 模式切换
mode = st.sidebar.radio("运行模式", ["智慧展厅全链路", "基础检测调试"])

# 3. 标签规则配置 (核心改动)
st.sidebar.subheader("🏷️ 标签触发规则")
with st.sidebar.expander("点击调整标签阈值", expanded=True):
    expert_min_dwell = st.slider("资深: 最短停留(秒)", 10, 300, 60)
    expert_max_focus = st.slider("资深: 最大视线偏离", 5, 50, 15)
    
    normal_min_dwell = st.slider("一般: 最短停留(秒)", 5, 100, 20)
    normal_max_focus = st.slider("一般: 最大视线偏离", 10, 90, 30)
    
    brief_min_dwell = st.slider("短暂: 最短停留(秒)", 3, 30, 10)
    casual_max_dwell = st.slider("走马观花: 最长停留(秒)", 1, 10, 3)

    # 逻辑校验：防止设置冲突
    if expert_min_dwell <= normal_min_dwell:
        st.error("❌ 逻辑冲突：资深时间应 > 一般时间")
    
    # 实时同步到规则引擎
    current_tag_rules = {
        "expert_min_dwell": expert_min_dwell,
        "expert_max_focus": expert_max_focus,
        "normal_min_dwell": normal_min_dwell,
        "normal_max_focus": normal_max_focus,
        "brief_min_dwell": brief_min_dwell,
        "casual_max_dwell": casual_max_dwell
    }
    update_thresholds_from_ui(current_tag_rules)

# 4. 实时标签调试区 (性能优化改动)
if st.sidebar.checkbox("开启模拟规则测试"):
    st.sidebar.markdown("---")
    st.sidebar.caption("💡 基于当前滑块数值的模拟结果")
    test_data = [
        {"ID": "A", "停留(秒)": expert_min_dwell + 5, "视线偏离": expert_max_focus - 2},
        {"ID": "B", "停留(秒)": normal_min_dwell + 5, "视线偏离": normal_max_focus - 5},
        {"ID": "C", "停留(秒)": casual_max_dwell - 1, "视线偏离": 80}
    ]
    test_results = []
    for data in test_data:
        label = rule_engine.generate_label(data["停留(秒)"], data["视线偏离"])
        test_results.append({"访客": data["ID"], "结果标签": label})
    st.sidebar.table(pd.DataFrame(test_results))

# --- 主界面 ---
st.title("🏛️ 智慧展厅行为感知监控后台")

col1, col2 = st.columns([2, 1])

with col1:
    video_placeholder = st.empty()
    start = st.button("🚀 启动检测系统", use_container_width=True)
    stop = st.button("🛑 停止检测", use_container_width=True)

    if start:
        if mode == "智慧展厅全链路":
            st.session_state.pipeline = FullChainPipeline(video_path, FrameQueue())
            st.session_state.pipeline.start()
        else:
            st.session_state.processor = Processor(video_path, FrameQueue())
            st.session_state.processor.start()
        st.success(f"已启动模式: {mode}")

    if stop:
        if st.session_state.pipeline: st.session_state.pipeline.stop()
        if st.session_state.processor: st.session_state.processor.stop()
        st.warning("系统已停止")

with col2:
    st.subheader("📊 实时统计")
    stat_placeholder = st.empty()
    
    st.subheader("⚠️ 异常预警")
    anomaly_placeholder = st.empty()

# --- 视频流渲染循环 ---
current_engine = st.session_state.pipeline if mode == "智慧展厅全链路" else st.session_state.processor

if current_engine and current_engine.running:
    while current_engine.running:
        frame_data = current_engine.frame_queue.get_latest()
        if frame_data:
            frame, info = frame_data
            # 1. 渲染视频
            video_placeholder.image(frame, channels="BGR", use_container_width=True)
            
            # 2. 更新统计 (Stage 4 数据展示)
            labels = info.get("visitor_labels", {})
            # 使用列表推导式分类，效率更高
            experts = [tid for tid, L in labels.items() if L == "资深爱好者"]
            normals = [tid for tid, L in labels.items() if L == "一般观众"]
            
            stat_markdown = f"""
            **当前总人数**: {info.get('total_people', 0)}  
            ---
            🌟 **资深爱好者 ({len(experts)}人)**: {', '.join(map(str, experts[:3]))}  
            👤 **一般观众 ({len(normals)}人)**: {', '.join(map(str, normals[:3]))}  
            ⏱️ **更新时间**: {info.get('timestamp')}
            """
            stat_placeholder.markdown(stat_markdown)
        else:
            import time
            time.sleep(0.01)