import os
from typing import Optional

import cv2
import pandas as pd
import numpy as np
import streamlit as st

from config_manager import define_rois
from database import get_today_flow_by_time, init_db
from processor import FrameQueue, Processor


def ensure_db():
  init_db()


def run_roi_config(video_path: str, config_path: str = "config.json"):
  define_rois(video_path=video_path, config_path=config_path)


def main():
  st.set_page_config(page_title="Museum-Flow-AI", layout="wide")
  st.title("Museum-Flow-AI 美术馆人流检测与可视化工具")

  ensure_db()

  st.sidebar.header("基础配置")
  video_path = st.sidebar.text_input("视频路径 / 摄像头索引", value="sample.mp4")
  model_path = st.sidebar.text_input("YOLO 模型路径", value="yolov8n.pt")

  if st.sidebar.button("定义/编辑观测区域 (ROI)"):
    if not os.path.exists(video_path) and not video_path.isdigit():
      st.error("视频路径不存在，请检查。")
    else:
      if video_path.isdigit():
        st.warning("当前 ROI 工具仅支持本地视频文件，请先录制一小段视频用于标注。")
      else:
        st.info("已在桌面弹出 OpenCV 窗口，请在窗口中用鼠标拖拽矩形，按 S 保存并退出。")
        run_roi_config(video_path)

  st.sidebar.markdown("---")
  st.sidebar.header("检测控制")

  if "processor" not in st.session_state:
    st.session_state.processor = None
    st.session_state.frame_queue = FrameQueue()

  col_left, col_right = st.columns([2, 1])

  with col_left:
    st.subheader("实时视频流")
    frame_placeholder = st.empty()

  with col_right:
    st.subheader("实时数据面板")
    total_people_placeholder = st.metric("当前馆内总人数", 0)
    roi_text_placeholder = st.empty()

  st.markdown("---")
  st.subheader("历史趋势图（当日人流曲线）")
  chart_placeholder = st.empty()

  start = st.sidebar.button("启动检测")
  stop = st.sidebar.button("停止检测")

  if start and st.session_state.processor is None:
    if not os.path.exists(video_path) and not video_path.isdigit():
      st.error("视频路径不存在或摄像头索引非法。")
    elif not os.path.exists(model_path):
      st.warning("未找到 YOLO 模型文件，将尝试默认路径 'yolov8n.pt'，请确保本地已下载。")
      st.session_state.processor = Processor(
        video_path=video_path,
        frame_queue=st.session_state.frame_queue,
      )
      st.session_state.processor.start()
    else:
      st.session_state.processor = Processor(
        video_path=video_path,
        frame_queue=st.session_state.frame_queue,
        model_path=model_path,
      )
      st.session_state.processor.start()

  if stop and st.session_state.processor is not None:
    st.session_state.processor.stop()
    st.session_state.processor = None

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
    total_people_placeholder = st.metric("当前馆内总人数", total_people)

    roi_counts = latest_info.get("roi_counts", {})
    lines = []
    for roi_id, cnt in roi_counts.items():
      lines.append(f"区域 {roi_id}: {cnt} 人")
    if lines:
      roi_text_placeholder.markdown("  \n".join(lines))
    else:
      roi_text_placeholder.markdown("暂无区域数据")

  # 历史趋势图
  times, counts = get_today_flow_by_time()
  if times:
    # 将时间与人数封装为 DataFrame，使用列名作为 x 轴
    df = pd.DataFrame({"时间": times, "人数": counts})
    chart_placeholder.line_chart(df, x="时间", y="人数")
  else:
    chart_placeholder.write("暂无今日历史数据")


if __name__ == "__main__":
  main()

