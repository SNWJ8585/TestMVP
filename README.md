# 智慧展厅全链路行为感知系统

本项目基于 PRD/Tech Spec 需求文档，实现**四阶段全链路**：环境建模 → 感知计算 → 逻辑判定 → 数据资产标签化，并通过 Pydantic 校验与 CentralDataHub 分阶段存储。

## 系统架构（四阶段）

| 阶段 | 核心任务 | 主要模块 |
|------|----------|----------|
| **Stage 1** | 环境建模：比例尺、单应性、展品 2D/3D | `env_engine.py`、`models/hub.py` (ModelingDataset) |
| **Stage 2** | 感知计算：人体位置、追踪、世界坐标、DBSCAN | `perception_engine.py`、YOLOv8+ByteTrack |
| **Stage 3** | 逻辑判定：视线、停留时长、兴趣类型 | `logic_engine.py`（Shapely 射线-多边形）、滑动窗口 |
| **Stage 4** | 标签化：游客属性（资深爱好者/走马观花者等） | `stage4_tags.py`、CentralDataHub.stage4_assets |

- **数据校验**：Pydantic 定义 `ModelingDataset`、`PerceptionData`、`BehaviorMetrics`、`VisitorProfile`。
- **独立存储**：`CentralDataHub` 分桶存储；Stage 3 通过 `hub.get_history(track_id, window=30)` 查询 Stage 2 历史。
- **TraceID**：每阶段输入输出记录 trace_id，支持全链路查找。

## 运行模式

- **展示模式 / 调试模式**：经典人流统计（ROI 人数、停留、DBSCAN、日志导出）。
- **智慧展厅全链路**：启用 Stage 1～4 流水线，实时输出游客标签与行为指标。

## 目录结构

```text
TestMVP/
├── app.py              # Streamlit 入口（三种运行模式）
├── pipeline.py         # 全链路流水线（Stage 1→2→3→4）
├── env_engine.py       # Stage 1：透视矫正、比例尺、展品 2D
├── perception_engine.py # Stage 2：YOLO+ByteTrack、世界坐标、DBSCAN
├── logic_engine.py     # Stage 3：视线判定、停留时长、兴趣类型
├── stage4_tags.py      # Stage 4：游客标签
├── models/
│   ├── __init__.py
│   └── hub.py          # Pydantic 模型 + CentralDataHub
├── processor.py        # 经典检测线程（展示/调试模式）
├── config_manager.py   # ROI 画框（config.json）
├── database.py         # SQLite 存储
├── log_manager.py      # 会话日志 JSON/CSV
├── components/         # UI 组件
├── assets/style.css    # 深色主题
├── config.json         # ROI/展品区域
├── logs/               # 检测会话日志
└── requirements.txt
```

## 环境准备

1. 建议使用 Python 3.9+。
2. 安装依赖（在项目根目录执行）：

```bash
pip install -r requirements.txt
```

3. 下载 YOLOv8n 模型（轻量版本，性能优先）：

```bash
pip install ultralytics  # 若未安装
```

第一次运行 `ultralytics` 时会自动下载 `yolov8n.pt`；或者你也可以手动下载并放到项目根目录。

## 使用步骤

### 1. 准备视频

- 推荐先用监控摄像头录制一小段馆内视频，例如 `sample.mp4`，放在项目根目录。

### 2. 定义 ROI 区域

1. 在命令行进入项目目录：

```bash
cd path/to/museum_tool
```

2. 启动 Streamlit：

```bash
streamlit run app.py
```

3. 在网页左侧输入你的视频路径（例如 `sample.mp4`）。
4. 点击「**🎯 定义/编辑观测区域 (ROI)**」，桌面会弹出 OpenCV 窗口：
   - 鼠标左键按下并拖拽：绘制一个矩形区域；
   - 可以连续绘制多个区域；
   - 区域会自动编号为「区域1、区域2…」；
   - 按 `S` 键：保存并退出；
   - 按 `ESC` 或 `q`：放弃本次修改。

区域数据会以如下格式写入 `config.json`：

```json
{
  "rois": [
    {
      "id": 1,
      "name": "区域1",
      "vertices": [x1, y1, x2, y2]
    }
  ]
}
```

### 3. 调试模式（可选）

1. 在侧边栏选择「**调试模式**」
2. 调整参数：
   - **YOLO 置信度**：控制检测灵敏度（0.1-0.9）
   - **DBSCAN 邻域半径 (ε)**：控制聚类松散度（10-200 像素）
   - **DBSCAN 最小人数 (MinPts)**：定义达到多少人才触发"群体拥挤"报警（2-10）
   - **最小停留时间 (秒)**：只有停留时间 ≥ 此值的点才标记为"有效观看"（0-300秒）
3. 点击「**▶️ 启动检测**」，观察视频画面中聚类圆圈的覆盖情况，调整参数直到准确圈出人群。

### 4. 展示模式

1. 在侧边栏选择「**展示模式**」
2. 点击「**▶️ 启动检测**」：
   - **左侧主区域**：实时叠加检测框、ROI 矩形、人员 ID、朝向箭头、DBSCAN 聚类圆圈
   - **右侧面板**：
     - 当前馆内总人数
     - 各区域当前人数和平均停留时间
     - **异常报警**：实时滚动显示"停留超时"或"高密度聚集"的具体点位坐标
   - **底部**：
     - **历史趋势图**：当日人流变化曲线（Plotly 交互式图表）
     - **热力图**：过去 5 分钟的人流密度热力图

### 5. 历史回放（展示模式）

1. 在「**⏱️ 历史回放**」区域选择开始和结束日期/时间
2. 点击「**🔍 查询历史**」查看该时间段内的历史数据
3. 系统会从数据库调取历史坐标进行离线回放渲染（功能扩展中）

### 6. 停止检测

点击「**⏹️ 停止检测**」即可安全停止后台处理线程。

## 技术要点对应需求

### 核心算法公式

- **朝向计算**：\( \theta = \operatorname{atan2}(y_t - y_{t-1}, x_t - x_{t-1}) \times \frac{180}{\pi} \)
  - 使用当前帧与上一帧的坐标差，计算移动方向角度

- **停留时长**：\( \Delta T_{dwell}(ID_i) = t_{exit}(ID_i) - t_{entry}(ID_i) \)
  - 只有超过最小停留时间阈值 \( T_{min} \) 的记录才会写入数据库

- **聚类密度**：基于 \( dist(p,q) = \sqrt{(x_p-x_q)^2 + (y_p-y_q)^2} \) 判断邻域 \( N_{\epsilon}(p) \)
  - 若点集数量 \( \ge MinPts \)，则标记为核心人群
  - 可调参数：`eps`（邻域半径）、`min_samples`（最小人数）

### 性能优化

- **性能优先**
  - 使用 `cv2.VideoCapture` 直接读取视频帧；
  - 使用 `yolov8n.pt` 轻量模型，避免普通台式机卡顿；
  - 处理线程中轻量 `sleep(0.001)`，避免 CPU 跑满。

- **异步处理**
  - `Processor` 继承自 `threading.Thread`，在后台线程中执行 YOLO 检测与跟踪；
  - 和前端通过 `FrameQueue`（基于 `queue.Queue`）通信，防止 Streamlit UI 卡死。

### 数据库设计

- **`raw_events`**：记录每帧、每人位置及关联信息
  - 新增 `timestamp`（REAL）：用于时间范围查询和视频回溯
  - 新增 `frame_id`（INTEGER）：用于视频跳转 `cv2.VideoCapture.set(cv2.CAP_PROP_POS_FRAMES, frame_id)`

- **`stay_records`**：记录 ID、进入时间、离开时间、所属区域 ID、停留时长

- **`daily_stats`**：提供 `aggregate_daily_stats(date_str)` 对指定日期进行 `GROUP BY` 聚合

### UI 功能

- **调试模式**：实时调整算法参数，观察聚类效果
- **展示模式**：
  - 实时区域状态：每个 area 的当前人数、平均停留时间
  - 异常点报警：实时滚动显示"停留超时"或"高密度聚集"的具体点位坐标
  - 热力图：使用 Plotly 绘制过去 5 分钟的坐标累计密度
  - 历史趋势图：Plotly 交互式图表展示当日人流变化
  - 深色主题：自定义 CSS，符合美术馆数据大屏的视觉风格

### 视频回溯功能

- 通过时间轴选择器选择特定时间段（如"昨日 14:00-15:00"）
- 系统从数据库调取历史坐标进行离线回放渲染或热力图生成
- 使用 `cv2.VideoCapture.set(cv2.CAP_PROP_POS_FRAMES, frame_id)` 实现视频快速跳转

如需进一步扩展（多摄像头、多楼层、复杂报表等），可以在现有结构上继续迭代。若你希望，我也可以帮你加上定时任务，自动每天对 `stay_records` 做聚合并写入 `daily_stats`。 

