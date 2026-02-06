import json
from dataclasses import dataclass, asdict
from typing import List, Tuple

import cv2


@dataclass
class ROI:
  """矩形区域定义."""

  id: int
  name: str
  vertices: Tuple[int, int, int, int]  # x1, y1, x2, y2


class ROIConfigManager:
  """使用 OpenCV 在视频第一帧上手动画框，保存到 config.json。"""

  def __init__(self, config_path: str = "config.json") -> None:
    self.config_path = config_path
    self.rois: List[ROI] = []
    self._drawing = False
    self._start_point = (0, 0)
    self._end_point = (0, 0)
    self._current_frame = None
    self._frame_for_draw = None
    self._window_name = "ROI_Config_Window"  # 统一的窗口名称

  def _mouse_callback(self, event, x, y, flags, param):
    # 鼠标按下：开始绘制，并立即显示一个小矩形预览
    if event == cv2.EVENT_LBUTTONDOWN:
      self._drawing = True
      self._start_point = (x, y)
      self._end_point = (x, y)
      self._update_preview()
      try:
        cv2.imshow(self._window_name, self._frame_for_draw)
      except:
        pass
    # 按住左键拖动时，持续更新预览矩形
    elif event == cv2.EVENT_MOUSEMOVE:
      if self._drawing or (flags & cv2.EVENT_FLAG_LBUTTON):
        self._end_point = (x, y)
        self._update_preview()
        try:
          cv2.imshow(self._window_name, self._frame_for_draw)
        except:
          pass
    # 松开左键：确认一个 ROI
    elif event == cv2.EVENT_LBUTTONUP:
      self._drawing = False
      self._end_point = (x, y)
      x1, y1 = self._start_point
      x2, y2 = self._end_point
      x_min, y_min = min(x1, x2), min(y1, y2)
      x_max, y_max = max(x1, x2), max(y1, y2)
      if abs(x_max - x_min) > 10 and abs(y_max - y_min) > 10:
        roi_id = len(self.rois) + 1
        name = f"区域{roi_id}"
        self.rois.append(ROI(id=roi_id, name=name, vertices=(x_min, y_min, x_max, y_max)))
        self._draw_all_rois()
        try:
          cv2.imshow(self._window_name, self._frame_for_draw)
        except:
          pass
        print(f"已添加区域 {roi_id}: ({x_min}, {y_min}) -> ({x_max}, {y_max})")

  def _draw_all_rois(self):
    # 若还没有缓冲图像，先复制一份原始帧
    if self._current_frame is None:
      return
    self._frame_for_draw = self._current_frame.copy()
    for roi in self.rois:
      x1, y1, x2, y2 = roi.vertices
      # 绿色粗边框，便于看清
      cv2.rectangle(self._frame_for_draw, (x1, y1), (x2, y2), (0, 255, 0), 3)
      cv2.putText(
        self._frame_for_draw,
        f"{roi.id}:{roi.name}",
        (x1, max(0, y1 - 5)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
      )

  def _update_preview(self):
    # 重绘所有已确认 ROI，再叠加当前正在拖拽的预览矩形
    if self._current_frame is None:
      return
    self._draw_all_rois()
    if self._frame_for_draw is None:
      return
    # 绘制当前正在拖拽的预览矩形（亮黄色，粗边框）
    x1, y1 = self._start_point
    x2, y2 = self._end_point
    cv2.rectangle(self._frame_for_draw, (x1, y1), (x2, y2), (0, 255, 255), 3)
    # 显示尺寸信息
    w_box = abs(x2 - x1)
    h_box = abs(y2 - y1)
    cv2.putText(
      self._frame_for_draw,
      f"{w_box}x{h_box}",
      (min(x1, x2) + 5, min(y1, y2) + 20),
      cv2.FONT_HERSHEY_SIMPLEX,
      0.5,
      (0, 255, 255),
      2,
      cv2.LINE_AA,
    )

  def save(self):
    data = {
      "rois": [
        {
          "id": roi.id,
          "name": roi.name,
          "vertices": list(roi.vertices),
        }
        for roi in self.rois
      ]
    }
    with open(self.config_path, "w", encoding="utf-8") as f:
      json.dump(data, f, ensure_ascii=False, indent=2)

  def load(self):
    try:
      with open(self.config_path, "r", encoding="utf-8") as f:
        data = json.load(f)
      self.rois = [
        ROI(id=item["id"], name=item["name"], vertices=tuple(item["vertices"]))
        for item in data.get("rois", [])
      ]
    except FileNotFoundError:
      self.rois = []

  def define_rois_on_first_frame(self, video_path: str):
    """自定义鼠标交互：左键拖拽画矩形，按 S 保存并退出，ESC 取消。"""
    cap = cv2.VideoCapture(video_path)
    ok, frame = cap.read()
    cap.release()
    if not ok:
      raise RuntimeError(f"无法读取视频: {video_path}")

    self._current_frame = frame
    self._frame_for_draw = frame.copy()
    self._draw_all_rois()

    # 先尝试销毁可能存在的旧窗口
    try:
      cv2.destroyWindow(self._window_name)
    except:
      pass

    # 创建新窗口
    cv2.namedWindow(self._window_name, cv2.WINDOW_NORMAL)
    # 根据视频大小调整窗口尺寸，避免窗口太小看不清
    h, w = frame.shape[:2]
    try:
      cv2.resizeWindow(self._window_name, min(w, 1280), min(h, 720))
    except:
      pass
    
    cv2.setMouseCallback(self._window_name, self._mouse_callback)

    print("\n" + "="*60)
    print("提示：在 OpenCV 窗口中")
    print("  - 左键拖拽：绘制矩形区域（应该能看到黄色矩形框）")
    print("  - 松开左键：确认区域（变成绿色）")
    print("  - 按 S 键：保存并退出")
    print("  - 按 ESC 或 Q：取消退出")
    print("="*60 + "\n")

    # 首次显示
    cv2.imshow(self._window_name, self._frame_for_draw)
    
    window_closed = False
    while True:
      # 检查窗口是否还存在
      try:
        # 持续刷新显示，确保矩形可见
        cv2.imshow(self._window_name, self._frame_for_draw)
        key = cv2.waitKey(30) & 0xFF
        if key == 27 or key == ord("q") or key == ord("Q"):  # ESC 或 q 取消
          print("已取消，未保存。")
          break
        if key == ord("s") or key == ord("S"):  # S 保存
          if len(self.rois) > 0:
            self.save()
            print(f"已保存 {len(self.rois)} 个区域到 {self.config_path}")
          else:
            print("警告：没有定义任何区域，未保存。")
          break
        # 检查窗口是否被用户关闭
        if cv2.getWindowProperty(self._window_name, cv2.WND_PROP_VISIBLE) < 1:
          window_closed = True
          break
      except cv2.error as e:
        if "NULL window" in str(e):
          window_closed = True
          break
        raise

    # 安全销毁窗口
    try:
      cv2.destroyWindow(self._window_name)
    except:
      pass
    
    if window_closed:
      print("窗口已关闭。")


def define_rois(video_path: str, config_path: str = "config.json"):
  """对外暴露的简易接口。"""
  manager = ROIConfigManager(config_path=config_path)
  manager.load()
  manager.define_rois_on_first_frame(video_path)

