# test_roi.py
from config_manager import define_rois

if __name__ == "__main__":
    video_path = input("请输入视频路径: ").strip()
    define_rois(video_path)