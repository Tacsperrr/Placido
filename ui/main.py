import os
import traceback

import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
from PyQt5 import QtCore, QtGui
from PyQt5.QtWidgets import QFileDialog, QMessageBox, QApplication, QMainWindow, QDialog, QHBoxLayout, QVBoxLayout, QWidget, QLabel, QPushButton
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import QTimer, Qt
import sys
import cv2

# 确保根目录和ui目录都在路径中
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ui_dir = os.path.dirname(os.path.abspath(__file__))
if _root not in sys.path:
    sys.path.insert(0, _root)
if _ui_dir not in sys.path:
    sys.path.insert(0, _ui_dir)

import index_ui
import photo_ui
import video_ui
import src.csy_rc
import detect
import video_detector


def load_stylesheet():
    style_path = os.path.join(os.path.dirname(__file__), "style.qss")
    if os.path.exists(style_path):
        with open(style_path, 'r', encoding='utf-8') as f:
            return f.read()
    return ""


class UI_index(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = index_ui.Ui_Form()
        self.ui.setupUi(self)
        self.setStyleSheet(load_stylesheet())
        self.resize(1000, 650)
        self.ui.pushButton.clicked.connect(self.go_to_photo)
        self.ui.pushButton_2.clicked.connect(self.go_to_video)
        self.ui.pushButton_3.clicked.connect(self.exit)
        self.show()

    def go_to_photo(self):
        self.win = UI_photo()
        self.close()

    def go_to_video(self):
        self.win = UI_video()
        self.close()

    def exit(self):
        self.close()


class UI_photo(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = photo_ui.Ui_Form()
        self.ui.setupUi(self)
        self.img_path = None
        self.setStyleSheet(load_stylesheet())
        self.resize(1000, 650)
        self.ui.pushButton.clicked.connect(self.openimage)
        self.ui.pushButton_2.clicked.connect(self.detect)
        self.ui.pushButton_3.clicked.connect(self.exit)
        self.show()

    def go_to_photo(self):
        self.win = UI_photo()
        self.close()

    def go_to_video(self):
        self.win = UI_video()
        self.close()

    def exit(self):
        self.win = UI_index()
        self.close()

    def openimage(self):
        imgName, _ = QFileDialog.getOpenFileName(
            self,
            "打开图片",
            "",
            "Image Files (*.png *.jpg *.jpeg);;All Files (*)"
        )
        if not imgName or not os.path.exists(imgName):
            return
        pixmap = QPixmap(imgName)
        if pixmap.isNull():
            QMessageBox.critical(self, "错误", "图片加载失败！")
            return
        scaled_pixmap = pixmap.scaled(
            self.ui.label.width(),
            self.ui.label.height(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.ui.label.setPixmap(scaled_pixmap)
        self.img_path = imgName

    def cv2_imread_chinese(self, path):
        try:
            stream = np.fromfile(path, dtype=np.uint8)
            img = cv2.imdecode(stream, cv2.IMREAD_COLOR)
            return img
        except Exception as e:
            print(f"读取图片失败: {e}")
            return None

    def detect(self):
        try:
            if not self.img_path:
                QMessageBox.warning(self, "警告", "请先选择图片!")
                return
            img = self.cv2_imread_chinese(self.img_path)
            if img is None:
                QMessageBox.critical(self, "文件读取错误", f"无法读取图片：{self.img_path}")
                return

            detector = detect.PlacidoDetector()
            center, vis_img, mask, has_break, distortion = detector.detect(self.img_path)

            if center is None:
                QMessageBox.warning(self, "警告", "未检测到圆环，请输入有效眼部图像!")
                # 显示原始图像（BGR转RGB）
                vis_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                vis_rgb = vis_img  # vis_img 已经是 RGB 格式
                if has_break:
                    QMessageBox.information(self, "检测结果", "当前图像检测到泪膜破裂！")
                else:
                    QMessageBox.information(self, "检测结果", "当前图像未检测到泪膜破裂，泪膜状态良好！")

            # 将 numpy RGB 数组转换为 QPixmap
            height, width, channel = vis_rgb.shape
            bytes_per_line = 3 * width
            # 关键修复：使用 tobytes() 代替 .data
            q_img = QImage(vis_rgb.tobytes(), width, height, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img)
            scaled_pixmap = pixmap.scaled(
                self.ui.label_2.width(),
                self.ui.label_2.height(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.ui.label_2.setPixmap(scaled_pixmap)

        except Exception as e:
            QMessageBox.critical(self, "错误", f"检测失败: {str(e)}")
            print(traceback.format_exc())


class UI_video(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = video_ui.Ui_Form()
        self.ui.setupUi(self)
        self.setStyleSheet(load_stylesheet())
        self.resize(1000, 650)

        # 视频相关
        self.cap = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.is_playing = False

        # 破裂记录
        self.break_periods = []
        self.is_breaking = False
        self.break_start_time = 0.0
        self.frame_count = 0
        self.fps = 0
        self.total_frames = 0
        self.first_break_time = None
        self.break_vote_count = 0
        self.vote_threshold = 1
        self.video_duration = 0
        self.blink_end_time = 0.0
        self.current_distortion = 0.0
        self.distortion_history = []

        # 帧缓存
        self.frame_cache = []
        self.original_frames = []
        self.video_path = None

        # 论文方法序列
        self.rms_sequence = []
        self.variance_sequence = []
        self.formation_time = None
        self.break_time_paper = None

        # 按钮连接
        self.ui.pushButton.clicked.connect(self.open_video)
        self.ui.pushButton_2.clicked.connect(self.toggle_video)
        self.ui.pushButton_3.clicked.connect(self.exit)

        self.detector = detect.PlacidoDetector()
        self.video_detector = video_detector.VideoDetector(self.detector)
        self.show()

    def open_video(self):
        videoName, _ = QFileDialog.getOpenFileName(self, "打开视频", "", "*.mp4;*.avi;All Files(*)")
        if videoName:
            self.video_path = videoName
            self.cap = cv2.VideoCapture(videoName)
            if not self.cap.isOpened():
                print("无法打开视频文件")
                return

            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.video_duration = self.total_frames / self.fps if self.fps > 0 else 0
            self.frame_count = 0
            self.break_periods.clear()
            self.is_breaking = False
            self.break_vote_count = 0
            self.break_start_time = 0.0
            self.first_break_time = None
            self.distortion_history.clear()
            self.current_distortion = 0.0
            self.frame_cache.clear()
            self.original_frames.clear()
            self.rms_sequence.clear()
            self.variance_sequence.clear()
            self.formation_time = None
            self.break_time_paper = None

            ret, first_frame = self.cap.read()
            if ret:
                first_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
                self.display_original_frame(first_frame)

            self.is_playing = False

    def start_playback(self):
        if self.cap.isOpened():
            self.is_playing = False

    def update_frame(self):
        if self.is_playing:
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.display_original_frame(frame)
                self.detect_frame(frame)
            else:
                self.is_playing = False
                self.timer.stop()
                self.show_break_results()

    def display_original_frame(self, frame):
        height, width, channel = frame.shape
        bytes_per_line = 3 * width
        # 修复：使用 tobytes()
        q_img = QImage(frame.tobytes(), width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        scaled_pix = pixmap.scaled(
            self.ui.label.width(),
            self.ui.label.height(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.ui.label.setPixmap(scaled_pix)

    def detect_frame(self, frame):
        try:
            center, vis_img, mask, has_break, distortion = self.detector.detect(frame)

            # 计算RMS和方差
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            else:
                gray = frame
            max_radius = self.detector.max_roi if hasattr(self.detector, 'max_roi') else 200
            rms = self.video_detector.calculate_zernike_rms(gray, center, max_radius)
            variance = self.video_detector.calculate_ring_variance(gray, center, max_radius)
            self.rms_sequence.append(rms)
            self.variance_sequence.append(variance)

            self.current_distortion = distortion

            current_time = round(self.frame_count / self.fps, 2) if self.fps > 0 else 0

            # 投票机制
            if has_break:
                self.break_vote_count += 1
            else:
                self.break_vote_count = 0
            final_has_break = self.break_vote_count >= self.vote_threshold

            if final_has_break:
                if not self.is_breaking:
                    self.is_breaking = True
                    self.break_start_time = current_time
            else:
                if self.is_breaking:
                    self.is_breaking = False
                    self.break_periods.append((self.break_start_time, current_time))

            # 保存缓存
            self.original_frames.append(frame.copy())
            self.frame_cache.append({
                'has_break': final_has_break,
                'vis_img': vis_img.copy() if vis_img is not None else None,
                'center': center,
                'time': current_time,
                'distortion': distortion
            })
            self.distortion_history.append({'time': current_time, 'distortion': distortion})

            self.frame_count += 1

            # 显示检测结果（vis_img 已经是 RGB 格式）
            if vis_img is not None:
                height, width, channel = vis_img.shape
                bytes_per_line = 3 * width
                # 修复：使用 tobytes()
                q_img = QImage(vis_img.tobytes(), width, height, bytes_per_line, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(q_img)
            else:
                pixmap = QPixmap(self.ui.label_2.width(), self.ui.label_2.height())
                pixmap.fill(Qt.black)

            scaled_pix = pixmap.scaled(
                self.ui.label_2.width(),
                self.ui.label_2.height(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.ui.label_2.setPixmap(scaled_pix)

        except Exception as e:
            print(f"检测失败: {str(e)}")

    def toggle_video(self):
        if self.cap is None or not self.cap.isOpened():
            print("请先选择一个视频文件")
            return
        self.is_playing = not self.is_playing
        if self.is_playing:
            if not self.timer.isActive():
                self.timer.start(1000 // 30)
            print("检测已恢复")
        else:
            self.timer.stop()
            self.show_break_results()
            print("检测已暂停")

    def exit(self):
        if self.cap:
            self.cap.release()
        self.main_win = UI_index()
        self.main_win.show()
        self.close()

    def calculate_dry_eye_grade(self, but_value):
        if but_value is None:
            return "稳定（泪膜完整）", "green"
        if but_value >= 10.0:
            return "正常", "green"
        elif but_value >= 5.0:
            return "轻度干眼", "orange"
        else:
            return "重度干眼", "red"

    def plot_distortion_curve(self):
        if not self.distortion_history or len(self.distortion_history) < 2:
            return None
        times = [d['time'] for d in self.distortion_history]
        distortions = [d['distortion'] for d in self.distortion_history]
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(times, distortions, 'b-', linewidth=2, label='Distortion')
        ax.fill_between(times, distortions, alpha=0.3)
        ax.axhline(y=0.1, color='green', linestyle='--', alpha=0.7, label='Stable (0.1)')
        ax.axhline(y=0.3, color='orange', linestyle='--', alpha=0.7, label='Warning (0.3)')
        ax.set_xlabel('Time (seconds)', fontsize=12)
        ax.set_ylabel('Distortion Value', fontsize=12)
        ax.set_title('Tear Film Distortion Over Time', fontsize=14)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, max(0.5, max(distortions) * 1.2))
        plt.tight_layout()
        save_path = 'distortion_curve.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        return save_path

    def show_break_results(self):
        if self.is_breaking:
            total_time = round(self.frame_count / self.fps, 2) if self.fps > 0 else 0
            self.break_periods.append((self.break_start_time, total_time))

        # 使用论文方法估计
        if len(self.rms_sequence) >= 10 and self.fps > 0:
            timestamps = np.arange(len(self.rms_sequence)) / self.fps
            self.formation_time = self.video_detector.estimate_formation_time(self.rms_sequence, timestamps)
            self.break_time_paper = self.video_detector.estimate_break_time(self.variance_sequence, timestamps)

        real_break_time = self.break_periods[0][0] if self.break_periods else None
        video_duration = self.video_duration if self.video_duration > 0 else 10.0

        result_text = "=== 泪膜破裂时间检测结果 ===\n\n"
        result_text += f"【视频信息】\n帧率: {self.fps:.1f} FPS\n总时长: {video_duration:.1f} 秒\n总帧数: {self.total_frames}\n\n"
        result_text += f"【干眼分级诊断】\n"
        if real_break_time is None:
            result_text += f"分级: 正常\n说明: 视频时间内未检测到破裂，泪膜稳定\n"
        else:
            if real_break_time >= 10.0:
                result_text += f"分级: 正常 (BUT ≥ 10秒)\n"
            elif real_break_time >= 5.0:
                result_text += f"分级: 轻度干眼 (5秒 ≤ BUT < 10秒)\n"
            else:
                result_text += f"分级: 重度干眼 (BUT < 5秒)\n"

        result_text += f"\n【基于Zernike RMS和环计数方差分析结果】\n"
        if self.formation_time is not None:
            result_text += f"泪膜形成时间: {self.formation_time:.2f} 秒\n"
        else:
            result_text += f"泪膜形成时间: 无法估计\n"
        if self.break_time_paper is not None:
            result_text += f"泪膜破裂时间: {self.break_time_paper:.2f} 秒\n"
        else:
            result_text += f"泪膜破裂时间: > {video_duration:.1f} 秒 (未检测到破裂)\n"

        if self.rms_sequence:
            valid_rms = [r for r in self.rms_sequence if r < 1e6]
            if valid_rms:
                result_text += f"RMS均值: {np.mean(valid_rms):.4f}\nRMS最大: {np.max(valid_rms):.4f}\n"
        if self.variance_sequence:
            result_text += f"方差均值: {np.mean(self.variance_sequence):.4f}\n方差最大: {np.max(self.variance_sequence):.4f}\n"
        if self.distortion_history:
            distortions = [d['distortion'] for d in self.distortion_history]
            result_text += f"平均扭曲值: {np.mean(distortions):.4f}\n最大扭曲值: {np.max(distortions):.4f}\n"
            if np.mean(distortions) < 0.1:
                result_text += f"评估: 泪膜稳定\n"
            elif np.mean(distortions) < 0.3:
                result_text += f"评估: 泪膜轻微不稳定\n"
            else:
                result_text += f"评估: 泪膜不稳定，建议注意\n"

        distortion_plot_path = self.plot_distortion_curve()

        # 弹窗显示结果
        dialog = QDialog(self)
        dialog.setWindowTitle("视频检测完成")
        dialog.resize(1100, 600)
        dialog.setStyleSheet("background-color: #f5f7fa;")
        main_layout = QHBoxLayout(dialog)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(20)

        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_widget.setStyleSheet("background: white; border-radius: 12px;")
        left_layout.setContentsMargins(20, 20, 20, 20)
        title_label = QLabel("检测结果")
        title_label.setStyleSheet("color: #2c3e50; font-size: 18px; font-weight: bold;")
        left_layout.addWidget(title_label)
        result_label = QLabel(result_text)
        result_label.setStyleSheet("color: #34495e; font-size: 13px;")
        result_label.setWordWrap(True)
        left_layout.addWidget(result_label)
        left_layout.addStretch()

        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_widget.setStyleSheet("background: white; border-radius: 12px;")
        right_layout.setContentsMargins(20, 20, 20, 20)
        curve_title = QLabel("扭曲值变化曲线")
        curve_title.setStyleSheet("color: #2c3e50; font-size: 18px; font-weight: bold;")
        curve_title.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(curve_title)

        if distortion_plot_path and os.path.exists(distortion_plot_path):
            curve_label = QLabel()
            pixmap = QPixmap(distortion_plot_path)
            if not pixmap.isNull():
                scaled_pixmap = pixmap.scaled(500, 350, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                curve_label.setPixmap(scaled_pixmap)
            curve_label.setAlignment(Qt.AlignCenter)
            right_layout.addWidget(curve_label)
        else:
            no_data_label = QLabel("无扭曲值数据")
            no_data_label.setStyleSheet("color: #7f8c8d; font-size: 14px;")
            no_data_label.setAlignment(Qt.AlignCenter)
            right_layout.addWidget(no_data_label)

        right_layout.addStretch()
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        close_btn = QPushButton("关闭")
        close_btn.setMinimumSize(100, 40)
        close_btn.setStyleSheet("""
            QPushButton {
                background: #3498db;
                color: white;
                border: none;
                border-radius: 20px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: #2980b9;
            }
        """)
        close_btn.clicked.connect(dialog.close)
        button_layout.addWidget(close_btn)
        right_layout.addLayout(button_layout)

        main_layout.addWidget(left_widget, 1)
        main_layout.addWidget(right_widget, 1)
        dialog.exec_()

    def update_slider_style(self):
        # 此方法在UI中未使用滑块，保留空实现以防调用
        pass

    def on_slider_pressed(self):
        pass

    def on_slider_released(self):
        pass

    def on_slider_value_changed(self):
        pass

    def display_frame_at_second(self, second):
        pass

    def seek_to_frame(self, frame_idx):
        pass

    def seek_to_frame_by_second(self, second):
        pass


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = UI_index()
    sys.exit(app.exec_())