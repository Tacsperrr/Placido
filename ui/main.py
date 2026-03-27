import os
import traceback

import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
from PyQt5 import QtCore, QtGui
from PyQt5.QtWidgets import  QFileDialog
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import cv2
from PyQt5.QtCore import QTimer, Qt
import sys
import index_ui
import ui.photo_ui as photo_ui
import ui.video_ui as video_ui
import src.csy_rc
import detect
import video_detector

# 加载样式表
def load_stylesheet():
    style_path = os.path.join(os.path.dirname(__file__), "style.qss")
    if os.path.exists(style_path):
        with open(style_path, 'r', encoding='utf-8') as f:
            return f.read()
    return ""

# 主界面1
class UI_index(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = index_ui.Ui_Form()
        self.ui.setupUi(self)
        # 应用样式表
        self.setStyleSheet(load_stylesheet())
        # 调整窗口大小
        self.resize(1000, 650)
        self.ui.pushButton.clicked.connect(lambda: self.go_to_photo())
        self.ui.pushButton_2.clicked.connect(lambda: self.go_to_video())
        self.ui.pushButton_3.clicked.connect(lambda: self.exit())
        self.show()

    # 跳转到主界面2
    def go_to_photo(self):
        self.win = UI_photo()
        self.close()
    # 跳转到主界面3
    def go_to_video(self):
        self.win = UI_video()
        self.close()
    # 关闭程序
    def exit(self):
        self.close()

# 图像检测
class UI_photo(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = photo_ui.Ui_Form()
        self.ui.setupUi(self)
        self.img_path = None
        # 应用样式表
        self.setStyleSheet(load_stylesheet())
        # 调整窗口大小
        self.resize(1000, 650)
        self.ui.pushButton.clicked.connect(self.openimage)
        self.ui.pushButton_2.clicked.connect(lambda: self.detect())
        self.ui.pushButton_3.clicked.connect(lambda: self.exit())
        self.show()

    # 跳转到主界面2
    def go_to_photo(self):
        self.win = UI_photo()
        self.close()
    # 跳转到主界面3
    def go_to_video(self):
        self.win = UI_video()
        self.close()
    # 关闭程序
    def exit(self):
        self.win = UI_index()
        self.close()

    # 读取图片,将路径存入txt文件中
    def openimage(self):
        imgName, _ = QFileDialog.getOpenFileName(
            self,
            "打开图片",
            "",
            "Image Files (*.png *.jpg *.jpeg);;All Files (*)"  # 修正过滤器格式
        )

        # 验证路径有效性
        if not imgName:
            return
        if not os.path.exists(imgName):
            QMessageBox.warning(self, "错误", "文件不存在！")
            return

        # 验证是否为图片文件（简单扩展名检查）
        valid_extensions = ['.png', '.jpg', '.jpeg']
        if not imgName.lower().endswith(tuple(valid_extensions)):
            QMessageBox.warning(self, "错误", "不支持的图片格式！")
            return

        # 加载时检查QPixmap有效性
        pixmap = QPixmap(imgName)
        if pixmap.isNull():
            QMessageBox.critical(self, "错误", "图片加载失败！\n可能原因：\n1. 文件已损坏\n2. 格式不受支持")
            return

        # 保持宽高比缩放
        scaled_pixmap = pixmap.scaled(
            self.ui.label.width(),
            self.ui.label.height(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.ui.label.setPixmap(scaled_pixmap)
        self.img_path = imgName

    def cv2_imread_chinese(self,path):
        """
        读取图片，兼容中文路径
        :param path: 图片路径（可含中文）
        :return: 图片数组（None=读取失败）
        """
        try:
            # 读取文件流，避免 OpenCV 中文路径问题
            stream = np.fromfile(path, dtype=np.uint8)
            # 解码为图像
            img = cv2.imdecode(stream, cv2.IMREAD_COLOR)
            return img
        except Exception as e:
            print(f"读取图片失败: {e}")
            return None

    def detect(self):
        try:
            # 1. 检查是否选择了图片
            if not self.img_path:
                QMessageBox.warning(self, "警告", "请先选择图片!")
                return

            # 2. 读取图片（兼容中文路径 + 增加失败判断）
            img = self.cv2_imread_chinese(self.img_path)
            if img is None:
                QMessageBox.critical(
                    self,
                    "文件读取错误",
                    f"无法读取图片：{self.img_path}\n可能原因：\n1. 文件不存在或路径错误\n2. 路径包含中文（已兼容，但仍请检查）\n3. 文件损坏或格式不支持"
                )
                return

            print(f"main read - 图像尺寸: {img.shape}")

            # 3. 初始化检测器并执行检测
            detector = detect.PlacidoDetector()
            # ===== 关键修改：从 4 个返回值改为 5 个 =====
            center, vis_img, mask, has_break, distortion = detector.detect(self.img_path)  # 新增 distortion

            # 4. 检查检测结果有效性
            if center is None:
                QMessageBox.warning(self, "警告", "未检测到圆环，请输入有效眼部图像!")
                vis_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                if not (0 <= center[0] < img.shape[1] and 0 <= center[1] < img.shape[0]):
                    raise ValueError(f"无效中心坐标: {center}，超出图像范围")
                vis_rgb = cv2.cvtColor(vis_img if vis_img is not None else img, cv2.COLOR_BGR2RGB)

                # 在 4. 检查检测结果有效性 之后，5. 显示图像 之前新增：
                # 新增：图像检测的破裂状态提示
                if has_break:
                    QMessageBox.information(self, "检测结果", "当前图像检测到泪膜破裂！")
                else:
                    QMessageBox.information(self, "检测结果", "当前图像未检测到泪膜破裂，泪膜状态良好！")
            # 5. 显示图像（不变）
            height, width, channel = vis_rgb.shape
            bytes_per_line = 3 * width
            q_img = QImage(vis_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img)

            scaled_pixmap = pixmap.scaled(
                self.ui.label_2.width(),
                self.ui.label_2.height(),
                QtCore.Qt.KeepAspectRatio,
                QtCore.Qt.SmoothTransformation
            )
            self.ui.label_2.setPixmap(scaled_pixmap)

        except Exception as e:
            QMessageBox.critical(self, "错误", f"检测失败: {str(e)}")
            print(f"错误详情: {traceback.format_exc()}")

    def non_empty_lines(text):
        return [line.strip() for line in text.splitlines() if line.strip()]

# 视频检测
class UI_video(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = video_ui.Ui_Form()
        self.ui.setupUi(self)
        # 应用样式表
        self.setStyleSheet(load_stylesheet())
        # 调整窗口大小
        self.resize(1000, 650)

        # 视频相关初始化
        self.cap = None
        self.timer = QTimer(self)  # 初始化定时器
        self.timer.timeout.connect(self.update_frame)
        self.is_playing = False

        # ===== 新增：泪膜破裂时间段记录变量 =====
        self.break_periods = []  # 存储破裂时间段 [(开始时间1, 结束时间1), ...]
        self.is_breaking = False  # 当前帧是否处于破裂状态
        self.break_start_time = 0.0  # 破裂开始时间（秒）
        self.frame_count = 0  # 记录当前处理到第几帧
        self.fps = 0  # 视频帧率
        self.total_frames = 0  # 视频总帧数
        self.first_break_time = None  # 保留用于兼容，实际使用论文方法

        # ===== 新增：投票机制变量 =====
        self.break_vote_count = 0  # 连续判定破裂的帧数
        self.vote_threshold = 1  # 改为1帧，快速响应
        self.video_duration = 0  # 视频总时长（秒）
        self.blink_end_time = 0.0  # 瞬目结束时间（秒），用于计算BUT
        self.current_distortion = 0.0  # 当前帧的扭曲值
        self.distortion_history = []  # 扭曲值历史记录

        # ===== 新增：帧缓存用于进度条点击回溯 =====
        self.frame_cache = []  # 存储每帧信息 [(frame, has_break, breaks_info), ...]
        self.original_frames = []  # 存储原始帧图像
        self.video_path = None  # 视频路径

        # ===== 新增：论文方法 - RMS和方差序列 =====
        self.rms_sequence = []  # 存储每帧的RMS值
        self.variance_sequence = []  # 存储每帧的方差值
        self.formation_time = None  # 泪膜形成时间
        self.break_time_paper = None  # 论文方法的破裂时间

        # 按钮连接
        self.ui.pushButton.clicked.connect(self.open_video)
        self.ui.pushButton_2.clicked.connect(self.toggle_video)
        self.ui.pushButton_3.clicked.connect(self.exit)

        self.detector = detect.PlacidoDetector()
        # 使用 video_detector 模块进行视频分析
        self.video_detector = video_detector.VideoDetector(self.detector)
        self.show()

    def open_video(self):
        videoName, _ = QFileDialog.getOpenFileName(self, "打开视频", "", "*.mp4;*.avi;All Files(*)")
        if videoName:
            self.video_path = videoName
            self.cap = cv2.VideoCapture(videoName)

            # 检查视频是否成功打开
            if not self.cap.isOpened():
                print("无法打开视频文件")
                return

            # ===== 新增：获取视频帧率 + 重置检测变量 =====
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)  # 获取视频帧率
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 获取总帧数
            self.video_duration = self.total_frames / self.fps if self.fps > 0 else 0  # 视频总时长（秒）
            self.frame_count = 0  # 重置帧计数
            self.break_periods.clear()  # 清空上一次的时间段
            self.is_breaking = False  # 重置破裂状态
            self.break_vote_count = 0  # 重置投票计数
            self.break_start_time = 0.0  # 重置开始时间
            self.first_break_time = None  # 重置首次破裂时间
            self.distortion_history = []  # 重置扭曲值历史
            self.current_distortion = 0.0  # 重置当前扭曲值

            # 重置帧缓存
            self.frame_cache = []
            self.original_frames = []

            # 重置论文方法变量
            self.rms_sequence = []
            self.variance_sequence = []
            self.formation_time = None
            self.break_time_paper = None

            # 重置状态标签
            pass  # 状态标签已删除

            # 读取视频的第一帧
            ret, first_frame = self.cap.read()
            if ret:
                # 将第一帧转换为 RGB 格式并显示在 label_1 上
                first_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
                self.display_original_frame(first_frame)

            # 初始化播放状态
            self.is_playing = False

    def start_playback(self):
        if self.cap.isOpened():
            self.is_playing = False #操作初始化是否播放按键

    def update_frame(self):
        if self.is_playing:  # 只有在播放状态下才读取和处理帧
            ret, frame = self.cap.read()
            if ret:
                # 颜色空间转换
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # 显示原始帧
                self.display_original_frame(frame)

                # 调用检测方法
                self.detect_frame(frame)
            else:
                # ===== 新增：视频播放完毕，自动停止并显示结果 =====
                self.is_playing = False
                self.timer.stop()
                self.show_break_results()
        else:
            pass

    def display_original_frame(self, frame):
        """显示原始帧"""
        height, width, channel = frame.shape
        bytes_per_line = 3 * width
        q_img = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        scaled_pix = pixmap.scaled(
            self.ui.label.width(),
            self.ui.label.height(),
            QtCore.Qt.KeepAspectRatio,
            QtCore.Qt.SmoothTransformation
        )
        self.ui.label.setPixmap(scaled_pix)

    def detect_frame(self, frame):
        """对当前帧进行检测并显示结果"""
        try:
            # ===== 关键修改：接收第五个返回值 distortion =====
            center, vis_img, mask, has_break, distortion = self.detector.detect(frame)

            # ===== 直接使用detect的原始结果，不过滤任何帧 =====
            # (去掉跳过帧逻辑，让投票机制处理)

            # ===== 新增：计算RMS和方差（使用video_detector模块）=====
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame
            # 计算Zernike拟合RMS（使用video_detector）
            max_radius = self.detector.max_roi if hasattr(self.detector, 'max_roi') else 200
            rms = self.video_detector.calculate_zernike_rms(gray, center, max_radius)
            # 计算环计数方差
            variance = self.video_detector.calculate_ring_variance(gray, center, max_radius)
            # 保存到序列
            self.rms_sequence.append(rms)
            self.variance_sequence.append(variance)

            # 保存当前帧的扭曲值
            if hasattr(self, 'current_distortion'):
                self.current_distortion = distortion
            else:
                self.current_distortion = distortion

            # ===== 新增：计算当前帧时间 + 投票机制判定破裂 =====
            current_time = round(self.frame_count / self.fps, 2)  # 当前帧对应的视频时间（秒，保留2位小数）

            # ===== 投票机制：连续3帧判定破裂才认为真正破裂 =====
            if has_break:
                self.break_vote_count += 1
            else:
                self.break_vote_count = 0

            # 只有连续满足阈值帧数才判定为真正破裂
            final_has_break = self.break_vote_count >= self.vote_threshold

            # 状态机：记录破裂时间段（使用投票后的结果）
            if final_has_break:  # 投票判定破裂
                if not self.is_breaking:  # 之前未破裂，标记开始
                    self.is_breaking = True
                    self.break_start_time = current_time
            else:  # 投票判定未破裂
                if self.is_breaking:  # 之前破裂，标记结束
                    self.is_breaking = False
                    self.break_periods.append((self.break_start_time, current_time))

            # ===== 保存帧缓存 =====
            # 使用投票后的结果 final_has_break
            self.original_frames.append(frame.copy())
            self.frame_cache.append({
                'has_break': final_has_break,  # 使用投票后的结果
                'vis_img': vis_img.copy() if vis_img is not None else None,
                'center': center,
                'time': current_time,
                'distortion': distortion
            })

            # 保存扭曲值历史
            self.distortion_history.append({
                'time': current_time,
                'distortion': distortion
            })

            self.frame_count += 1  # 帧计数+1

            # 更新状态标签显示当前时间
            current_sec = int(current_time)
            pass  # 状态标签已删除

            # 显示检测结果帧
            if vis_img is not None:
                vis_img = cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)
                height, width, channel = vis_img.shape
                bytes_per_line = 3 * width
                q_img = QImage(vis_img.data, width, height, bytes_per_line, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(q_img)
            else:
                # 如果没有检测结果，显示空白
                pixmap = QPixmap(self.ui.label_2.width(), self.ui.label_2.height())
                pixmap.fill(Qt.black)

            scaled_pix = pixmap.scaled(
                self.ui.label_2.width(),
                self.ui.label_2.height(),
                QtCore.Qt.KeepAspectRatio,
                QtCore.Qt.SmoothTransformation
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
            if not self.timer.isActive():  # 如果定时器未激活，则启动它
                self.timer.start(1000 // 30)  # 按30fps播放
            print("检测已恢复")
        else:
            self.timer.stop()
            # ===== 新增：停止播放时显示破裂时间段 =====
            self.show_break_results()
            print("检测已暂停，已生成破裂时间段结果")

    def exit(self):
        # 释放视频资源
        if self.cap:
            self.cap.release()
        # 正确返回主界面
        self.main_win = UI_index()
        self.main_win.show()
        self.close()

    def calculate_dry_eye_grade(self, but_value):
        """根据BUT值计算干眼分级（针对10秒视频调整）"""
        if but_value is None:
            return "稳定（泪膜完整）", "green"

        if but_value >= 10.0:
            return "正常", "green"
        elif but_value >= 5.0:
            return "轻度干眼", "orange"
        else:
            return "重度干眼", "red"

    def plot_distortion_curve(self):
        """绘制扭曲值随时间变化的曲线"""
        if not self.distortion_history or len(self.distortion_history) < 2:
            return None

        times = [d['time'] for d in self.distortion_history]
        distortions = [d['distortion'] for d in self.distortion_history]

        # 创建图表
        fig, ax = plt.subplots(figsize=(10, 5))

        # 绘制扭曲值曲线
        ax.plot(times, distortions, 'b-', linewidth=2, label='Distortion')
        ax.fill_between(times, distortions, alpha=0.3)

        # 添加阈值线
        ax.axhline(y=0.1, color='green', linestyle='--', alpha=0.7, label='Stable (0.1)')
        ax.axhline(y=0.3, color='orange', linestyle='--', alpha=0.7, label='Warning (0.3)')

        # 设置图表属性
        ax.set_xlabel('Time (seconds)', fontsize=12)
        ax.set_ylabel('Distortion Value', fontsize=12)
        ax.set_title('Tear Film Distortion Over Time', fontsize=14)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, max(0.5, max(distortions) * 1.2))

        plt.tight_layout()

        # 保存图表
        save_path = 'distortion_curve.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        return save_path

    def show_break_results(self):
        """显示泪膜破裂时间段结果"""
        # 视频播放结束后，检查是否还处于破裂状态
        if self.is_breaking:
            total_time = round(self.frame_count / self.fps, 2)
            self.break_periods.append((self.break_start_time, total_time))

        # ===== 新增：使用论文方法估计泪膜形成和破裂时间 =====
        # ===== 添加调试信息 =====
        print(f"\n========== 方差序列调试信息 ==========")
        print(f"方差序列长度: {len(self.variance_sequence)}")
        if len(self.variance_sequence) > 0:
            print(f"前10帧方差: {[round(v, 6) for v in self.variance_sequence[:10]]}")
            print(f"方差均值: {np.mean(self.variance_sequence):.6f}")
            print(f"方差最大值: {np.max(self.variance_sequence):.6f}")
            print(f"方差最大值位置(帧): {np.argmax(self.variance_sequence)}")
            # 打印梯度变化
            if len(self.variance_sequence) > 5:
                var_arr = np.array(self.variance_sequence[:20])
                grad = np.gradient(var_arr)
                print(f"前20帧方差梯度: {[round(g, 6) for g in grad]}")

        if len(self.rms_sequence) >= 10 and self.fps > 0:
            timestamps = np.arange(len(self.rms_sequence)) / self.fps
            # 估计泪膜形成时间
            self.formation_time = self.video_detector.estimate_formation_time(
                self.rms_sequence, timestamps
            )
            # 估计泪膜破裂时间（论文方法）
            self.break_time_paper = self.video_detector.estimate_break_time(
                self.variance_sequence, timestamps
            )
            print(f"计算的破裂时间: {self.break_time_paper}")
        print(f"=========================================\n")

        # 计算首次破裂时间（BUT）
        if self.break_periods:
            self.first_break_time = self.break_periods[0][0]  # 首次破裂开始时间
        else:
            self.first_break_time = None

        # 根据视频时长确定分级标准
        video_duration = self.video_duration if self.video_duration > 0 else 10.0

        # 格式化结果文本
        result_text = "=== 泪膜破裂时间检测结果 ===\n\n"

        # 添加视频信息
        result_text += f"【视频信息】\n"
        result_text += f"帧率: {self.fps:.1f} FPS\n"
        result_text += f"总时长: {video_duration:.1f} 秒\n"
        result_text += f"总帧数: {self.total_frames}\n\n"

        # ===== 修改：优先使用实时检测的首次破裂时间 =====
        # 实时检测基于多特征融合，更直接准确
        # 方差计算作为辅助参考
        real_break_time = self.first_break_time

        result_text += f"【泪膜破裂时间 (BUT)】\n"
        result_text += f"检测方法: 多特征融合实时检测\n"
        if real_break_time is not None:
            result_text += f"首次破裂时间: {real_break_time:.2f} 秒\n"
        else:
            result_text += f"首次破裂时间: > {video_duration:.1f} 秒\n"
            result_text += f"（视频内未检测到破裂）\n"

        # 附加：方差分析参考
        if self.break_time_paper is not None:
            result_text += f"方差分析参考: {self.break_time_paper:.2f} 秒\n\n"
        else:
            result_text += f"方差分析参考: 无法估计\n\n"

        # 干眼分级诊断（使用实时检测结果）
        result_text += f"【干眼分级诊断】\n"

        # 使用实时检测的破裂时间
        if real_break_time is not None:
            but_for_grade = real_break_time
        else:
            but_for_grade = video_duration  # 未破裂视为稳定

        if real_break_time is None:
            result_text += f"分级: 正常\n"
            result_text += f"说明: 视频时间内未检测到破裂，泪膜稳定\n"
        else:
            if but_for_grade >= 10.0:
                result_text += f"分级: 正常\n"
                result_text += f"说明: BUT ≥ 10秒，泪膜稳定\n"
            elif but_for_grade >= 5.0:
                result_text += f"分级: 轻度干眼\n"
                result_text += f"说明: 5秒 ≤ BUT < 10秒，泪膜不稳定\n"
            else:
                result_text += f"分级: 重度干眼\n"
                result_text += f"说明: BUT < 5秒，泪膜破裂风险高\n"


        # ===== 新增：论文方法的分析结果 =====
        result_text += f"\n【基于Zernike RMS和环计数方差分析结果】\n"
        if self.formation_time is not None:
            result_text += f"泪膜形成时间: {self.formation_time:.2f} 秒\n"
        else:
            result_text += f"泪膜形成时间: 无法估计\n"

        if self.break_time_paper is not None:
            result_text += f"泪膜破裂时间: {self.break_time_paper:.2f} 秒\n"
            # 破裂时间分级
            if self.break_time_paper >= 10:
                result_text += f"分级评估: 正常 (≥10秒)\n"
            elif self.break_time_paper >= 5:
                result_text += f"分级评估: 轻度干眼 (5-10秒)\n"
            else:
                result_text += f"分级评估: 重度干眼 (<5秒)\n"
        else:
            result_text += f"泪膜破裂时间: > {video_duration:.1f} 秒 (未检测到破裂)\n"

        # 添加RMS和方差统计
        if self.rms_sequence:
            valid_rms = [r for r in self.rms_sequence if r < 1e6]
            if valid_rms:
                result_text += f"RMS均值: {np.mean(valid_rms):.4f}\n"
                result_text += f"RMS最大: {np.max(valid_rms):.4f}\n"
        if self.variance_sequence:
            result_text += f"方差均值: {np.mean(self.variance_sequence):.4f}\n"
            result_text += f"方差最大: {np.max(self.variance_sequence):.4f}\n"
        if self.distortion_history:
            distortions = [d['distortion'] for d in self.distortion_history]
            avg_distortion = sum(distortions) / len(distortions)
            max_distortion = max(distortions)
            final_distortion = distortions[-1] if distortions else 0

            result_text += f"平均扭曲值: {avg_distortion:.4f}\n"
            result_text += f"最大扭曲值: {max_distortion:.4f}\n"

            # 根据扭曲值给出评估
            if avg_distortion < 0.1:
                result_text += f"评估: 泪膜稳定\n"
            elif avg_distortion < 0.3:
                result_text += f"评估: 泪膜轻微不稳定\n"
            else:
                result_text += f"评估: 泪膜不稳定，建议注意\n"
        else:
            result_text += f"无扭曲值数据\n"

        result_text += f"\n【分级标准】\n"
        result_text += f"正常: BUT ≥ 10秒\n"
        result_text += f"轻度干眼: 5秒 ≤ BUT < 10秒\n"
        result_text += f"重度干眼: BUT < 5秒\n"


        # 绘制并显示扭曲值曲线
        distortion_plot_path = self.plot_distortion_curve()

        # 创建自定义对话框：左边文字，右边曲线（上下布局）+ 关闭按钮
        dialog = QDialog(self)
        dialog.setWindowTitle("视频检测完成")
        dialog.resize(1100, 600)
        dialog.setStyleSheet("background-color: #f5f7fa;")

        # 主布局：水平分布
        main_layout = QHBoxLayout(dialog)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(20)

        # 左边：文字信息
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

        # 右边：扭曲值曲线 + 关闭按钮（右下角）
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_widget.setStyleSheet("background: white; border-radius: 12px;")
        right_layout.setContentsMargins(20, 20, 20, 20)

        # 上半部分：标题和图片
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

        # 添加一个stretch将按钮推到最下方
        right_layout.addStretch()

        # 关闭按钮放在右下角
        button_layout = QHBoxLayout()
        button_layout.addStretch()  # 左侧添加stretch，将按钮推到右边
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

        # 添加到主布局
        main_layout.addWidget(left_widget, 1)
        main_layout.addWidget(right_widget, 1)

        dialog.exec_()

    def update_slider_style(self):
        """根据破裂情况更新滑块样式"""
        if not self.frame_cache:
            return

        # 检查是否有破裂帧
        has_any_break = any(frame['has_break'] for frame in self.frame_cache)

        if not has_any_break:
            # 无破裂，全部绿色
            style = """
                QSlider::groove:horizontal {
                    border: none;
                    height: 8px;
                    background: #27ae60;
                    border-radius: 4px;
                }
                QSlider::handle:horizontal {
                    width: 16px;
                    margin: -4px 0;
                    background: #1e8449;
                    border-radius: 8px;
                }
                QSlider::handle:horizontal:hover {
                    background: #239b56;
                }
            """
        else:
            # 有破裂，使用渐变色（绿色到红色）
            # 计算破裂帧比例
            break_count = sum(1 for f in self.frame_cache if f['has_break'])
            total = len(self.frame_cache)
            break_ratio = break_count / total if total > 0 else 0

            style = f"""
                QSlider::groove:horizontal {{
                    border: none;
                    height: 8px;
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                        stop:0 #27ae60,
                        stop:{1-break_ratio:.2f} #27ae60,
                        stop:{1-break_ratio:.2f} #e74c3c,
                        stop:1 #e74c3c);
                    border-radius: 4px;
                }}
                QSlider::handle:horizontal {{
                    width: 16px;
                    margin: -4px 0;
                    background: #3498db;
                    border-radius: 8px;
                }}
                QSlider::handle:horizontal:hover {{
                    background: #2980b9;
                }}
            """

        self.ui.videoSlider.setStyleSheet(style)

    def on_slider_pressed(self):
        """滑块按下时暂停播放"""
        if self.is_playing:
            self.was_playing = True
            self.timer.stop()
        else:
            self.was_playing = False

    def on_slider_released(self):
        """滑块释放时恢复播放"""
        if hasattr(self, 'was_playing') and self.was_playing:
            self.timer.start(1000 // 30)

    def on_slider_value_changed(self):
        """滑块值改变时显示对应帧"""
        # 检查视频是否已加载
        if not self.frame_cache or self.video_path is None:
            return

        # 获取当前秒数
        current_second = self.ui.videoSlider.value()

        # 更新时间显示
        current_sec = int(current_second)
        self.ui.timeLabel.setText(f"{current_sec // 60:02d}:{current_sec % 60:02d}")

        # 显示对应帧
        self.display_frame_at_second(current_second)

    def display_frame_at_second(self, second):
        """根据秒数显示对应帧"""
        # 检查数据是否有效
        if not self.frame_cache or self.fps <= 0:
            return

        # 根据秒数找到对应的帧索引
        frame_idx = int(second * self.fps)

        # 限制帧索引范围
        frame_idx = max(0, min(frame_idx, len(self.frame_cache) - 1))

        # 如果已有缓存数据，直接使用
        if frame_idx >= 0 and frame_idx < len(self.frame_cache):
            frame_info = self.frame_cache[frame_idx]

            # 显示原始帧
            if frame_idx < len(self.original_frames):
                frame = self.original_frames[frame_idx]
                self.display_original_frame(frame)

            # 显示检测结果帧
            if frame_info['vis_img'] is not None:
                vis_img = cv2.cvtColor(frame_info['vis_img'], cv2.COLOR_BGR2RGB)
                height, width, channel = vis_img.shape
                bytes_per_line = 3 * width
                q_img = QImage(vis_img.data, width, height, bytes_per_line, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(q_img)
            else:
                pixmap = QPixmap(self.ui.label_2.width(), self.ui.label_2.height())
                pixmap.fill(Qt.black)

            scaled_pix = pixmap.scaled(
                self.ui.label_2.width(),
                self.ui.label_2.height(),
                QtCore.Qt.KeepAspectRatio,
                QtCore.Qt.SmoothTransformation
            )
            self.ui.label_2.setPixmap(scaled_pix)

            # 在页面上显示当前帧的破裂信息
            if frame_info['has_break']:
                status_text = f"时间: {second:.2f}秒 | 帧号: {frame_idx} | 状态: ⚠️ 检测到泪膜破裂"
                pass  # 状态标签已删除
            else:
                status_text = f"时间: {second:.2f}秒 | 帧号: {frame_idx} | 状态: ✅ 泪膜完好"
                pass  # 状态标签已删除

    def seek_to_frame(self, frame_idx):
        """从视频中seek到指定帧"""
        if self.cap is None and self.video_path:
            # 重新打开视频
            self.cap = cv2.VideoCapture(self.video_path)

        if self.cap is None:
            return

        # 保存当前播放状态
        was_playing = self.is_playing
        if was_playing:
            self.timer.stop()

        # seek到指定帧
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.display_original_frame(frame)

            # 实时检测
            center, vis_img, mask, has_break, distortion = self.detector.detect(frame)

            if vis_img is not None:
                vis_img = cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)
                height, width, channel = vis_img.shape
                bytes_per_line = 3 * width
                q_img = QImage(vis_img.data, width, height, bytes_per_line, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(q_img)
            else:
                pixmap = QPixmap(self.ui.label_2.width(), self.ui.label_2.height())
                pixmap.fill(Qt.black)

            scaled_pix = pixmap.scaled(
                self.ui.label_2.width(),
                self.ui.label_2.height(),
                QtCore.Qt.KeepAspectRatio,
                QtCore.Qt.SmoothTransformation
            )
            self.ui.label_2.setPixmap(scaled_pix)

            # 更新时间
            current_time = frame_idx / self.fps if self.fps > 0 else 0
            current_sec = int(current_time)
            self.ui.timeLabel.setText(f"{current_sec // 60:02d}:{current_sec % 60:02d}")

            # 在页面上显示状态
            if has_break:
                status_text = f"时间: {current_time:.2f}秒 | 帧号: {frame_idx} | 状态: ⚠️ 检测到泪膜破裂"
                pass  # 状态标签已删除
            else:
                status_text = f"时间: {current_time:.2f}秒 | 帧号: {frame_idx} | 状态: ✅ 泪膜完好"
                pass  # 状态标签已删除

    def seek_to_frame_by_second(self, second):
        """从视频中seek到指定秒数"""
        if self.cap is None and self.video_path:
            # 重新打开视频
            self.cap = cv2.VideoCapture(self.video_path)

        if self.cap is None:
            return

        # 保存当前播放状态
        was_playing = self.is_playing
        if was_playing:
            self.timer.stop()

        # seek到指定秒数
        self.cap.set(cv2.CAP_PROP_POS_MSEC, second * 1000)
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.display_original_frame(frame)

            # 实时检测
            center, vis_img, mask, has_break, distortion = self.detector.detect(frame)

            if vis_img is not None:
                vis_img = cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)
                height, width, channel = vis_img.shape
                bytes_per_line = 3 * width
                q_img = QImage(vis_img.data, width, height, bytes_per_line, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(q_img)
            else:
                pixmap = QPixmap(self.ui.label_2.width(), self.ui.label_2.height())
                pixmap.fill(Qt.black)

            scaled_pix = pixmap.scaled(
                self.ui.label_2.width(),
                self.ui.label_2.height(),
                QtCore.Qt.KeepAspectRatio,
                QtCore.Qt.SmoothTransformation
            )
            self.ui.label_2.setPixmap(scaled_pix)

            # 更新时间
            current_sec = int(second)
            self.ui.timeLabel.setText(f"{current_sec // 60:02d}:{current_sec % 60:02d}")

            # 在页面上显示状态
            if has_break:
                status_text = f"时间: {second:.2f}秒 | 状态: ⚠️ 检测到泪膜破裂"
                pass  # 状态标签已删除
            else:
                status_text = f"时间: {second:.2f}秒 | 状态: ✅ 泪膜完好"
                pass  # 状态标签已删除

        # 恢复播放状态
        if was_playing:
            self.timer.start(1000 // 30)

if __name__ =="__main__":
    app =QApplication(sys.argv)
    win = UI_index()
    sys.exit(app.exec_())