"""
视频检测模块 - 基于 Placido 环的泪膜视频分析
包含：RMS计算、方差计算、时间估计等视频分析功能
"""

import numpy as np
import cv2
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks


class VideoDetector:
    """视频检测器 - 处理泪膜视频序列分析"""

    def __init__(self, placido_detector=None):
        """
        初始化视频检测器

        :param placido_detector: PlacidoDetector实例，用于单帧检测
        """
        self.detector = placido_detector

    def calculate_zernike_rms(self, gray_img, center, max_radius):
        """
        计算径向轮廓RMS值（泪膜形成时间估计）

        :param gray_img: 灰度图像
        :param center: 中心点 (x, y)
        :param max_radius: 最大半径
        :return: RMS值
        """
        try:
            if center is None:
                return float('inf')
            if gray_img is None or gray_img.size == 0:
                return float('inf')

            cx, cy = center

            # 极坐标变换
            polar_img = self._polar_transform(gray_img, center, max_radius)
            if polar_img is None or polar_img.size == 0:
                return float('inf')

            # 水平投影求平均获取径向轮廓
            radial_profile = cv2.reduce(polar_img, 0, cv2.REDUCE_AVG, dtype=cv2.CV_32F).flatten()

            if len(radial_profile) < 10:
                return float('inf')

            # 平滑处理
            kernel_size = min(9, len(radial_profile) // 2 * 2 + 1)
            smoothed = cv2.GaussianBlur(radial_profile.reshape(1, -1), (1, kernel_size), 2).flatten()

            # 梯度作为高度估计
            gradient = np.gradient(smoothed)

            # 计算RMS：使用径向轮廓直接计算
            if len(gradient) < 10:
                return float('inf')

            # 去除边缘区域（前5%和后10%）
            n = len(gradient)
            valid_gradient = gradient[int(n*0.05):int(n*0.9)]

            if len(valid_gradient) < 5:
                return float('inf')

            # 平滑处理
            smoothed_grad = uniform_filter1d(valid_gradient, size=3)

            # RMS = sqrt(mean(gradient^2))
            rms = float(np.sqrt(np.mean(smoothed_grad ** 2)))

            # 归一化到合理范围
            rms = rms / 255.0 * 100 if rms > 0 else 0

            return rms if np.isfinite(rms) else float('inf')
        except Exception as e:
            print(f"RMS计算异常: {e}")
            return float('inf')

    def calculate_ring_variance(self, gray_img, center, max_radius):
        """
        计算环计数方差（泪膜破裂时间估计）

        :param gray_img: 灰度图像
        :param center: 中心点 (x, y)
        :param max_radius: 最大半径
        :return: 方差值
        """
        try:
            if center is None:
                return 0.0
            if gray_img is None or gray_img.size == 0:
                return 0.0

            cx, cy = center

            # 极坐标变换
            polar_img = self._polar_transform(gray_img, center, max_radius)
            if polar_img is None or polar_img.size == 0:
                return 0.0

            # Marr-Hildreth边缘检测 (LoG)
            sigma = 1.5
            blurred = cv2.GaussianBlur(polar_img, (0, 0), sigma)
            log = cv2.Laplacian(blurred, cv2.CV_64F)

            # 除零错误处理
            max_log = np.max(np.abs(log))
            if max_log == 0 or np.isnan(max_log):
                return 0.0

            threshold = 0.25 * max_log
            edges = np.abs(log) > threshold
            edges = edges.astype(np.uint8)

            # 统计每列的边缘数量
            ring_counts = []
            for col in range(edges.shape[1]):
                valid_data = edges[5:-5, col]
                edge_points = np.where(valid_data > 0)[0]
                if len(edge_points) == 0:
                    ring_counts.append(0)
                else:
                    segments = 1
                    for i in range(1, len(edge_points)):
                        if edge_points[i] - edge_points[i-1] > 5:
                            segments += 1
                    ring_counts.append(segments)

            ring_counts = np.array(ring_counts)

            if len(ring_counts) == 0:
                return 0.0

            mean_val = np.mean(ring_counts)
            if mean_val == 0 or np.isnan(mean_val):
                return 0.0
            ring_counts_normalized = ring_counts - mean_val

            variance = float(np.var(ring_counts_normalized))
            if np.isnan(variance):
                return 0.0

            return variance
        except Exception as e:
            print(f"calculate_ring_variance异常: {e}")
            return 0.0

    def estimate_formation_time(self, rms_sequence, timestamps):
        """
        估计泪膜形成时间
        使用三参数函数拟合RMS趋势，找到局部最小值

        :param rms_sequence: RMS值列表
        :param timestamps: 时间戳列表
        :return: 泪膜形成时间 (秒)
        """
        if len(rms_sequence) < 10:
            return None

        rms = np.array(rms_sequence)
        t = np.array(timestamps)

        valid_mask = np.isfinite(rms)
        if np.sum(valid_mask) < 10:
            return None

        rms = rms[valid_mask]
        t = t[valid_mask]

        # 平滑处理
        rms_smooth = uniform_filter1d(rms, size=3)

        # 找局部最小值
        from scipy.signal import argrelmin
        local_mins = argrelmin(rms_smooth, order=3)[0]

        # 过滤：在时间轴前1/3范围内找最小值
        valid_mins = [m for m in local_mins if m < len(t) * 0.33]

        if valid_mins:
            # 返回第一个局部最小值
            return float(t[valid_mins[0]])

        # 备用：返回RMS开始稳定的点
        for i in range(3, len(rms_smooth) - 3):
            if np.std(rms_smooth[i:i+5]) < np.std(rms_smooth) * 0.5:
                return float(t[i])

        return None

    def estimate_break_time(self, variance_sequence, timestamps):
        """
        估计泪膜破裂时间
        适应视频开始时方差就较高的情况
        策略：找方差从"相对稳定"变为"持续上升"的转折点

        :param variance_sequence: 方差列表
        :param timestamps: 时间戳列表
        :return: 泪膜破裂时间 (秒)
        """
        if len(variance_sequence) < 10:
            return None

        variance = np.array(variance_sequence)
        t = np.array(timestamps)

        valid_mask = np.isfinite(variance)
        if np.sum(valid_mask) < 10:
            return None

        variance = variance[valid_mask]
        t = t[valid_mask]

        # 滑动窗口计算局部均值
        window_size = min(5, len(variance) // 3)
        if window_size < 2:
            return None

        local_means = []
        for i in range(len(variance) - window_size + 1):
            local_means.append(np.mean(variance[i:i+window_size]))
        local_means = np.array(local_means)

        # 方法1：找局部均值开始持续上升的点
        diff = np.diff(local_means)

        rise_start = None
        consecutive_rise = 0
        min_rise_length = 3

        for i in range(len(diff)):
            if diff[i] > 0:
                consecutive_rise += 1
                if consecutive_rise >= min_rise_length and rise_start is None:
                    rise_start = i - consecutive_rise + 1
            else:
                consecutive_rise = 0

        if rise_start is not None and rise_start < len(t):
            return float(t[rise_start])

        # 方法2：找方差显著高于基线的点
        baseline_window = min(5, len(variance) // 4)
        if baseline_window >= 2:
            baseline = np.mean(variance[:baseline_window])
            for i in range(baseline_window, len(variance)):
                if variance[i] > baseline * 1.1:
                    return float(t[i])

        # 方法3：找方差最大值之前的点
        max_idx = np.argmax(variance)
        if max_idx > 0 and max_idx < len(variance) - 1:
            return float(t[max_idx - 1])

        return None

    def analyze_video_sequence(self, frames, fps=30):
        """
        分析视频序列，估算泪膜形成和破裂时间

        :param frames: 图像帧列表 [frame1, frame2, ...]
        :param fps: 帧率
        :return: dict 包含形成时间、破裂时间、分析数据
        """
        if not frames or fps <= 0:
            return None

        rms_sequence = []
        variance_sequence = []
        center_sequence = []

        for frame in frames:
            # 转为灰度
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame

            # 检测中心（使用detector如果可用）
            center = None
            if self.detector:
                mask = self.detector._find_central_region(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) if len(frame.shape) == 3 else frame)
                if mask is not None:
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                    if contours:
                        best_contour = max(contours, key=cv2.contourArea)
                        M = cv2.moments(best_contour)
                        if M["m00"] > 0:
                            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

            max_radius = 200  # 默认值

            # 计算RMS和方差
            rms = self.calculate_zernike_rms(gray, center, max_radius)
            variance = self.calculate_ring_variance(gray, center, max_radius)

            rms_sequence.append(rms)
            variance_sequence.append(variance)
            center_sequence.append(center)

        timestamps = np.arange(len(rms_sequence)) / fps

        # 估计时间
        formation_time = self.estimate_formation_time(rms_sequence, timestamps)
        break_time = self.estimate_break_time(variance_sequence, timestamps)

        return {
            'formation_time': formation_time,
            'break_time': break_time,
            'rms_sequence': rms_sequence,
            'variance_sequence': variance_sequence,
            'center_sequence': center_sequence,
            'timestamps': timestamps
        }

    def _polar_transform(self, img, center, max_radius, output_size=None):
        """极坐标变换"""
        if output_size is None:
            output_size = (int(max_radius + 50), 360)
        flags = cv2.INTER_LINEAR + cv2.WARP_POLAR_LINEAR
        polar_img = cv2.warpPolar(img, dsize=output_size,
                                  center=center, maxRadius=max_radius + 50, flags=flags)
        return polar_img


class VideoProcessor:
    """视频处理器 - 用于UI层的视频实时检测"""

    def __init__(self, placido_detector):
        self.detector = placido_detector
        self.video_detector = VideoDetector(placido_detector)

        # 检测参数
        self.vote_threshold = 1  # 投票阈值
        self.skip_frames = 0  # 跳过的帧数

        # 状态变量
        self.rms_sequence = []
        self.variance_sequence = []
        self.break_periods = []
        self.is_breaking = False
        self.break_start_time = 0.0
        self.break_vote_count = 0
        self.frame_count = 0

    def reset(self):
        """重置状态"""
        self.rms_sequence = []
        self.variance_sequence = []
        self.break_periods = []
        self.is_breaking = False
        self.break_start_time = 0.0
        self.break_vote_count = 0
        self.frame_count = 0

    def process_frame(self, frame, fps):
        """
        处理单帧

        :param frame: 图像帧
        :param fps: 帧率
        :return: dict 包含检测结果
        """
        self.frame_count += 1
        current_time = self.frame_count / fps

        # 跳过初始帧
        if self.frame_count <= self.skip_frames:
            return {
                'has_break': False,
                'center': None,
                'rms': float('inf'),
                'variance': 0.0,
                'time': current_time
            }

        # 检测
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        center, vis_img, mask, has_break, distortion = self.detector.detect(frame)
        # ========== 修复：删除取反操作 ==========
        # 原代码：has_break = not has_break  (已删除)

        # 计算RMS和方差
        max_radius = self.detector.max_roi
        rms = self.video_detector.calculate_zernike_rms(gray, center, max_radius)
        variance = self.video_detector.calculate_ring_variance(gray, center, max_radius)

        # 投票机制
        if has_break:
            self.break_vote_count += 1
        else:
            self.break_vote_count = 0

        final_has_break = self.break_vote_count >= self.vote_threshold

        # 状态机：记录破裂时间段
        if final_has_break:
            if not self.is_breaking:
                self.is_breaking = True
                self.break_start_time = current_time
        else:
            if self.is_breaking:
                self.is_breaking = False
                self.break_periods.append((self.break_start_time, current_time))

        # 保存序列
        self.rms_sequence.append(rms)
        self.variance_sequence.append(variance)

        return {
            'has_break': final_has_break,
            'center': center,
            'rms': rms,
            'variance': variance,
            'distortion': distortion,
            'vis_img': vis_img,      # 添加可视化图像，方便UI直接显示
            'time': current_time
        }

    def finish(self, fps):
        """
        视频结束处理

        :param fps: 帧率
        :return: dict 包含最终结果
        """
        # 检查是否还处于破裂状态
        if self.is_breaking:
            total_time = self.frame_count / fps
            self.break_periods.append((self.break_start_time, total_time))

        # 计算时间估计
        timestamps = np.arange(len(self.rms_sequence)) / fps

        formation_time = self.video_detector.estimate_formation_time(
            self.rms_sequence, timestamps
        )
        break_time = self.video_detector.estimate_break_time(
            self.variance_sequence, timestamps
        )

        # 首次破裂时间
        first_break_time = self.break_periods[0][0] if self.break_periods else None

        return {
            'break_periods': self.break_periods,
            'first_break_time': first_break_time,
            'formation_time': formation_time,
            'break_time': break_time,
            'rms_sequence': self.rms_sequence,
            'variance_sequence': self.variance_sequence,
            'total_frames': self.frame_count
        }