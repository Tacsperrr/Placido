import cv2
import numpy as np
import time as timer
from matplotlib import pyplot as plt
from skimage.exposure import adjust_gamma
from scipy.signal import find_peaks
from scipy.ndimage import uniform_filter1d
from scipy.optimize import curve_fit
import visual


def cv2_imread_chinese(path):
    """
    读取图片，解决OpenCV对中文路径的支持问题
    :param path: 图片路径（可含中文）
    :return: 图片数组（None=读取失败）
    """
    try:
        stream = np.fromfile(path, dtype=np.uint8)
        image = cv2.imdecode(stream, cv2.IMREAD_COLOR)
        return image
    except Exception as e:
        print(f"读取图片失败（路径/文件问题）: {e}")
        return None


class PlacidoDetector:
    def __init__(self):
        self.gray_threshold = 47  # 灰度阈值
        self.max_roi = 400
        self.morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        # ========== 多特征融合加权投票机制 ==========
        # 恢复均衡的特征权重（宽度核心，灰度辅助）
        self.feature_weights = {
            'width': 0.55,  # 恢复核心权重，避免灰度过度干扰
            'gray_uniformity': 0.30,  # 降回原权重，减少噪声敏感
            'edge_continuity': 0.15  # 边缘连续性
        }
        # 双阈值设计（平衡版）- 放宽阈值，避免过度过滤
        self.filter_threshold = 0.56  # 从0.60降回0.52，减少误判
        self.supplement_threshold = 0.35  # 从0.45降回0.38，平衡漏检
        # 破裂判定参数（恢复合理值）
        self.break_angle_threshold = 2  # 连续异常角度阈值
        self.break_span_min = 4  # 最小破裂跨度，从6改回4
        # 性能优化参数
        self.angle_step = 2  # 角度步进
        self.use_vectorized = True  # 向量化计算开关

    def _enhance_contrast(self, image):
        """增强中心区域对比度"""
        print(f"Input image shape: {image.shape}")  # 输出输入图像的形状

        if len(image.shape) == 3 and image.shape[2] == 3:
            # Gamma校正
            gamma = 0.7
            inv_gamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)], dtype="uint8")
            gamma_corrected = cv2.LUT(image, table)

            # 转换为灰度
            gray = cv2.cvtColor(gamma_corrected, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)

            enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
            return enhanced_bgr

        # 如果输入图像是单通道灰度图像，直接进行CLAHE
        elif len(image.shape) == 2:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(image)
            return enhanced
        else:
            raise ValueError("输入的图像格式不正确")

    # ========== 性能优化：简化adaptive_enhancement ==========
    def adaptive_enhancement(self, gray_img):
        """自适应图像增强管道 - 优化版"""
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(16, 16))
        enhanced = clahe.apply(gray_img)

        mean_val = np.mean(enhanced)
        if mean_val < 80:
            gamma = 1.5
            enhanced = adjust_gamma(enhanced, gamma=gamma)
        elif mean_val > 180:
            gamma = 0.7
            enhanced = adjust_gamma(enhanced, gamma=gamma)

        filter_size = 3
        filtered = cv2.medianBlur(enhanced, filter_size)
        return filtered

    # ========== 多特征融合：宽度特征计算（核心）==========
    def _calculate_width_feature(self, r_start, r_end, standard_width):
        if standard_width <= 0:
            return 0.0
        actual_width = r_end - r_start
        width_ratio = actual_width / standard_width
        score = np.exp(-((width_ratio - 1) ** 2) / 0.25)
        if width_ratio < 0.3 or width_ratio > 1.8:
            score *= 0.3
        elif width_ratio < 0.4 or width_ratio > 1.6:
            score *= 0.5
        elif width_ratio < 0.5 or width_ratio > 1.5:
            score *= 0.7
        return np.clip(score, 0, 1)

    # ========== 多特征融合：灰度均匀度特征（增强）==========
    def _calculate_gray_uniformity_feature(self, roi, center, angle_deg, r_start, r_end):
        """计算灰度均匀度特征 - 破裂区域灰度变化更剧烈"""
        angle = np.deg2rad(angle_deg)
        gray_vals = []
        step = max(1, int((r_end - r_start) / 15))
        for r in range(int(r_start), int(r_end), step):
            x = int(center[0] + r * np.cos(angle))
            y = int(center[1] + r * np.sin(angle))
            if 0 <= x < roi.shape[1] and 0 <= y < roi.shape[0]:
                gray_vals.append(roi[y, x])
        if len(gray_vals) < 5:
            return 1.0

        gray_arr = np.array(gray_vals, dtype=np.float32)
        mean_val = np.mean(gray_arr)
        if mean_val > 0:
            cv = np.std(gray_arr) / mean_val  # 变异系数
        else:
            cv = 0
        # 平衡灰度敏感度
        uniformity_score = np.clip(1 - cv * 1.3, 0, 1)
        return uniformity_score

    # ========== 多特征融合：边缘连续性特征（增强）==========
    def _calculate_edge_continuity_feature(self, derivative, inner_idx, outer_idx):
        """计算边缘连续性特征 - 破裂区域边缘不连续"""
        if len(inner_idx) == 0 or len(outer_idx) == 0:
            return 0.0

        inner_sorted = np.sort(inner_idx)
        outer_sorted = np.sort(outer_idx)

        inner_gaps = np.diff(inner_sorted) if len(inner_sorted) > 1 else np.array([0])
        outer_gaps = np.diff(outer_sorted) if len(outer_sorted) > 1 else np.array([0])

        gap_threshold = 2
        inner_continuity = 1 - np.mean(inner_gaps > gap_threshold) if len(inner_gaps) > 0 else 1.0
        outer_continuity = 1 - np.mean(outer_gaps > gap_threshold) if len(outer_gaps) > 0 else 1.0

        inner_density = len(inner_idx) / max(1, inner_sorted[-1] - inner_sorted[0]) if len(inner_idx) > 1 else 0
        outer_density = len(outer_idx) / max(1, outer_sorted[-1] - outer_sorted[0]) if len(outer_idx) > 1 else 0
        density_score = np.clip((inner_density + outer_density) / 10, 0, 1)

        continuity_score = (inner_continuity * 0.6 + outer_continuity * 0.3 + density_score * 0.1)
        return np.clip(continuity_score, 0, 1)

    # ========== 多特征融合：加权投票机制 ==========
    def _fused_break_score(self, width_score, gray_score, edge_score):
        """
        多特征融合加权投票
        返回综合得分：0-1之间，低于阈值判定为破裂
        """
        weights = self.feature_weights
        fused_score = (
            weights['width'] * width_score +
            weights['gray_uniformity'] * gray_score +
            weights['edge_continuity'] * edge_score
        )
        return fused_score

    # ========== 新增：扭曲值(Distortion)计算 ==========
    def calculate_distortion(self, roi, center, layers):
        """
        计算泪膜扭曲值 - 基于普拉多环的不规则程度
        扭曲值越大，表示泪膜越不稳定

        :param roi: 灰度图像
        :param center: 中心点 (x, y)
        :param layers: 环层信息 [(r_min, r_max), ...]
        :return: distortion_value (float): 扭曲值，0表示无扭曲
        """
        if not layers or len(layers) == 0:
            return 0.0

        max_radius = self.max_roi
        angle_step = self.angle_step
        num_angles = 360

        # 存储每个角度的环层半径
        inner_radii = []
        outer_radii = []

        cos_sin_cache = {ad: (np.cos(np.deg2rad(ad)), np.sin(np.deg2rad(ad)))
                        for ad in range(0, num_angles, angle_step)}

        for angle_deg in range(0, num_angles, angle_step):
            cos_a, sin_a = cos_sin_cache[angle_deg]
            profile_intensity = []

            # 采样获取径向强度分布
            for r in range(0, max_radius, 2):
                x = int(center[0] + r * cos_a)
                y = int(center[1] + r * sin_a)
                if 0 <= x < roi.shape[1] and 0 <= y < roi.shape[0]:
                    profile_intensity.append(roi[y, x])
                else:
                    break

            if len(profile_intensity) < 10:
                continue

            intensity = np.array(profile_intensity).astype(float)
            kernel_size = min(9, len(intensity) // 2 * 2 + 1)
            intensity_smooth = cv2.GaussianBlur(intensity.reshape(1, -1), (kernel_size, 1), 2).flatten()
            derivative = np.gradient(intensity_smooth)

            # 找峰值和谷值（内外边缘）
            peaks, _ = find_peaks(derivative, distance=8, prominence=0.8, width=3)
            valleys, _ = find_peaks(-derivative, distance=8, prominence=0.8, width=3)

            # 筛选有效的峰值和谷值
            valid_peaks = [p for p in peaks if derivative[p] > 0.6]
            valid_valleys = [v for v in valleys if derivative[v] < -0.6]

            if valid_peaks and valid_valleys:
                # 取最内层和最外层
                inner_r = min(valid_peaks) * 2  # 乘2是因为采样步长为2
                outer_r = max(valid_valleys) * 2

                # 过滤异常值
                if 20 < inner_r < max_radius and 20 < outer_r < max_radius:
                    inner_radii.append(inner_r)
                    outer_radii.append(outer_r)

        if len(inner_radii) < 10 or len(outer_radii) < 10:
            return 0.0

        # 计算扭曲值：基于半径的标准差
        inner_radii = np.array(inner_radii)
        outer_radii = np.array(outer_radii)

        # 计算每个环层半径的标准差
        inner_std = np.std(inner_radii)
        outer_std = np.std(outer_radii)

        # 计算每个环层半径与平均值的偏差百分比
        inner_mean = np.mean(inner_radii)
        outer_mean = np.mean(outer_radii)

        inner_cv = inner_std / inner_mean if inner_mean > 0 else 0  # 变异系数
        outer_cv = outer_std / outer_mean if outer_mean > 0 else 0

        # 综合扭曲值（归一化到0-1）
        # 论文中扭曲值随时间增加，这里用环的不规则程度作为扭曲指标
        distortion_value = (inner_cv + outer_cv) / 2

        # 归一化：假设正常扭曲值在0.05以内，超过0.2表示严重扭曲
        distortion_normalized = min(distortion_value / 0.2, 1.0)

        return float(distortion_normalized)

    def _vote_break_judgment(self, angle_scores):
        """
        加权投票机制判断是否破裂
        :param angle_scores: list of (angle, fused_score)
        :return: list of break ranges [(start_angle, end_angle), ...]  # 标准化为不跨越0°
        """
        if not angle_scores:
            return []

        breaks = []
        num_angles = 360
        scores = np.array([s for _, s in angle_scores])
        angles = np.array([a for a, _ in angle_scores])

        # 使用双阈值判断
        is_abnormal = scores < self.supplement_threshold
        is_uncertain = (scores >= self.supplement_threshold) & (scores < self.filter_threshold)

        # 合并abnormal + uncertain角度
        abnormal_angles = set(angles[is_abnormal])
        uncertain_angles = set(angles[is_uncertain])
        all_abnormal = sorted(abnormal_angles | uncertain_angles)

        if not all_abnormal:
            return []

        # 找连续异常区域
        i = 0
        while i < len(all_abnormal):
            start = all_abnormal[i]
            end = start
            j = i + 1

            # 合并间隔≤5度
            while j < len(all_abnormal):
                if all_abnormal[j] - end <= 5:
                    end = all_abnormal[j]
                else:
                    break
                j += 1

            span = (end - start + 1) % num_angles
            # 最小破裂跨度从6改回4
            if span >= 2:
                breaks.append([start, end])

            i = j

        # ========== 修复：处理跨越0°的破裂区域 ==========
        normalized_breaks = []
        for start, end in breaks:
            if start <= end:
                normalized_breaks.append((start, end))
            else:
                # 跨越0°，拆分为两段
                normalized_breaks.append((start, 359))
                normalized_breaks.append((0, end))
        return normalized_breaks

    # ========== 性能优化：简化_find_central_region ==========
    def _find_central_region(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 使用合适的自适应阈值
        binary_pre = cv2.adaptiveThreshold(gray, 255,
                                           cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY_INV, 51, 5)

        # 调整开闭运算的结构元素
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

        stage1 = cv2.morphologyEx(binary_pre, cv2.MORPH_OPEN, kernel_open, iterations=2)
        cleaned_pre = cv2.morphologyEx(stage1, cv2.MORPH_CLOSE, kernel_close)

        contours, _ = cv2.findContours(cleaned_pre, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_contours = []

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w > 0 and h > 0:
                aspect_ratio = max(w / h, h / w)
                if aspect_ratio <= 2.0:  # 放宽比例
                    valid_contours.append(cnt)

        if valid_contours:
            max_contour = max(valid_contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(max_contour)
            roi = gray[y:y + h, x:x + w]
        else:
            roi = gray

        # 调用对比度增强
        enhanced = self._enhance_contrast(roi)
        binary = cv2.threshold(enhanced, self.gray_threshold, 255, cv2.THRESH_BINARY_INV)[1]
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, self.morph_kernel, iterations=2)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, self.morph_kernel, iterations=1)

        if valid_contours:
            final_mask = np.zeros_like(gray)
            final_mask[y:y + h, x:x + w] = cleaned
            return final_mask
        return cleaned

    def _score_contour(self, cnt, img_center):
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)

        circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        diameter = 2 * radius

        if diameter <= 10:
            return 0

        cnt_center = np.array([x, y])
        position_score = 1 - (np.linalg.norm(cnt_center - img_center) / np.linalg.norm(img_center))

        aspect_ratio = [cv2.boundingRect(cnt)[2] / cv2.boundingRect(cnt)[3] for cnt in cnt]
        aspect_score = 1 - abs(np.mean(aspect_ratio) - 1.0)

        area_score = area / (img_center[0] * img_center[1])  # 归一化面积评分
        return (circularity * 0.5 + aspect_score * 0.3 + position_score * 0.2 + area_score * 0.1)
    def generate_terrain_map(self, img1, x_c, y_c):
        """生成地形图"""
        h, w = img1.shape[:2]
        vis_img = img1.copy()
        roi_radius = self.max_roi
        mask = np.zeros((h, w), np.uint8)
        cv2.circle(mask, (x_c, y_c), roi_radius, 255, -1)
        roi_image = cv2.bitwise_and(img1, vis_img, mask=mask)
        roi_gray = cv2.cvtColor(roi_image, cv2.COLOR_RGB2GRAY)
        roi_enhanced = self.adaptive_enhancement(roi_gray)

        try:
            breaks, layers, distortion = self.detect_breaks_combined((x_c, y_c), roi_gray, roi_enhanced)
        except Exception as e:
            print(f"detect_breaks_combined失败: {e}")
            breaks, layers, distortion = [], [], 0.0

        try:
            vis_roi = visual.visualize_analysis(roi_image.copy(), (x_c, y_c), breaks, layers)
        except Exception as e:
            print(f"可视化失败: {e}")
            vis_roi = roi_image.copy()

        img_f = img1.copy()
        img_f = np.where(mask[..., None].astype(bool), vis_roi, img_f)
        scale_factor = 1.5
        roi_radius = int(self.max_roi * scale_factor)
        h, w = img_f.shape[:2]
        aspect_ratio = w / h
        crop_h = int(2 * roi_radius)
        crop_w = int(crop_h * aspect_ratio)
        crop_w += crop_w % 2
        crop_h += crop_h % 2
        x1 = max(x_c - crop_w // 2, 0)
        y1 = max(y_c - crop_h // 2, 0)
        x2 = min(x1 + crop_w, w)
        y2 = min(y1 + crop_h, h)
        if x2 - x1 < crop_w:
            x1 = max(x2 - crop_w, 0)
        if y2 - y1 < crop_h:
            y1 = max(y2 - crop_h, 0)
        cropped_img_f = img_f[y1:y2, x1:x2]

        return cropped_img_f, breaks, distortion

    def detect_breaks_combined(self, center, raw_gray_img, enhanced_gray_img):
        """组合检测环层和破裂 - 优化版：缓存极坐标变换"""
        if center is None:
            return [], [], 0.0

        # ========== 修复1：一次计算极坐标，多处复用 ==========
        polar_cache = {}
        try:
            # 计算一次极坐标变换，供多个函数使用
            raw_polar = self.polar_transform(raw_gray_img, center)
            if raw_polar is None or raw_polar.size == 0:
                return [], [], 0.0
            polar_cache['raw_polar'] = raw_polar
            polar_cache['enhanced_polar'] = self.polar_transform(enhanced_gray_img, center)
        except Exception as e:
            print(f"极坐标变换失败: {e}")
            return [], [], 0.0

        # ========== 修复4：环层检测时过滤过小/过大的环 ==========
        try:
            layers = self.detect_ring_layers(center, raw_gray_img, polar_cache)
            # 过滤无效环层：半径过小(<13)或过大(>max_roi)
            min_layer_radius = 13
            max_layer_radius = self.max_roi
            layers = [(r_min, r_max) for r_min, r_max in layers
                     if min_layer_radius <= r_min and r_max <= max_layer_radius]
        except Exception as e:
            print(f"detect_ring_layers失败: {e}")
            layers = []

        try:
            breaks = self.detect_breaks(enhanced_gray_img, center, layers)
        except Exception as e:
            print(f"detect_breaks失败: {e}")
            breaks = []

        # 计算扭曲值
        distortion = 0.0
        try:
            distortion = self.calculate_distortion(enhanced_gray_img, center, layers)
        except Exception as e:
            print(f"扭曲值计算失败: {e}")
            distortion = 0.0

        return breaks, layers, distortion

    # ========== 性能优化：简化detect_ring_layers ==========
    def detect_ring_layers(self, center, gray_img, polar_cache=None):
        """检测环层 - 优化版：减少计算量，添加异常处理和环层过滤"""
        if center is None:
            return []
        if gray_img is None or gray_img.size == 0:
            return []

        try:
            # 优先使用缓存的极坐标，避免重复计算
            if polar_cache is not None and 'raw_polar' in polar_cache:
                polar = polar_cache['raw_polar']
            else:
                gray_img = cv2.medianBlur(gray_img, 3)
                enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray_img)
                polar = self.polar_transform(enhanced, center)

            if polar is None or polar.size == 0:
                return []

            horizontal_proj = cv2.reduce(polar, 0, cv2.REDUCE_AVG, dtype=cv2.CV_32F).flatten()
            if len(horizontal_proj) < 2:
                return []

            kernel_size = min(9, len(horizontal_proj) // 2 * 2 + 1)
            proj_smoothed = cv2.GaussianBlur(horizontal_proj.reshape(1, -1), (kernel_size, 1), 2).flatten()
            derivative = np.gradient(proj_smoothed)

            peaks, _ = find_peaks(derivative, distance=8, prominence=0.12, width=2)
            valleys, _ = find_peaks(-derivative, distance=8, prominence=0.12, width=2)

            valid_pairs = []
            used_valleys = set()
            for peak in peaks:
                if peak < 10:  # 过滤半径过小的环
                    continue
                candidate_valleys = [v for v in valleys if v > peak and v not in used_valleys]
                if candidate_valleys:
                    closest_valley = candidate_valleys[0]
                    if derivative[peak] > 0 and derivative[closest_valley] < 0:
                        valid_pairs.append((peak, closest_valley))
                        used_valleys.add(closest_valley)

            filtered_layers = valid_pairs

            # 环层过滤
            min_layer_radius = 13
            max_layer_radius = self.max_roi if self.max_roi > 0 else 300
            filtered_layers = [(r_min, r_max) for r_min, r_max in filtered_layers
                              if min_layer_radius <= r_min and r_max <= max_layer_radius]

            print(f"检测到 {len(filtered_layers)} 个有效环层")
            return filtered_layers
        except Exception as e:
            print(f"detect_ring_layers异常: {e}")
            return []

    # ========== 优化后的detect_breaks：多特征融合 + 性能优化 ==========
    def detect_breaks(self, roi, center, layers):
        """检测破裂 - 添加异常处理"""
        try:
            if center is None:
                return []
            if roi is None or roi.size == 0:
                return []
            if not layers:
                print("警告：无有效环层，跳过破损检测")
                return []
        except Exception as e:
            print(f"detect_breaks参数检查异常: {e}")
            return []

        max_radius = self.max_roi
        num_angles = 360
        angle_step = self.angle_step

        standard_widths = [r_max - r_min for r_min, r_max in layers]
        if not standard_widths:
            print("警告：无有效环层宽度，跳过破损检测")
            return []
        # ========== 修复：环层数量不足时直接返回 ==========
        if len(standard_widths) < 2:
            print("警告：环层数量不足2，无法检测破裂")
            return []

        standard_width00 = [standard_widths[0] * 0.25, standard_widths[0] * 3.8, standard_widths[0] * 0.3]
        num_expected_layers = min(len(standard_widths) - 1, 10)
        every_width = {}
        finish_counts = {}

        # 三角函数缓存
        cos_sin_cache = {ad: (np.cos(np.deg2rad(ad)), np.sin(np.deg2rad(ad)))
                        for ad in range(0, num_angles, angle_step)}

        for angle_deg in range(0, num_angles, angle_step):
            cos_a, sin_a = cos_sin_cache[angle_deg]
            profile_r = []
            profile_intensity = []
            r_step = 2
            for r in range(0, max_radius, r_step):
                x = int(center[0] + r * cos_a)
                y = int(center[1] + r * sin_a)
                if 0 <= x < roi.shape[1] and 0 <= y < roi.shape[0]:
                    profile_r.append(r)
                    profile_intensity.append(roi[y, x])
                else:
                    break

            if len(profile_intensity) < 5:
                every_width[angle_deg] = []
                finish_counts[angle_deg] = 0
                continue

            radii = np.array(profile_r)
            intensity = np.array(profile_intensity).astype(float)
            kernel_size = min(9, len(intensity) // 2 * 2 + 1)
            intensity_smooth = cv2.GaussianBlur(intensity.reshape(1, -1), (kernel_size, 1), 2).flatten()
            derivative = np.gradient(intensity_smooth)
            inner_indices_all, _ = find_peaks(derivative, distance=8, prominence=0.8, width=3)
            outer_indices_all, _ = find_peaks(-derivative, distance=8, prominence=0.8, width=3)
            inner_indices = [i for i in inner_indices_all if derivative[i] > 0.5]
            outer_indices = [i for i in outer_indices_all if derivative[i] < -0.5]

            if len(inner_indices) == 0 or len(outer_indices) == 0:
                every_width[angle_deg] = []
                finish_counts[angle_deg] = 0
                continue

            bands = []
            if inner_indices and outer_indices:
                sorted_inner = sorted(inner_indices, key=lambda x: radii[x])
                sorted_outer = sorted(outer_indices, key=lambda x: radii[x])
                i, j = 0, 0
                used_inner = set()
                used_outer = set()
                while i < len(sorted_inner) and j < len(sorted_outer):
                    while i < len(sorted_inner) and sorted_inner[i] in used_inner:
                        i += 1
                    if i >= len(sorted_inner):
                        break
                    curr_peak = sorted_inner[i]
                    r_peak = radii[curr_peak]
                    valid_valley = None
                    while j < len(sorted_outer):
                        curr_valley = sorted_outer[j]
                        if curr_valley in used_outer:
                            j += 1
                            continue
                        r_valley = radii[curr_valley]
                        width = r_valley - r_peak
                        if (r_valley > r_peak and
                                standard_width00[0] <= width <= standard_width00[1]):
                            valid_valley = curr_valley
                            break
                        j += 1
                    if valid_valley is None:
                        i += 1
                        continue
                    bands.append((r_peak, radii[valid_valley]))
                    used_inner.add(curr_peak)
                    used_outer.add(valid_valley)
                    next_peak = None
                    for k in range(i + 1, len(sorted_inner)):
                        if sorted_inner[k] in used_inner:
                            continue
                        if radii[sorted_inner[k]] > radii[valid_valley]:
                            next_peak = k
                            break
                    i = next_peak if next_peak is not None else len(sorted_inner)
            merged_bands = []
            for band in bands:
                if not merged_bands:
                    merged_bands.append(list(band))
                else:
                    prev_start, prev_end = merged_bands[-1]
                    curr_start, curr_end = band
                    if curr_start - prev_end < standard_width00[2]:
                        merged_bands[-1][1] = max(prev_end, curr_end)
                    else:
                        merged_bands.append(list(band))
            every_width[angle_deg] = [tuple(b) for b in merged_bands]
            finish_counts[angle_deg] = len(merged_bands)

        # ========== 多特征融合：加权投票机制判断破裂 ==========
        breaks = []
        roi_h, roi_w = roi.shape

        for layer_idx in range(1, num_expected_layers):
            angle_scores = []

            for angle in range(0, num_angles, angle_step):
                band_list = every_width.get(angle, [])
                num_detected_bands = finish_counts.get(angle, 0)

                if layer_idx >= num_detected_bands or not band_list:
                    # ========== 修复：默认分数改为1.0（表示正常） ==========
                    angle_scores.append((angle, 1.0))
                    continue

                r_start, r_end = band_list[layer_idx]
                standard = standard_widths[layer_idx]

                # 1. 宽度特征
                width_score = self._calculate_width_feature(r_start, r_end, standard)

                # 2. 灰度均匀度特征
                gray_score = self._calculate_gray_uniformity_feature(
                    roi, center, angle, r_start, r_end
                )

                # 3. 边缘连续性特征 - 恢复宽松的判定区间
                actual_width = r_end - r_start
                width_ratio = actual_width / standard if standard > 0 else 1
                if 0.55 <= width_ratio <= 1.45:
                    edge_score = 1.0
                elif 0.45 <= width_ratio <= 1.55:
                    edge_score = 0.7
                else:
                    edge_score = 0.4

                # 加权融合
                fused_score = self._fused_break_score(width_score, gray_score, edge_score)
                angle_scores.append((angle, fused_score))

            break_ranges = self._vote_break_judgment(angle_scores)

            if break_ranges:
                breaks.append((layer_idx, break_ranges))

        print("breaks", breaks)
        return breaks

    # ========== 性能优化：简化polar_transform ==========
    def polar_transform(self, img, center, output_size=None):
        """极坐标变换 - 优化版：减少输出分辨率"""
        max_radius = self.max_roi  # 修复：使用 self.max_roi 而非 +50，保持一致
        if output_size is None:
            output_size = (int(max_radius), 360)
        flags = cv2.INTER_LINEAR + cv2.WARP_POLAR_LINEAR
        polar_img = cv2.warpPolar(img, dsize=output_size,
                                  center=center, maxRadius=max_radius, flags=flags)
        return polar_img

    def _get_contour_center(self, contour):
        """获取轮廓中心"""
        M = cv2.moments(contour)
        if M["m00"] > 0:
            return (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        else:
            (x, y), _ = cv2.minEnclosingCircle(contour)
            return (int(x), int(y))

    def detect(self, image_input):
        """主检测函数"""
        start = timer.time()
        if isinstance(image_input, str):
            image = cv2_imread_chinese(image_input)
            if image is None:
                return None, "无法读取图像（路径含中文/文件不存在/文件损坏）", None, False
        elif isinstance(image_input, np.ndarray):
            image = image_input.copy()
        else:
            return None, "无效的输入类型", None, False

        h, w = image.shape[:2]
        self.max_roi = min(int(min(h, w) / 5), 400)
        print("h=", h, " w=", w, "max_roi=", self.max_roi)
        img_center = np.array([w // 2, h // 2])
        vis_img = image.copy()

        mask = self._find_central_region(image)
        kernel_circle = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        cleaned_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_circle)
        cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel_circle)
        contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        center = None
        terrain_map = vis_img
        has_break = False
        if len(contours) == 0:
            print("警告：未检出眼部轮廓")
        else:
            try:
                best_contour = max(contours, key=lambda c: self._score_contour(c, img_center))
                center = self._get_contour_center(best_contour)
            except Exception as e:
                print(f"轮廓处理失败: {e}")
                center = None

        if center is not None:
            cx, cy = center
            try:
                img1 = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
                terrain_map, breaks, distortion = self.generate_terrain_map(img1, cx, cy)
                has_break = len(breaks) > 0
            except Exception as e:
                print(f"地形图生成错误: {e}")
                terrain_map = vis_img
                has_break = False
                distortion = 0.0
        else:
            distortion = 0.0

        print("检测时间", timer.time() - start)
        print(f"扭曲值: {distortion:.4f}")
        return center, terrain_map, mask, has_break, distortion