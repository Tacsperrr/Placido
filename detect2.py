import cv2
import numpy as np
import time as timer
from matplotlib import pyplot as plt
from skimage.exposure import adjust_gamma
from scipy.signal import find_peaks
import visual


# 在 detect2.py 文件开头添加这个函数
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
        self.gray_threshold = 35  # 灰度阈值
        self.max_roi=400
        self.morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))# 形态学参数

    def _enhance_contrast(self, image):
        """增强中心区域对比度"""
        # 使用伽马校正增强暗部细节
        gamma = 0.7
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255
                          for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(image, table)
    def adaptive_enhancement(self, gray_img):
        """自适应图像增强管道"""
        # CLAHE自适应直方图均衡
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray_img)

        # 动态伽马校正
        mean_val = np.mean(enhanced)
        gamma = 1.6 if mean_val < 80 else (0.6 if mean_val > 180 else 1.0)
        gamma_corrected = adjust_gamma(enhanced, gamma=gamma)

        # 自适应滤波
        filter_size = 5 if gamma_corrected.shape[0] > 1000 else 3
        filtered = cv2.medianBlur(gamma_corrected, filter_size)

        return filtered

    def _find_central_region(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # 二值化
        binary_pre = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 51, 5
        )
        # visual.essay_image(image.copy(), binary_pre)
        # 形态学
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        stage1 = cv2.morphologyEx(binary_pre, cv2.MORPH_OPEN, kernel_open, iterations=1)
        kernel_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        cleaned_pre = cv2.morphologyEx(stage1, cv2.MORPH_CLOSE, kernel_ellipse)
        # visual.essay_image(binary_pre,cleaned_pre)

        contours, _ = cv2.findContours(cleaned_pre, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_contours = []
        for cnt in contours:
            # 计算轮廓外接矩形
            x, y, w, h = cv2.boundingRect(cnt)
            # 计算宽高比（最大边/最小边）
            if w == 0 or h == 0:
                continue
            aspect_ratio = max(w / h, h / w)
            # 筛选轮廓
            if aspect_ratio <= 1.8:
                valid_contours.append(cnt)
        # con=np.ones_like(cleaned_pre) * 1
        # img=cv2.drawContours(con, valid_contours, -1, 255, 2)
        # visual.essay_image(cleaned_pre,img)
        # 选择符合条件的最大的轮廓
        if valid_contours:
            max_contour = max(valid_contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(max_contour)
            roi = gray[y:y + h, x:x + w]
        else:
            roi = gray
        # visual.essay_image(cleaned_pre, roi)
        enhanced = self._enhance_contrast(roi)
        binary = cv2.threshold(enhanced, self.gray_threshold, 255, cv2.THRESH_BINARY_INV)[1]
        # 形态学优化
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, self.morph_kernel, iterations=2)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, self.morph_kernel, iterations=1)

        # 使用了ROI将结果放回原图位置
        if valid_contours:
            final_mask = np.zeros_like(gray)
            final_mask[y:y + h, x:x + w] = cleaned
            return final_mask
        return cleaned

    def _score_contour(self, cnt, img_center):
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0# 圆度得分
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        diameter = 2 * radius
        if diameter <= 10:    # 过小直接排除
            return 0
        cnt_center = np.array([x, y])
        position_score = 1 - (np.linalg.norm(cnt_center - img_center) / np.linalg.norm(img_center))# 位置得分
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w / h if h != 0 else 0
        aspect_score = 1 - abs(aspect_ratio - 1.0) # 半径得分
        return (circularity * 0.6 +aspect_score * 0.2 +position_score * 0.2)

    def generate_terrain_map(self, img1, x_c, y_c):
        h, w = img1.shape[:2]
        vis_img = img1.copy()

        # ROI划定
        roi_radius = self.max_roi  # 限制最大ROI范围
        mask = np.zeros((h, w), np.uint8)
        cv2.circle(mask, (x_c, y_c), roi_radius, 255, -1)
        # cv2.imwrite('mask.png', mask)
        roi_image = cv2.bitwise_and(img1, vis_img, mask=mask)
        # roi_out=roi_image[y_c-500:y_c+500, x_c-500:x_c+500]
        # cv2.imwrite("roi.jpg", roi_out)
        # visual.essay_image(vis_img, roi_image)
        # 单通道处理
        roi_gray = cv2.cvtColor(roi_image, cv2.COLOR_RGB2GRAY)
        # 图像增强
        roi_enhanced = self.adaptive_enhancement(roi_gray)
        # plt.subplot(121)
        # plt.imshow(roi_enhanced, cmap='gray')  # 增强后的灰度图
        # plt.title("Enhanced Gray")
        # plt.subplot(122)
        # plt.imshow(cv2.cvtColor(roi_image, cv2.COLOR_BGR2RGB))  # 原图转RGB显示
        # plt.title("Original ROI")
        # plt.show()

        breaks,layers = self.detect_breaks_combined((x_c, y_c), roi_gray, roi_enhanced)
        # print(breaks, layers)

        # 可视化系统
        vis_roi = visual.visualize_analysis(roi_image.copy(),(x_c, y_c), breaks, layers)
        # cv2.imshow("ROI", vis_roi)

        # 合成最终图像 (将可视化结果放回原图)
        img_f = img1.copy()
        # 只更新圆形ROI区域
        img_f = np.where(mask[..., None].astype(bool), vis_roi, img_f)
        # 设定 ROI 扩展比例（例如 1.5 倍半径）
        scale_factor = 1.5
        roi_radius = int(self.max_roi * scale_factor)

        # 原图尺寸和长宽比
        h, w = img_f.shape[:2]
        aspect_ratio = w / h

        # 基于半径扩展，按原图比例计算裁剪框尺寸
        crop_h = int(2 * roi_radius)
        crop_w = int(crop_h * aspect_ratio)

        # 确保宽高为偶数，避免 cv2 报错（可选）
        crop_w += crop_w % 2
        crop_h += crop_h % 2

        # 计算裁剪框边界
        x1 = max(x_c - crop_w // 2, 0)
        y1 = max(y_c - crop_h // 2, 0)
        x2 = min(x1 + crop_w, w)
        y2 = min(y1 + crop_h, h)

        # 若裁剪框超出图像边界，则反向调整左上角坐标
        if x2 - x1 < crop_w:
            x1 = max(x2 - crop_w, 0)
        if y2 - y1 < crop_h:
            y1 = max(y2 - crop_h, 0)

        # 执行裁剪
        cropped_img_f = img_f[y1:y2, x1:x2]

        # visual.essay_image(img1,cropped_img_f)
        # cv2.imwrite("../result/1.jpg",img_f)
        return cropped_img_f

    def detect_breaks_combined(self, center, raw_gray_img, enhanced_gray_img):
        # 1. 检测环层，内部完成增强、极坐标、还原坐标
        layers = self.detect_ring_layers(center, raw_gray_img)

        # 2. 执行破损检测
        breaks = self.detect_breaks(enhanced_gray_img, center, layers)

        # 3.5.4可视化
        # print(layers)
        # visual.debug_breaks_354(raw_gray_img, center, layers, breaks)
        return breaks, layers

    def detect_ring_layers(self, center, gray_img):
        gray_img = cv2.medianBlur(gray_img, 5)
        enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(gray_img)
        # 极坐标变换
        polar = self.polar_transform(enhanced, center)
        # visual.essay_image(enhanced,polar)
        # 水平投影
        horizontal_proj = cv2.reduce(polar, 0, cv2.REDUCE_AVG, dtype=cv2.CV_32F).flatten()
        if len(horizontal_proj) < 2:
            return []
        # 平滑 导数计算
        kernel_size = min(15, len(horizontal_proj) // 2 * 2 + 1)
        proj_smoothed = cv2.GaussianBlur(horizontal_proj.reshape(1, -1), (kernel_size, 1),
                                         3).flatten()
        derivative = np.gradient(proj_smoothed)
        # visual.debug_layers_der(polar,derivative,horizontal_proj,proj_smoothed)
        # 峰值检测
        peaks, _ = find_peaks(derivative, distance=10, prominence=0.15, width=2)
        valleys, _ = find_peaks(-derivative, distance=10, prominence=0.15, width=2)
        # 边界配对
        valid_pairs = []
        used_valleys = set()
        for peak in peaks:
            if peak<10: continue
            candidate_valleys = [v for v in valleys if v > peak and v not in used_valleys]
            if candidate_valleys:
                closest_valley = candidate_valleys[0]
                if derivative[peak] > 0 and derivative[closest_valley] < 0:
                    valid_pairs.append((peak, closest_valley))
                    used_valleys.add(closest_valley)

        filtered_layers = valid_pairs
        # 调试输出
        # visual.debug_visualize_horizontal(polar, derivative, filtered_layers)
        # visual.debug_polar_layers(gray_img,center,filtered_layers)
        # print(peaks, valleys)
        print(f"检测到 {len(filtered_layers)} 个有效环层")

        return filtered_layers


    def detect_breaks(self, roi, center, layers):
        max_radius = self.max_roi
        num_angles = 360
        angle_step = 1
        standard_widths = [r_max - r_min for r_min, r_max in layers]
        standard_width00=[standard_widths[0]*0.25,standard_widths[0]*10,standard_widths[0]*0.3] #采用第一条环带作为基准
        num_expected_layers = min(len(standard_widths) - 1,10)
        every_width = {}
        finish_counts = {}
        for angle_deg in range(0, num_angles, angle_step):
            angle = np.deg2rad(angle_deg)
            profile_r = []
            profile_intensity = []
            for r in range(0, max_radius):
                x = int(center[0] + r * np.cos(angle))
                y = int(center[1] + r * np.sin(angle))
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
            kernel_size = min(15, len(intensity) // 2 * 2 + 1)
            intensity_smooth = cv2.GaussianBlur(intensity.reshape(1, -1), (kernel_size, 1), 2).flatten()
            derivative = np.gradient(intensity_smooth)
            inner_indices_all, inner_props = find_peaks(derivative, distance=10, prominence=1, width=4)  # 检测峰值
            outer_indices_all, outer_props = find_peaks(-derivative, distance=10, prominence=1, width=4)
            inner_indices = [i for i in inner_indices_all if derivative[i] > 0.5]  # 筛选边缘点
            outer_indices = [i for i in outer_indices_all if derivative[i] < -0.5]
            # # ===== 可视化3.5.1数据保存 =====
            # if angle_deg in [0, 120, 240]:
            #     if not hasattr(self, 'scanline_data'):
            #         self.scanline_data = {
            #             'angles': [],
            #             'points': [],
            #             'derivatives': []
            #         }
            #
            #     # 记录扫描线端点
            #     line_points = []
            #     for r in [0, max_radius]:  # 起点和终点
            #         x = int(center[0] + r * np.cos(np.deg2rad(angle_deg)))
            #         y = int(center[1] + r * np.sin(np.deg2rad(angle_deg)))
            #         line_points.append((x, y))
            #
            #     self.scanline_data['angles'].append(angle_deg)
            #     self.scanline_data['points'].append(line_points)
            #     self.scanline_data['derivatives'].append({
            #         'r': profile_r,
            #         'derivative': derivative,
            #         'inner': inner_indices,
            #         'outer': outer_indices
            #     })
            # # ===== 结束可视化数据保存 =====
            if len(inner_indices) == 0 or len(outer_indices) == 0:
                every_width[angle_deg] = []
                finish_counts[angle_deg] = 0
                continue

            bands = []
            if inner_indices and outer_indices:  # 双指针匹配
                # 按半径升序排列
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
                    bands.append((r_peak, radii[valid_valley])) # 记录匹配对
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
                    if curr_start - prev_end < standard_width00[2]:  # 合并相邻条带
                        merged_bands[-1][1] = max(prev_end, curr_end)
                    else:
                        merged_bands.append(list(band))
            every_width[angle_deg] = [tuple(b) for b in merged_bands] # 保存结果
            finish_counts[angle_deg] = len(merged_bands)
        # # ===== 可视化代码3.5.1 =====
        # # 第一个图：导数曲线（3行1列）
        # plt.figure(figsize=(10, 12))
        # for i, angle in enumerate([0, 120, 240], 1):
        #     plt.subplot(3, 1, i)
        #     data = self.scanline_data['derivatives'][self.scanline_data['angles'].index(angle)]
        #     plt.plot(data['r'], data['derivative'], 'b-', label='Derivative')
        #     plt.scatter([data['r'][i] for i in data['inner']],
        #                 [data['derivative'][i] for i in data['inner']],
        #                 c='red', marker='o', label='Inner Edges')
        #     plt.scatter([data['r'][i] for i in data['outer']],
        #                 [data['derivative'][i] for i in data['outer']],
        #                 c='green', marker='x', label='Outer Edges')
        #     plt.grid(True)
        #     plt.legend()
        #     plt.xlabel("Radius (px)")
        #     plt.ylabel(f"Derivative Value-{angle}°")
        #
        # # 第二个图：圆形ROI扫描线
        # plt.figure(figsize=(8, 8))
        # cx, cy = int(center[0]), int(center[1])
        # radius = 400  # 根据实际ROI半径修改
        #
        # # 创建圆形掩膜
        # y, x = np.ogrid[-cy:roi.shape[0] - cy, -cx:roi.shape[1] - cx]
        # mask = x ** 2 + y ** 2 <= radius ** 2
        #
        # # 显示圆形ROI区域
        # plt.imshow(roi * mask, cmap='gray',
        #            extent=[0, roi.shape[1], roi.shape[0], 0])
        #
        # # 绘制三条扫描线（自动限制在圆形内）
        # for angle in [0, 120, 240]:
        #     # 计算扫描线终点坐标
        #     rad = np.deg2rad(angle)
        #     x_end = cx + radius * np.cos(rad)
        #     y_end = cy + radius * np.sin(rad)
        #
        #     plt.plot([cx, x_end], [cy, y_end],
        #              'r--', linewidth=2, alpha=0.8)
        #     # 添加角度标签（自动调整位置）
        #     text_x = x_end + 20 * np.cos(rad)
        #     text_y = y_end + 20 * np.sin(rad)
        #
        #     plt.text(text_x, text_y, f"{angle}°",
        #              color='yellow', fontsize=16, weight='bold',
        #              ha='center', va='center',
        #              bbox=dict(facecolor='black', alpha=0.7, pad=2, edgecolor='none'))
        #
        # # 设置显示范围为中心±半径
        # plt.xlim(cx - radius-50, cx + radius+50)
        # plt.ylim(cy + radius+50, cy - radius-50)  # 保持图像坐标系
        # plt.axis('off')
        # plt.title("ROI with Scan Lines(0/120/240°)")
        # plt.gca().set_aspect('equal')  # 保持比例
        # plt.tight_layout()
        # plt.show()
        # visual.debug_breaks_352(roi,center,every_width) #3.5.2
        # # ===== 可视化代码结束 =====
        # print("every_width", every_width)
        # print("finish_counts", finish_counts)
        # print("standard_widths", standard_widths)

        # 逐层分析宽度，判断破损
        breaks = []
        for layer_idx in range(1,num_expected_layers):
            widths = []
            for angle in range(num_angles):
                band_list = every_width.get(angle, [])
                num_detected_bands = finish_counts.get(angle, 0)
                if layer_idx < num_detected_bands:
                    r_start, r_end = band_list[layer_idx]
                    widths.append(r_end - r_start)
                else:
                    widths.append(-1)
            bad_ranges = []  # 找出破损角度段
            angle = 0
            while angle < num_angles:
                w = widths[angle]
                standard = standard_widths[layer_idx]
                if w is None or w < 0:
                    angle += 1
                    continue
                if not (0.65 * standard <= w <= 1.35 * standard): # 初始破损点判断
                    current_start = angle
                    last_valid = angle
                    miss_count = 0
                    offset = 1
                    while (angle + offset) < num_angles:
                        next_angle = angle + offset
                        w_next = widths[next_angle]
                        if w_next is None or w_next < 0: break # 如果是无效角度，强制断开
                        if not (0.65 * standard <= w_next <= 1.35 * standard):
                            last_valid = next_angle
                            miss_count = 0
                        else:
                            miss_count += 1
                            if miss_count >= 2: break # 两点连续无误 结束
                        offset += 1
                    if last_valid-current_start>=5:
                        bad_ranges.append((current_start, last_valid))
                    angle = last_valid + 1  # 跳过已检测区域
                else:
                    angle += 1

            merged = []
            for start, end in bad_ranges:
                if not merged:
                    merged.append([start, end])
                else:
                    last_start, last_end = merged[-1]
                    if start - last_end <= 3:  # 合并间隔 ≤ 3度的破损段
                        merged[-1][1] = end
                    else:
                        merged.append([start, end])
            filtered = []
            for start, end in merged:
                span = (end - start + 1) % 360
                if span >= 10: # 去除总跨度 < 10 度的破损段
                    filtered.append([start, end])

            if filtered:
                breaks.append((layer_idx, filtered))
        print("breaks", breaks)
        return breaks

    def polar_transform(self, img, center, output_size=None):
        max_radius = self.max_roi+100            # 计算最大半径
        if output_size is None:                  # 动态调整输出尺寸
            output_size = (int(max_radius), 720)
        flags = cv2.INTER_LINEAR + cv2.WARP_POLAR_LINEAR
        polar_img = cv2.warpPolar(img, dsize=output_size,
                                  center=center, maxRadius=max_radius, flags=flags)
        return polar_img

    def _get_contour_center(self, contour):
        """从轮廓提取中心点"""
        M = cv2.moments(contour)
        if M["m00"] > 0:
            return (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        else:
            (x, y), _ = cv2.minEnclosingCircle(contour)
            return (int(x), int(y))

    def detect(self, image_input):
        start = timer.time()
        # 读取图像
        # 判断输入类型
        if isinstance(image_input, str):  # 如果是文件路径
            image = cv2_imread_chinese(image_input)
            if image is None:
                return None, "无法读取图像", None
        elif isinstance(image_input, np.ndarray):  # 如果是 NumPy 数组（视频帧）
            image = image_input.copy()
        else:
            return None, "无效的输入类型", None

        # 获取图像中心坐标
        h, w = image.shape[:2]
        self.max_roi=min(int(min(h,w)/5),400)
        print("h=",h, " w=",w, "max_roi=",self.max_roi)
        img_center = np.array([w // 2, h // 2])

        #初始化输出
        vis_img = image.copy()
        # 划定中心区域
        mask = self._find_central_region(image)
        # visual.essay_image(vis_img, mask)
        # 形态学计算 刨除roi直边干扰
        kernel_circle = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        cleaned_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_circle)
        cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel_circle)
        contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # # 创建图像副本以绘制轮廓
        # image_with_contours = image.copy()
        # # 绘制轮廓
        # cv2.drawContours(image_with_contours, contours, -1, (0, 255, 0), thickness=2)  # 绿色轮廓，线宽为 2
        # visual.essay_image(mask, image_with_contours)

        # 轮廓处理逻辑
        if len(contours) == 0:
            # 无轮廓 考虑可能是闭眼或过于模糊
            return None, "未检出眼部信息", mask
        else:
            # 检测到轮廓
            best_contour = max(contours, key=lambda c: self._score_contour(c, img_center))
            # cv2.drawContours(image_with_contours, best_contour, -1, (0, 255, 10), thickness=10)
            # cv2.imwrite("detect.jpg", image_with_contours)
            center = self._get_contour_center(best_contour)
            # cv2.drawMarker(image_with_contours,center, (0, 20, 255), cv2.MARKER_CROSS, 50, 10)
            # cv2.imwrite("detect.jpg", image_with_contours)
            # visual.essay_image(vis_img,image_with_contours)


        # 可视化与地形图生成
        if center is not None:
            cx, cy = center
            try:
                img1 = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
                terrain_map = self.generate_terrain_map(img1, cx, cy)
            except Exception as e:
                print(f"地形图生成错误: {e}")
                terrain_map = vis_img
        else:
            terrain_map = vis_img
        print("检测时间",timer.time() - start)
        return center, terrain_map, mask