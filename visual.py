from matplotlib import pyplot as plt
import cv2
import numpy as np
import pandas as pd
import seaborn as sns

def debug_breaks_354(raw_grey_roi, center, layers, breaks):
    cx, cy = center
    canvas = cv2.cvtColor(raw_grey_roi, cv2.COLOR_GRAY2BGR)
    overlay = canvas.copy()

    table_data = []
    for num, (layer_idx, segments) in enumerate(breaks, 1):
        if layer_idx - 1 < 0:
            continue  # 避免越界
        r1 = layers[layer_idx - 1][1]
        r2 = layers[layer_idx][1]

        for start_angle, end_angle in segments:
            table_data.append([num, layer_idx, start_angle, end_angle])

            angles = np.deg2rad(np.arange(start_angle, end_angle + 1))
            pts_outer = np.column_stack((cx + r2 * np.cos(angles), cy + r2 * np.sin(angles)))
            pts_inner = np.column_stack((cx + r1 * np.cos(angles[::-1]), cy + r1 * np.sin(angles[::-1])))
            polygon = np.vstack((pts_outer, pts_inner)).astype(np.int32)

            cv2.polylines(overlay, [polygon], isClosed=True, color=(0, 0, 255), thickness=5)

            # Draw border lines
            for angle_deg in [start_angle, end_angle]:
                angle_rad = np.deg2rad(angle_deg)
                pt1 = (int(cx + r1 * np.cos(angle_rad)), int(cy + r1 * np.sin(angle_rad)))
                pt2 = (int(cx + r2 * np.cos(angle_rad)), int(cy + r2 * np.sin(angle_rad)))
                cv2.line(overlay, pt1, pt2, (0, 255, 255), thickness=5)

    # 图像叠加并裁剪中心区域
    result_img = cv2.addWeighted(canvas, 0.6, overlay, 0.4, 0)
    h, w = result_img.shape[:2]
    left = max(cx - 450, 0)
    top = max(cy - 450, 0)
    cropped = result_img[top:top + 900, left:left + 900]

    # 创建图形，使用gridspec控制宽度比例
    fig = plt.figure(figsize=(15, 8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 0.6])  # 图像:表格 = 1:0.6

    # 图像子图
    ax_img = fig.add_subplot(gs[0])
    ax_img.imshow(cropped[..., ::-1])
    ax_img.axis('off')
    ax_img.set_title("Detected Breaks", pad=20, fontsize=20)  # 增加标题间距

    # 表格子图设置
    ax_table = fig.add_subplot(gs[1])
    ax_table.axis('off')

    # 表格数据展示
    col_labels = ["num", "layer", "start", "end"]
    if not table_data:
        table_data = [["No breaks", "-", "-", "-"]]

    # 创建表格（关键修改：添加bbox参数控制表格位置）
    table = ax_table.table(
        cellText=table_data,
        colLabels=col_labels,
        loc='center',
        cellLoc='center',
        colLoc='center',
        colWidths=[0.1, 0.1, 0.1, 0.1],
        bbox=[0, 0.3, 1, 0.6]  # [x0, y0, width, height] 控制表格在子图中的位置
    )

    # 计算标题位置（精确到像素）
    dpi = fig.dpi
    title_offset_px = 20  # 10像素间距
    # 将像素转换为相对坐标（考虑表格高度）
    title_offset_rel = title_offset_px / (fig.get_size_inches()[1] * dpi * 0.6)  # 0.6是表格高度占比

    # 添加标题（精确控制位置）
    table_title = ax_table.text(
        0.5,  # 水平居中
        0.3 + 0.6 + title_offset_rel,  # 表格顶部y位置 + 表格高度 + 偏移
        "Breaks Table",
        ha='center',
        va='bottom',
        fontsize=20,
        transform=ax_table.transAxes
    )

    # 表格样式调整
    table.auto_set_font_size(False)
    table.set_fontsize(18)
    table.scale(1.2, 1.5)  # 减少垂直缩放

    # 单元格样式
    for key, cell in table.get_celld().items():
        # cell.set_height(0.1)  # 增加行高使单行表格更明显
        if key[0] == 0:  # 标题行
            cell.set_facecolor('#f0f0f0')
            cell.set_text_props(weight='bold')
    plt.tight_layout()
    plt.show()


def debug_breaks_352(roi_img, center, every_width):
    cx, cy = center
    half = 450
    angle_start = 315
    angle_end = 330

    # === 图一：裁剪区域 + 扇形标记 ===
    crop_img = roi_img[cy - half:cy + half, cx - half:cx + half].copy()
    fig1, ax1 = plt.subplots(figsize=(6, 6))
    ax1.imshow(crop_img, cmap='gray')
    ax1.axis('off')
    plt.title("315°-330° in ROI", fontsize=14)

    # 绘制扇形边界（两条射线 + 圆弧连接）
    for angle in [angle_start, angle_end]:
        rad = np.deg2rad(angle)
        x = int(half + half * np.cos(rad))
        y = int(half + half * np.sin(rad))
        ax1.plot([half, x], [half, y], color='lime', linewidth=1)

    # 画圆弧连接两条边
    arc_angles = np.linspace(angle_start, angle_end, 100)
    arc_x = half + half * np.cos(np.deg2rad(arc_angles))
    arc_y = half + half * np.sin(np.deg2rad(arc_angles))
    ax1.plot(arc_x, arc_y, color='lime', linewidth=1)

    # 2. 构建300~315度环层信息表格
    angle_range = range(315, 330)
    max_layers = 7
    columns = ["Angle"] + [f"Layer {i + 1}" for i in range(max_layers)]
    data = [[f"{angle}°"] + [f"{r1}-{r2}" for r1, r2 in every_width.get(angle, [])[:max_layers]] +
            ['—'] * (max_layers - len(every_width.get(angle, []))) for angle in angle_range]
    df = pd.DataFrame(data, columns=columns)

    # 创建仅用于热图显示的数值 DataFrame
    placeholder_data = np.ones((len(df), len(columns)))

    # 画 seaborn heatmap 表格
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        placeholder_data,
        annot=df.values,
        fmt='',
        cbar=False,
        linewidths=0.5,
        linecolor='#cccccc',
        cmap=["#ffffff"],
        annot_kws={"size": 12},
        xticklabels=columns,
        yticklabels=False,
        ax=ax
    )

    # 设置轴样式
    ax.set_xticklabels(columns, rotation=45, ha='right')
    ax.set_yticks([])  # 删除多余的y轴刻度

    # 强制画最右边和底边框线
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.5)
        spine.set_edgecolor('#cccccc')

    plt.title("Ring Layer Info (315°–330°)", fontsize=14)
    plt.tight_layout()
    plt.show()

def debug_polar_layers(roi_img, center, radius_pairs):
    x_c, y_c = center
    half_size = 450
    color = (0, 255, 0)

    # 1. 确保图像有3个通道
    if len(roi_img.shape) == 2:
        roi_img = cv2.cvtColor(roi_img, cv2.COLOR_GRAY2BGR)

    # 2. 图像边界裁剪处理
    h, w = roi_img.shape[:2]
    x1, y1 = max(x_c - half_size, 0), max(y_c - half_size, 0)
    x2, y2 = min(x_c + half_size, w), min(y_c + half_size, h)
    cropped = roi_img[y1:y2, x1:x2].copy()

    # 3. 绘制圆
    offset_center = (x_c - x1, y_c - y1)  # 防止超出边界情况
    for (r_min, r_max) in radius_pairs:
        cv2.circle(cropped, offset_center, r_min, color, 1)
        cv2.circle(cropped, offset_center, r_max, color, 1)

    # 4. 显示图像
    plt.figure(figsize=(6, 6))
    plt.imshow(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
    plt.title("Detected Ring Layers (1000×1000 Crop)")
    plt.axis("off")
    plt.show()


def debug_visualize_horizontal(img, derivative, layers):
    plt.figure(figsize=(15, 6))  # 调整画布大小

    # 1. 导数信号分析 + 标记关键点坐标（不画垂直线）
    plt.subplot(121)
    plt.plot(derivative)
    for start, end in layers:
        # 标记 start 点（红色点 + 坐标文字）
        plt.scatter(start, derivative[start], color='red', s=30, zorder=5)
        plt.text(start, derivative[start], f'x={start}',
                 color='black', fontsize=12, ha='left', va='bottom')

        # 标记 end 点（绿色点 + 坐标文字）
        plt.scatter(end, derivative[end], color='green', s=30, zorder=5)
        plt.text(end, derivative[end], f'x={end}',
                 color='black', fontsize=12, ha='left', va='bottom')
    plt.title("Derivative with Key Points")

    # 2. 原始图像 + 垂直线标注（保持原样）
    plt.subplot(122)
    plt.imshow(img, cmap='gray')
    for start, end in layers:
        plt.axvline(x=start, color='r', linestyle='--')
        plt.axvline(x=end, color='g', linestyle='--')
    plt.title("Annotated Image")

    plt.tight_layout()
    plt.show()




def essay_image( original_img, binary_img):
    # 创建画布
    fig = plt.figure(figsize=(16, 7), dpi=150)  # 总画布尺寸（英寸）

    # 转换原始图像颜色空间
    original_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    binary_img = cv2.cvtColor(binary_img, cv2.COLOR_BGR2RGB)

    # # 调整图像显示尺寸
    # def resize_to_display(img, target_size=(800, 600)):
    #     h, w = img.shape[:2]
    #     scale = min(target_size[0] / w, target_size[1] / h)
    #     return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_NEAREST)
    #
    # # 调整图像尺寸
    # resized_original = resize_to_display(original_rgb)
    # resized_binary = resize_to_display(binary_img)

    # 创建子图
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    # 显示原始图像
    ax1.imshow(original_rgb)
    ax1.set_title('Original Image\nSize: {}x{}'.format(*original_img.shape[:2]))
    ax1.axis('off')

    # 显示处理后图像
    ax2.imshow(binary_img, cmap='gray')
    ax2.set_title('Visualized Image \nSize: {}x{}'.format(*binary_img.shape[:2]))
    ax2.axis('off')

    # 调整布局
    plt.tight_layout()
    plt.show()

def debug_layers_der(polar,derivative,horizontal_proj,proj_smoothed):
    # 可视化部分
    plt.figure(figsize=(12, 8))

    # 1. 极坐标变换图像
    plt.subplot(2, 2, 1)
    plt.imshow(polar, cmap='gray')
    plt.title('Polar Transformed Image')
    plt.axis('off')

    # 2. 原始水平投影
    plt.subplot(2, 2, 2)
    plt.plot(horizontal_proj, 'b-', label='Original')
    plt.title('Original Horizontal Projection')
    plt.xlabel('Column Index')
    plt.ylabel('Intensity')

    # 3. 平滑后曲线
    plt.subplot(2, 2, 3)
    plt.plot(proj_smoothed, 'r-', label='Smoothed')
    plt.title('Smoothed Projection')
    plt.xlabel('Column Index')
    plt.ylabel('Intensity')

    # 4. 导数曲线（仅保留y=0参考线）
    plt.subplot(2, 2, 4)
    plt.plot(derivative, 'g-', label='Derivative')
    plt.axhline(y=0, color='k', linestyle='--', linewidth=0.8)  # 添加y=0参考线
    plt.title('Projection Derivative')
    plt.xlabel('Column Index')
    plt.ylabel('Derivative Value')

    plt.tight_layout()
    plt.show()


def visualize_analysis(roi, center, breaks, layers):
    vis = cv2.addWeighted(roi, 0.75, np.zeros_like(roi), 0.25, 0)
    for r_min, r_max in layers:
        cv2.circle(vis, center, r_max, (0, 255, 0), 1)  # 外圆绿色细线
    for layer_idx, angle_ranges in breaks:
        for start, end in angle_ranges:
            if layer_idx > 0:
                prev_r_max = layers[layer_idx - 1][1]
                curr_r_max = layers[layer_idx][1]
                # 起点终点
                pt1_inner = (int(center[0] + prev_r_max * np.cos(np.deg2rad(start))),
                             int(center[1] + prev_r_max * np.sin(np.deg2rad(start))))
                pt2_inner = (int(center[0] + prev_r_max * np.cos(np.deg2rad(end))),
                             int(center[1] + prev_r_max * np.sin(np.deg2rad(end))))
                pt1_outer = (int(center[0] + curr_r_max * np.cos(np.deg2rad(start))),
                             int(center[1] + curr_r_max * np.sin(np.deg2rad(start))))
                pt2_outer = (int(center[0] + curr_r_max * np.cos(np.deg2rad(end))),
                             int(center[1] + curr_r_max * np.sin(np.deg2rad(end))))
                cv2.line(vis, pt1_inner, pt1_outer, (0, 165, 255), 1, lineType=cv2.LINE_AA)
                cv2.line(vis, pt2_inner, pt2_outer, (255, 165, 255), 1, lineType=cv2.LINE_AA)
                overlay = vis.copy() # 构建并绘制填充区域
                num_arc_points = 50
                arc_outer = [
                    (int(center[0] + curr_r_max * np.cos(np.deg2rad(a))),
                     int(center[1] + curr_r_max * np.sin(np.deg2rad(a))))
                    for a in np.linspace(start, end, num_arc_points)
                ] # 外圆弧
                arc_inner = [
                    (int(center[0] + prev_r_max * np.cos(np.deg2rad(a))),
                     int(center[1] + prev_r_max * np.sin(np.deg2rad(a))))
                    for a in np.linspace(end, start, num_arc_points)
                ] # 内圆弧
                polygon = [pt1_outer] + arc_outer + [pt2_outer, pt2_inner] + arc_inner + [pt1_inner] # 封闭区域点构造
                pts = np.array(polygon, dtype=np.int32)
                cv2.fillPoly(overlay, [pts], (255, 255, 200)) # 填充
                cv2.addWeighted(overlay, 0.8, vis, 0.2, 0, vis)

    cv2.drawMarker(vis, center, (0, 0, 255), cv2.MARKER_CROSS, 20, 2)  # 标记中心点
    return vis
