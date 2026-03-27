import os
import cv2
import numpy as np
import time
from collections import defaultdict
import seaborn as sns
from matplotlib import pyplot as plt

import detect  # 导入你的detect模块


class EnhancedEvaluator:
    def __init__(self, detector):
        self.detector = detector
        self.metrics = {
            'time': {'total': 0.0, 'avg': 0.0},
            'counts': defaultdict(int),
            'breaks_details': []
        }

    def evaluate_image(self, img_path, expected_break):
        # 修复：使用detect模块中兼容中文路径的读取方法
        if isinstance(img_path, str):
            img = detect.cv2_imread_chinese(img_path)
        else:
            img = cv2.imread(img_path)

        if img is None:
            print(f"警告：无法读取图片 {img_path}")
            return None, 0.0

        start_time = time.perf_counter()
        # detect返回5个值：center, terrain_map, mask, has_break, distortion
        center, terrain_map, mask, has_break, distortion = self.detector.detect(img)
        phase1_time = time.perf_counter() - start_time

        if center is not None:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            enhanced = self.detector.adaptive_enhancement(gray)

            start_time = time.perf_counter()
            try:
                # detect_breaks_combined返回3个值：breaks, layers, distortion
                breaks, _, _ = self.detector.detect_breaks_combined(
                    center=tuple(map(int, center)),
                    raw_gray_img=gray,
                    enhanced_gray_img=enhanced
                )
            except Exception as e:
                print(f"检测breaks失败 {img_path}: {e}")
                breaks = []
            phase2_time = time.perf_counter() - start_time
        else:
            breaks, phase2_time = [], 0.0

        # 记录时间指标
        total_time = phase1_time + phase2_time
        self.metrics['time']['total'] += total_time

        # 记录breaks详情
        result = {
            'file': os.path.basename(img_path),
            'expected': expected_break,
            'detected': len(breaks) > 0,
            'time': total_time,
            'breaks_count': len(breaks),
            'phase1_time': phase1_time,
            'phase2_time': phase2_time
        }
        return result

    def batch_evaluate(self, pos_dir, neg_dir):
        # 验证目录是否存在
        if not os.path.exists(pos_dir):
            print(f"错误：阳性样本目录不存在 {pos_dir}")
            return self.metrics
        if not os.path.exists(neg_dir):
            print(f"错误：阴性样本目录不存在 {neg_dir}")
            return self.metrics

        # 阳性样本（应有breaks）
        for f in os.listdir(pos_dir):
            file_path = os.path.join(pos_dir, f)
            # 过滤非图片文件
            if not f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                continue
            res = self.evaluate_image(file_path, True)
            if res:
                self._record_result(res, 'positive')

        # 阴性样本（应无breaks）
        for f in os.listdir(neg_dir):
            file_path = os.path.join(neg_dir, f)
            # 过滤非图片文件
            if not f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                continue
            res = self.evaluate_image(file_path, False)
            if res:
                self._record_result(res, 'negative')

        # 计算综合指标
        self._calculate_metrics()
        return self.metrics

    def _record_result(self, result, sample_type):
        is_correct = result['detected'] == result['expected']
        key_prefix = 'true_' if is_correct else 'false_'
        key = f"{key_prefix}{'positive' if result['expected'] else 'negative'}"

        self.metrics['counts'][key] += 1
        self.metrics['breaks_details'].append(result)

    def _calculate_metrics(self):
        counts = self.metrics['counts']
        tp = counts.get('true_positive', 0)
        fp = counts.get('false_positive', 0)
        tn = counts.get('true_negative', 0)
        fn = counts.get('false_negative', 0)

        total = tp + tn + fp + fn
        print(f"共计算了{total}张图像")
        self.metrics['time']['avg'] = self.metrics['time']['total'] / total if total > 0 else 0

        # 标准指标命名 - 增加零值保护
        self.metrics['accuracy'] = (tp + tn) / total if total > 0 else 0
        self.metrics['precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0
        self.metrics['recall'] = tp / (tp + fn) if (tp + fn) > 0 else 0
        pre = self.metrics['precision']
        rec = self.metrics['recall']
        self.metrics['f1'] = 2 * pre * rec / (pre + rec) if (pre + rec) > 0 else 0

    def plot_confusion_matrix(self):
        counts = self.metrics['counts']
        # 构建混淆矩阵 - 确保值不为空
        cm = np.array([
            [counts.get('true_positive', 0), counts.get('false_negative', 0)],
            [counts.get('false_positive', 0), counts.get('true_negative', 0)]
        ])

        plt.figure(figsize=(6, 5))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='BuGn',
            xticklabels=['Predicted Break', 'Predicted Normal'],
            yticklabels=['Actual Break', 'Actual Normal']
        )
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('confusion_matrix_en.png', dpi=300, bbox_inches='tight')
        plt.close()

    def print_report(self):
        print("\n=== 评估指标 ===")
        print(f"Accuracy 准确率: {self.metrics['accuracy']:.2%}")
        print(f"Precision 精确率: {self.metrics['precision']:.2%}")
        print(f"Recall 召回率: {self.metrics['recall']:.2%}")
        print(f"F1-score 平衡指标: {self.metrics['f1']:.2%}")
        print(f"平均检测时间: {self.metrics['time']['avg']:.4f} 秒")

        # 打印详细统计
        counts = self.metrics['counts']
        print("\n=== 详细统计 ===")
        print(f"真阳性(TP): {counts.get('true_positive', 0)}")
        print(f"假阳性(FP): {counts.get('false_positive', 0)}")
        print(f"真阴性(TN): {counts.get('true_negative', 0)}")
        print(f"假阴性(FN): {counts.get('false_negative', 0)}")

        self.plot_confusion_matrix()
        print("\n混淆矩阵已保存为 confusion_matrix_en.png")


# 使用示例
if __name__ == "__main__":
    # 创建检测器实例
    detector = detect.PlacidoDetector()
    evaluator = EnhancedEvaluator(detector)

    # 批量评估 - 请根据实际路径修改
    metrics = evaluator.batch_evaluate(
        pos_dir="./data/2",  # 阳性样本、有裂纹
        neg_dir="./data/1"  # 阴性样本、无裂纹
    )

    # 打印评估报告
    evaluator.print_report()