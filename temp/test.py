import os
import cv2
import time
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

# 定位破损区域
def locate_damage_regions(gray_img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray_img)
    edges = cv2.Canny(enhanced, 50, 150)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) > 100]
    return boxes

# 特征提取：加入破损框数量
def extract_features(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"警告：无法读取图像 {image_path}")
        return [0, 0, 0, 0]
    img = cv2.resize(img, (256, 256))
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    edge_density = np.sum(edges > 0) / (256 * 256)
    mean_val = np.mean(blur)
    _, binary = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY)
    bw_ratio = np.sum(binary == 0) / (np.sum(binary == 255) + 1e-5)

    # 新增：破损区域数量
    damage_boxes = locate_damage_regions(img)
    damage_count = len(damage_boxes)
    return [mean_val, bw_ratio, edge_density, damage_count]

# 加载图像路径
def load_image_paths(normal_dir, broken_dir):
    normal_images = [os.path.join(normal_dir, f) for f in os.listdir(normal_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    broken_images = [os.path.join(broken_dir, f) for f in os.listdir(broken_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    all_paths = normal_images + broken_images
    labels = [0] * len(normal_images) + [1] * len(broken_images)
    return all_paths, labels

# 评估函数
def evaluate_model(model, features, labels):
    prediction_times = []
    preds = []
    for feat in features:
        start_pred = time.time()
        pred = model.predict([feat])[0]
        end_pred = time.time()
        preds.append(pred)
        prediction_times.append((end_pred - start_pred))
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)
    avg_time = np.mean(prediction_times)
    return acc, f1, avg_time, len(labels)

# 路径设置
train_normal_dir = '../data/train/1'
train_broken_dir = '../data/train/2'
test_normal_dir = '../data/1'
test_broken_dir = '../data/2'

# 加载训练集
train_paths, train_labels = load_image_paths(train_normal_dir, train_broken_dir)
print(f"训练集：正常 {train_labels.count(0)} 张，破损 {train_labels.count(1)} 张")
train_features = [extract_features(p) for p in train_paths]
train_labels = np.array(train_labels)

# 加载测试集
test_paths, test_labels = load_image_paths(test_normal_dir, test_broken_dir)
print(f"测试集：正常 {test_labels.count(0)} 张，破损 {test_labels.count(1)} 张")
test_features = [extract_features(p) for p in test_paths]
test_labels = np.array(test_labels)

# 特征标准化
scaler = StandardScaler()
train_features_scaled = scaler.fit_transform(train_features)
test_features_scaled = scaler.transform(test_features)

# SVM 训练
svm_model = SVC(kernel='rbf', C=10, gamma=0.01)
start_train = time.time()
svm_model.fit(train_features_scaled, train_labels)
end_train = time.time()

# RF 训练
rf_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
start_train_rf = time.time()
rf_model.fit(train_features_scaled, train_labels)
end_train_rf = time.time()

# 测试评估
svm_acc, svm_f1, svm_time, total = evaluate_model(svm_model, test_features_scaled, test_labels)
rf_acc, rf_f1, rf_time, _ = evaluate_model(rf_model, test_features_scaled, test_labels)

# 结果展示
print({
    "模型": "SVM (RBF)",
    "测试样本数": total,
    "准确率": round(svm_acc, 4),
    "F1 分数": round(svm_f1, 4),
    "平均预测耗时 (s)": round(svm_time, 4),
    "训练时间 (s)": round(end_train - start_train, 2)
})
print({
    "模型": "随机森林",
    "测试样本数": total,
    "准确率": round(rf_acc, 4),
    "F1 分数": round(rf_f1, 4),
    "平均预测耗时 (s)": round(rf_time, 4),
    "训练时间 (s)": round(end_train_rf - start_train_rf, 2)
})
