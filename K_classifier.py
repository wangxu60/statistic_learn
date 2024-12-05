import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# 加载数据
train_path = "D:\\code\\train\\train.csv"
test_path = "D:\\code\\train\\test.csv"
ds_train = pd.read_csv(train_path)
ds_test = pd.read_csv(test_path)

# 准备数据
ds_train_data = ds_train.iloc[:, :-1]
ds_train_target = ds_train.iloc[:, -1]


# 标准化特征
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(ds_train_data)
X_test_scaled = scaler.transform(ds_test)

# 拆分数据集
X_train, X_val, y_train, y_val = train_test_split(X_train_scaled, ds_train_target, test_size=0.2, random_state=42)

# 创建 KNN 模型
knn = KNeighborsClassifier(n_neighbors=10)  # 选择邻居数为 3

# 训练模型
knn.fit(X_train, y_train)

# 评估模型
y_pred_val = knn.predict(X_val)
val_accuracy = accuracy_score(y_val, y_pred_val)
print(f"Validation Accuracy: {val_accuracy:.2f}")

# 测试模型
y_pred_test = knn.predict(X_test_scaled)
output_path = "D:\\code\\train\\test_predictions_knn.csv"
test_results = pd.DataFrame({
    'Id': ds_test.index,
    'Label': y_pred_test
})
test_results.to_csv(output_path, index=False)