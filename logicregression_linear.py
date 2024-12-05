import pandas as pd
from sklearn.linear_model import RidgeClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 加载数据
train_path = "D:\\code\\train\\train.csv"
test_path = "D:\\code\\train\\test.csv"
ds_train = pd.read_csv(train_path)
ds_test = pd.read_csv(test_path)

# 准备数据
ds_train_data = ds_train.iloc[:, :-1]
ds_train_target = ds_train.iloc[:, -1]

# 拆分数据集
X_train, X_val, y_train, y_val = train_test_split(ds_train_data, ds_train_target, test_size=0.2, random_state=42)

# 初始化 RidgeClassifierCV 模型
ridge_clf = RidgeClassifierCV(alphas=[0.1, 1.0, 10.0], cv=5)

# 训练模型
ridge_clf.fit(X_train, y_train)

# 预测验证集
y_pred = ridge_clf.predict(X_val)

# 计算准确率
accuracy = accuracy_score(y_val, y_pred)
print(f"Validation Accuracy: {accuracy:.2f}")

# 对测试集进行预测
output_path = "D:\\code\\train\\test_predictions_ridge.csv"
test_predictions = ridge_clf.predict(ds_test)
test_results = pd.DataFrame({
    'Id': ds_test.index,
    'Label': test_predictions
})
test_results.to_csv(output_path, index=False)