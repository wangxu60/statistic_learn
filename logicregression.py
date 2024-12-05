import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
train_path="D:\\code\\train\\train.csv"
test_path="D:\\code\\train\\test.csv"
ds_train=pd.read_csv(train_path)
ds_test=pd.read_csv(test_path)
# print(ds_train)
ds_train_data=ds_train.iloc[:,:-1]
ds_train_target=ds_train.iloc[:,-1]
# 逻辑回归代码
# print(ds_train_target)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 假设 ds_train_data 是特征，ds_train_target 是标签
X_train, X_val, y_train, y_val = train_test_split(ds_train_data, ds_train_target, test_size=0.2, random_state=42)

# 初始化逻辑回归模型
model = LogisticRegression(C=0.5)

# 训练模型
model.fit(X_train, y_train)

# 预测验证集
y_pred = model.predict(X_val)
# print(y_pred)
# print(X_val)
# 计算准确率
accuracy = accuracy_score(y_val, y_pred)
print(f"Validation Accuracy: {accuracy:.2f}")
output_path = "D:\\code\\train\\test_predictions.csv"
test_predictions = model.predict(ds_test)
test_results = pd.DataFrame({
    'Id': ds_test.index,
    'Label': test_predictions
})
test_results.to_csv(output_path, index=False)

# 如果有测试集，可以对测试集进行预测
# ds_test_data = ds_test.iloc[:, :-1]
# ds_test_target = ds_test.iloc[:, -1]

# test_predictions = model.predict(ds_test_data)
# test_accuracy = accuracy_score(ds_test_target, test_predictions)
# print(f"Test Accuracy: {test_accuracy:.2f}")