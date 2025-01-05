import pandas as pd
from collections import Counter
from imblearn.over_sampling import RandomOverSampler

# 假设df是你的DataFrame，并且'label'是你想要过采样的列
df = pd.read_csv('./datasets.csv')  # 如果你需要从CSV读取数据

# 检查每个label的数量
print("Original dataset shape %s" % Counter(df['label']))

# 初始化RandomOverSampler
ros = RandomOverSampler(sampling_strategy='all', random_state=42)

# 只对样本数少于1500的类别进行过采样
def custom_resample(X, y):
    counts = y.value_counts()
    # 获取需要过采样的标签
    under_1500 = counts[counts < 1500].index.tolist()
    # 过滤出需要过采样的数据
    X_under, y_under = X[y.isin(under_1500)], y[y.isin(under_1500)]
    # 对这些数据进行过采样
    X_res, y_res = ros.fit_resample(X_under, y_under)
    
    # 合并过采样的数据和原本不少于1500的类别数据
    X_rest, y_rest = X[~y.isin(under_1500)], y[~y.isin(under_1500)]
    X_resampled = pd.concat([X_res, X_rest])
    y_resampled = pd.concat([y_res, y_rest])
    
    return X_resampled, y_resampled

# 分离特征和标签
X = df.drop('label', axis=1)
y = df['label']

# 执行自定义过采样
X_res, y_res = custom_resample(X, y)

# 创建一个新的DataFrame
df_resampled = pd.concat([X_res, y_res], axis=1)

# 检查过采样后的数据集形状
print("Resampled dataset shape %s" % Counter(y_res))

# 保存到新的CSV文件
df_resampled.to_csv('resampled_output.csv', index=False)