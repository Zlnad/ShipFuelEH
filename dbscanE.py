import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn as s1
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
import warnings

warnings.filterwarnings('ignore')

# 读取数据
df = pd.read_csv('data/mingxi_0618_0715.csv')

print("数据基本信息:")
print(f"数据形状: {df.shape}")
print(f"时间范围: {df['PCTime'].min()} 到 {df['PCTime'].max()}")

# 选择用于聚类的特征（排除时间列）
features = ['MESFOC_nmile', 'MERpm', 'METorque', 'MEShaftPow', 'ShipSpdToWater',
            'ShipHeel', 'ShipTrim', 'ShipSlip', 'WindSpd', 'WindDir',
            'ShipDraughtBow', 'ShipDraughtAstern', 'ShipDraughtMidLft', 'ShipDraughtMidRgt']

eps = 0.8
min_samples = 150


# 数据预处理
X = df[features].copy()

# 检查并处理缺失值
print(f"\n缺失值统计:")
print(X.isnull().sum())

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 使用DBSCAN进行异常检测
print("\n使用DBSCAN进行异常检测...")

# 尝试不同的参数组合
best_dbscan = None
best_score = -1

# for eps in [0.5, 1.0, 1.5, 2.0]:
#     for min_samples in [5, 10, 15, 20]:
dbscan = DBSCAN(eps=eps, min_samples=min_samples)
labels = dbscan.fit_predict(X_scaled)

# 计算正常点的比例（非异常点比例越高越好）
normal_ratio = np.sum(labels != -1) / len(labels)
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

# 选择评分标准：正常点比例高且有一定数量的簇
score = normal_ratio * (1 + 0.1 * min(n_clusters, 5))
if score > best_score and n_clusters > 0:
            best_score = score
            best_dbscan = dbscan
            print(
                f"eps={eps}, min_samples={min_samples}: 正常点比例={normal_ratio:.3f}, 簇数量={n_clusters}, 得分={score:.3f}")

# 使用最佳参数
dbscan = best_dbscan
labels = dbscan.fit_predict(X_scaled)

# 分析结果
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = list(labels).count(-1)

print(f"\nDBSCAN结果:")
print(f"簇的数量: {n_clusters}")
print(f"异常点数量: {n_noise}")
print(f"正常点数量: {len(labels) - n_noise}")
print(f"异常点比例: {n_noise / len(labels):.3f}")

# 将结果添加到原始数据中
df['DBSCAN_Label'] = labels
df['Is_Anomaly'] = (labels == -1)

# 分析异常点的特征
anomaly_points = df[df['Is_Anomaly'] == True]
normal_points = df[df['Is_Anomaly'] == False]

print(f"\n异常点统计:")
print(f"异常点时间范围: {anomaly_points['PCTime'].min()} 到 {anomaly_points['PCTime'].max()}")

# 可视化结果
plt.figure(figsize=(15, 12))

# 1. 主要参数的时间序列图，标记异常点
plt.subplot(3, 2, 1)
plt.plot(df.index, df['MERpm'], 'b-', alpha=0.7, label='正常点')
plt.scatter(anomaly_points.index, anomaly_points['MERpm'], color='red', s=20, label='异常点')
plt.xlabel('时间索引')
plt.ylabel('主机转速 (RPM)')
plt.title('主机转速 - 异常点检测')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(3, 2, 2)
plt.plot(df.index, df['METorque'], 'g-', alpha=0.7, label='正常点')
plt.scatter(anomaly_points.index, anomaly_points['METorque'], color='red', s=20, label='异常点')
plt.xlabel('时间索引')
plt.ylabel('主机扭矩')
plt.title('主机扭矩 - 异常点检测')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(3, 2, 3)
plt.plot(df.index, df['MEShaftPow'], 'purple', alpha=0.7, label='正常点')
plt.scatter(anomaly_points.index, anomaly_points['MEShaftPow'], color='red', s=20, label='异常点')
plt.xlabel('时间索引')
plt.ylabel('轴功率')
plt.title('轴功率 - 异常点检测')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(3, 2, 4)
plt.plot(df.index, df['ShipSpdToWater'], 'orange', alpha=0.7, label='正常点')
plt.scatter(anomaly_points.index, anomaly_points['ShipSpdToWater'], color='red', s=20, label='异常点')
plt.xlabel('时间索引')
plt.ylabel('船速')
plt.title('船速 - 异常点检测')
plt.legend()
plt.grid(True, alpha=0.3)

# 5. 使用t-SNE降维可视化聚类结果
plt.subplot(3, 2, 5)
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X_tsne = tsne.fit_transform(X_scaled)

for label in set(labels):
    if label == -1:
        plt.scatter(X_tsne[labels == label, 0], X_tsne[labels == label, 1],
                    c='red', label='异常点', s=20, alpha=0.6)
    else:
        plt.scatter(X_tsne[labels == label, 0], X_tsne[labels == label, 1],
                    label=f'簇 {label}', s=20, alpha=0.6)

plt.xlabel('t-SNE 特征 1')
plt.ylabel('t-SNE 特征 2')
plt.title('DBSCAN聚类结果 (t-SNE可视化)')
plt.legend()
plt.grid(True, alpha=0.3)

# 6. 异常点分布直方图
plt.subplot(3, 2, 6)
cluster_counts = [np.sum(labels == i) for i in set(labels) if i != -1]
anomaly_count = np.sum(labels == -1)

plt.bar(['异常点'] + [f'簇{i}' for i in range(len(cluster_counts))],
        [anomaly_count] + cluster_counts,
        color=['red'] + [f'C{i}' for i in range(len(cluster_counts))])
plt.xlabel('聚类类别')
plt.ylabel('数据点数量')
plt.title('DBSCAN聚类分布')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# 详细分析异常点
print("\n异常点详细分析:")
print("=" * 50)

# 异常点的主要特征统计
print("\n异常点特征统计:")
anomaly_stats = anomaly_points[features].describe()
print(anomaly_stats)

print("\n正常点特征统计:")
normal_stats = normal_points[features].describe()
print(normal_stats)

# 找出异常点的时间模式
anomaly_time_patterns = anomaly_points['PCTime'].str.split(' ').str[1].str.split(':').str[0].value_counts().sort_index()
print(f"\n异常点按小时分布:")
print(anomaly_time_patterns)

# 保存结果
df.to_csv('mingxi_0618_0715_with_anomaly.csv', index=False)
print(f"\n结果已保存到: mingxi_0618_0715_with_anomaly.csv")

# 总结报告
print("\n" + "=" * 60)
print("DBSCAN异常检测总结报告")
print("=" * 60)
print(f"总数据点: {len(df)}")
print(f"异常点数量: {n_noise} ({n_noise / len(df) * 100:.2f}%)")
print(f"正常簇数量: {n_clusters}")
print(f"主要异常特征:")
for feature in features[:6]:  # 只看前6个重要特征
    anomaly_mean = anomaly_points[feature].mean()
    normal_mean = normal_points[feature].mean()
    diff_pct = abs(anomaly_mean - normal_mean) / normal_mean * 100
    if diff_pct > 10:  # 差异超过10%的特征
        print(f"  - {feature}: 正常值={normal_mean:.2f}, 异常值={anomaly_mean:.2f}, 差异={diff_pct:.1f}%")