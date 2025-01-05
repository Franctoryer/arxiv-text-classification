import matplotlib.pyplot as plt

# 数据
categories = ['cs.AI', 'cs.CE', 'cs.CV', 'cs.DS', 'cs.IT', 'cs.NE', 'cs.PL', 'cs.SY', 'math.AC', 'math.GR', 'math.ST']
frequencies = [1500, 1343, 1487, 1500, 1399, 1500, 1476, 1119, 607, 1002, 1500]

# 创建颜色列表（这里我们使用默认的颜色循环）
colors = plt.cm.viridis(range(len(categories)))

# 创建横向直方图
plt.figure(figsize=(5, 5))
bars = plt.barh(categories, frequencies, color='#6e9bc5')

# 添加标题和轴标签
plt.xlabel('Frequency')
plt.ylabel('Categories')

# 调整布局以适应标签
plt.tight_layout()

plt.savefig('distribution.pdf')
# 显示图表
plt.show()