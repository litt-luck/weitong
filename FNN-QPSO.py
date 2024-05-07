from tensorflow.keras.layers import Input, Dense, Dropout, Activation
import matplotlib.pyplot as plt
from tensorflow import saved_model
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import json
from copy import deepcopy
# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
# 设置正常显示负号
plt.rcParams['axes.unicode_minus'] = False


def data_handle(file_path):
    """
    读取数据并进行预处理，提取特征和目标标签
    :param file_path: 文件的绝对路径
    :return: 特征列表和目标标签列表
    """
    with open(file_path, encoding="utf-8") as f:
        header = []  # 存储数据集的特征列名
        for line in f:
            if line.startswith("@attribute"):
                header.append(line.split()[1])  # 提取每行以"@attribute"开头的特征列名
            elif line.startswith("@data"):
                break  # 遇到"@data"行后停止读取文件头部信息
        df = pd.read_csv(f, header=None)  # 使用 pandas 读取文件剩余内容作为数据集
        df.columns = header  # 将特征列名赋值给数据集的列名
        category_labels = np.array(
            [0 if label == 'N' else 1 for label in df.iloc[:, -1]])  # 将数据集最后一列的标签进行二分类转换，'N'转为0，其他值转为1
        df.insert(loc=0, column='results', value=category_labels)
        df = df.drop(columns='label')
    return df


def calFitness(particle, gBest):
    '''适应度函数，输入1个粒子的数组和全局最优适应度，返回该粒子对应的适应度'''
    nodeNum, p = particle  # 取出粒子的特征值
    net, history, valAcc, testAcc = buildNet(nodeNum, p)
    # 该粒子方案超过全局最优
    if valAcc > gBest:
        # 保存模型和对应信息
        net.save('Static/best.h5')
        history = pd.DataFrame(history)
        history.to_excel("Static/best.xlsx", index=None)
        with open("Static/info.json", "w") as f:
            f.write(json.dumps({"valAcc": valAcc, "testAcc": testAcc}))
    return valAcc


def buildNet(nodeNum, p):
    '''
    搭建全连接网络 进行训练，返回模型和训练历史、验证集准确率和测试集准确率
    :param nodeNum: 网络节点数
    :param p: dropout概率
    '''
    # 输入层 21个特征
    inputLayer = Input(shape=(21,))

    # 中间层
    middle = Dense(nodeNum)(inputLayer)
    middle = Dropout(p)(middle)

    # 输出层 二分类
    outputLayer = Dense(1, activation="sigmoid")(middle)

    # 建模 二分类损失
    model = Model(inputs=inputLayer, outputs=outputLayer)
    optimizer = Adam(learning_rate=1e-3)
    model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=['acc'])

    # 训练
    history = model.fit(xTrain, yTrain, verbose=0, batch_size=1000, epochs=100, validation_data=(xVal, yVal)).history

    # 验证集准确率
    valAcc = accuracy_score(yVal, model.predict(xVal).round(0))
    # 测试集准确率
    testAcc = accuracy_score(yTest, model.predict(xTest).round(0))
    return model, history, valAcc, testAcc


class QPSO():
    def __init__(self, featureNum, featureArea, featureLimit, featureType, particleNum=5, epochMax=10, c1=2, c2=2):
        '''
        量子粒子群算法
        :param featureNum: 粒子特征数
        :param featureArea: 特征上下限矩阵
        :param featureLimit: 特征上下阙界，也是区间的开闭 0为不包含 1为包含
        :param featureType: 特征类型 int float
        :param particleNum: 粒子个数
        :param epochMax: 最大迭代次数
        :param c1: 自身认知学习因子
        :param c2: 群体认知学习因子
        '''
        # 如上所示
        self.featureNum = featureNum
        self.featureArea = np.array(featureArea).reshape(featureNum, 2)
        self.featureLimit = np.array(featureLimit).reshape(featureNum, 2)
        self.featureType = featureType
        self.particleNum = particleNum
        self.epochMax = epochMax
        self.c1 = c1
        self.c2 = c2
        self.epoch = 0  # 已迭代次数
        # 自身最优适应度记录
        self.pBest = [-1e+10 for i in range(particleNum)]
        self.pBestArgs = [None for i in range(particleNum)]
        # 全局最优适应度记录
        self.gBest = -1e+10
        self.gBestArgs = None
        # 初始化所有粒子
        self.particles = [self.initParticle() for i in range(particleNum)]
        # 初始化所有粒子的学习速度
        self.vs = [np.random.uniform(0, 1, size=featureNum) for i in range(particleNum)]
        # 迭代历史
        self.gHistory = {"特征%d" % i: [] for i in range(featureNum)}
        self.gHistory["群内平均"] = []
        self.gHistory["全局最优"] = []

    def standardValue(self, value, lowArea, upArea, lowLimit, upLimit, valueType):
        '''
        规范一个特征值，使其落在区间内
        :param value: 特征值
        :param lowArea: 下限
        :param upArea: 上限
        :param lowLimit: 下限开闭区间
        :param upLimit: 上限开闭区间
        :param valueType: 特征类型
        :return: 修正后的值
        '''
        if value < lowArea:
            value = lowArea
        if value > upArea:
            value = upArea
        if valueType is int:
            value = np.round(value, 0)

            # 下限为闭区间
            if value <= lowArea and lowLimit == 0:
                value = lowArea + 1
            # 上限为闭区间
            if value >= upArea and upLimit == 0:
                value = upArea - 1
        elif valueType is float:
            # 下限为闭区间
            if value <= lowArea and lowLimit == 0:
                value = lowArea + 1e-10
            # 上限为闭=间
            if value >= upArea and upLimit == 0:
                value = upArea - 1e-10
        return value

    def initParticle(self):
        '''随机初始化1个粒子'''
        values = []
        # 初始化这么多特征数
        for i in range(self.featureNum):
            # 该特征的上下限
            lowArea = self.featureArea[i][0]
            upArea = self.featureArea[i][1]
            # 该特征的上下阙界
            lowLimit = self.featureLimit[i][0]
            upLimit = self.featureLimit[i][1]
            # 随机值
            value = np.random.uniform(0, 1) * (upArea - lowArea) + lowArea
            value = self.standardValue(value, lowArea, upArea, lowLimit, upLimit, self.featureType[i])
            values.append(value)
        return values

    def iterate(self, calFitness):
        '''
        开始迭代
        :param calFitness:适应度函数 输入为1个粒子的所有特征和全局最佳适应度，输出为适应度
        '''
        while self.epoch < self.epochMax:
            self.epoch += 1
            for i, particle in enumerate(self.particles):
                # 该粒子的适应度
                fitness = calFitness(particle, self.gBest)
                # 更新该粒子的自身认知最佳方案
                if self.pBest[i] < fitness:
                    self.pBest[i] = fitness
                    self.pBestArgs[i] = deepcopy(particle)
                # 更新全局最佳方案
                if self.gBest < fitness:
                    self.gBest = fitness
                    self.gBestArgs = deepcopy(particle)
            # 更新粒子
            for i, particle in enumerate(self.particles):
                # 更新速度
                self.vs[i] = np.array(self.vs[i]) + self.c1 * np.random.uniform(0, 1, size=self.featureNum) * (
                        np.array(self.pBestArgs[i]) - np.array(self.particles[i])) + self.c2 * np.random.uniform(0,
                                                                                                                 1,
                                                                                                                 size=self.featureNum) * (
                                     np.array(self.gBestArgs) - np.array(self.particles[i]))
                # 更新特征值
                self.particles[i] = np.array(particle) + self.vs[i]
                # 规范特征值
                values = []
                for j in range(self.featureNum):
                    # 该特征的上下限
                    lowArea = self.featureArea[j][0]
                    upArea = self.featureArea[j][1]
                    # 该特征的上下阙界
                    lowLimit = self.featureLimit[j][0]
                    upLimit = self.featureLimit[j][1]
                    # 随机值
                    value = self.particles[i][j]
                    value = self.standardValue(value, lowArea, upArea, lowLimit, upLimit, self.featureType[j])
                    values.append(value)
                self.particles[i] = values
            # 保存历史数据
            for i in range(self.featureNum):
                self.gHistory["特征%d" % i].append(self.gBestArgs[i])
            self.gHistory["群内平均"].append(np.mean(self.pBest))
            self.gHistory["全局最优"].append(self.gBest)
            print("QPSO epoch:%d/%d 群内平均:%.4f 全局最优:%.4f" % (
                self.epoch, self.epochMax, np.mean(self.pBest), self.gBest))


class QPSO_improved():
    def __init__(self, featureNum, featureArea, featureLimit, featureType, particleNum=5, epochMax=10, c1=2, c2=2, w=0.5, mutationRate=0.1):
        '''
        改进的量子粒子群算法
        :param featureNum: 粒子特征数
        :param featureArea: 特征上下限矩阵
        :param featureLimit: 特征上下阙界，也是区间的开闭 0为不包含 1为包含
        :param featureType: 特征类型 int float
        :param particleNum: 粒子个数
        :param epochMax: 最大迭代次数
        :param c1: 自身认知学习因子
        :param c2: 群体认知学习因子
        :param w: 惯性权重
        :param mutationRate: 变异率
        '''
        self.featureNum = featureNum
        self.featureArea = np.array(featureArea).reshape(featureNum, 2)
        self.featureLimit = np.array(featureLimit).reshape(featureNum, 2)
        self.featureType = featureType
        self.particleNum = particleNum
        self.epochMax = epochMax
        self.c1 = c1
        self.c2 = c2
        self.w = w
        self.mutationRate = mutationRate
        self.epoch = 0  # 已迭代次数
        self.pBest = [-1e+10 for _ in range(particleNum)]
        self.pBestArgs = [None for _ in range(particleNum)]
        self.gBest = -1e+10
        self.gBestArgs = None
        self.particles = [self.initParticle() for _ in range(particleNum)]
        self.vs = [np.random.uniform(0, 1, size=featureNum) for _ in range(particleNum)]
        self.gHistory = {"特征%d" % i: [] for i in range(featureNum)}
        self.gHistory["群内平均"] = []
        self.gHistory["全局最优"] = []

    def standardValue(self, value, lowArea, upArea, lowLimit, upLimit, valueType):
        if value < lowArea:
            value = lowArea
        if value > upArea:
            value = upArea
        if valueType is int:
            value = np.round(value, 0)
            if value <= lowArea and lowLimit == 0:
                value = lowArea + 1
            if value >= upArea and upLimit == 0:
                value = upArea - 1
        elif valueType is float:
            if value <= lowArea and lowLimit == 0:
                value = lowArea + 1e-10
            if value >= upArea and upLimit == 0:
                value = upArea - 1e-10
        return value

    def initParticle(self):
        values = []
        for i in range(self.featureNum):
            lowArea = self.featureArea[i][0]
            upArea = self.featureArea[i][1]
            lowLimit = self.featureLimit[i][0]
            upLimit = self.featureLimit[i][1]
            value = np.random.uniform(0, 1) * (upArea - lowArea) + lowArea
            value = self.standardValue(value, lowArea, upArea, lowLimit, upLimit, self.featureType[i])
            values.append(value)
        return values

    def iterate(self, calFitness):
        while self.epoch < self.epochMax:
            self.epoch += 1
            for i, particle in enumerate(self.particles):
                fitness = calFitness(particle, self.gBest)
                if self.pBest[i] < fitness:
                    self.pBest[i] = fitness
                    self.pBestArgs[i] = deepcopy(particle)
                if self.gBest < fitness:
                    self.gBest = fitness
                    self.gBestArgs = deepcopy(particle)

            for i, particle in enumerate(self.particles):
                # 计算惯性权重
                w = self.calculateInertiaWeight()

                # 更新速度
                self.vs[i] = w * np.array(self.vs[i]) + self.c1 * np.random.uniform(0, 1, size=self.featureNum) * (np.array(self.pBestArgs[i]) - np.array(self.particles[i])) + self.c2 * np.random.uniform(0, 1, size=self.featureNum) * (np.array(self.gBestArgs) - np.array(self.particles[i]))

                # 引入变异策略
                if np.random.uniform(0, 1) < self.mutationRate:
                    self.vs[i] = self.mutate(self.vs[i])

                # 更新特征值
                self.particles[i] = np.array(particle) + self.vs[i]
                values = []
                for j in range(self.featureNum):
                    lowArea = self.featureArea[j][0]
                    upArea = self.featureArea[j][1]
                    lowLimit = self.featureLimit[j][0]
                    upLimit = self.featureLimit[j][1]
                    value = self.particles[i][j]
                    value = self.standardValue(value, lowArea, upArea, lowLimit, upLimit, self.featureType[j])
                    values.append(value)
                self.particles[i] = values

            for i in range(self.featureNum):
                self.gHistory["特征%d" % i].append(self.gBestArgs[i])
            self.gHistory["群内平均"].append(np.mean(self.pBest))
            self.gHistory["全局最优"].append(self.gBest)
            print("QPSO epoch:%d/%d 群内平均:%.4f 全局最优:%.4f" % (
                self.epoch, self.epochMax, np.mean(self.pBest), self.gBest))

    def calculateInertiaWeight(self):
        # 自适应惯性权重策略
        w_min = 0.4
        w_max = 0.9
        return w_max - ((w_max - w_min) / self.epochMax) * self.epoch

    def mutate(self, vector):
        # 变异策略
        eta = 0.5
        beta = 0.2
        new_vector = []
        for v in vector:
            r = np.random.uniform(0, 1)
            if r < 0.5:
                delta = ((2 * r) ** (1 / (eta + 1))) - 1
            else:
                delta = 1 - ((2 * (1 - r)) ** (1 / (eta + 1)))
            new_v = v + beta * delta
            new_vector.append(new_v)
        return new_vector

data = data_handle('数据集/JM1.arff')

data = data.sample(frac=1.0)  # 打乱数据
trainData = data.iloc[:7720]  # 训练集
xTrain = trainData.values[:, 1:]
yTrain = trainData.values[:, :1]
valData = data.iloc[6000:6500]  # 验证集
xVal = valData.values[:, 1:]
yVal = valData.values[:, :1]
testData = data.iloc[6500:]  # 测试集
xTest = testData.values[:, 1:]
yTest = testData.values[:, :1]

# 为了跟优化好的模型有所对比，这里训练一个默认参数的神经网络，它的超参数取值即各超参数区间的平均值，训练并打印网络结构和训练指标
nodeArea = [10, 200]  # 节点数区间
pArea = [0, 0.5]  # dropout概率区间
# 按区间平均值训练一个神经网络
nodeNum = int(np.mean(nodeArea))
p = np.mean(pArea)
defaultNet, defaultHistory, defaultValAcc, defaultTestAcc = buildNet(nodeNum, p)
defaultNet.summary()
print("\n默认网络的 节点数:%d dropout概率:%.2f 验证集准确率:%.4f 测试集准确率:%.4f" % (
    nodeNum, p, defaultValAcc, defaultTestAcc))

# 实例化PSO模型，将区间信息输入，开始迭代，适应度函数就是输入1各粒子和全局最优适应度，返回该粒子对应方案的验证集准确率
featureNum = 2  # 2个需要优化的特征
featureArea = [nodeArea, pArea]  # 2个特征取值范围
featureLimit = [[1, 1], [0, 1]]  # 取值范围的开闭  0为闭区间 1为开区间
featureType = [int, float]  # 2个特征的类型
# 量子粒子群算法类
qpso = QPSO(featureNum, featureArea, featureLimit, featureType)

# 开始用量子粒子群算法迅游
qpso.iterate(calFitness)
# 载入最佳模型和对应的训练历史
bestNet = load_model("Static/best.h5")
with open("Static/info.json", "rb") as f:
    info = json.loads(f.read())
bestValAcc = float(info["valAcc"])
bestTestAcc = float(info["testAcc"])
bestHistory = pd.read_excel("Static/best.xlsx")
print("最优模型的验证集准确率:%.4f 测试集准确率:%.4f" % (bestValAcc, bestTestAcc))

# 查看QPSO最优解随迭代次数的变换，并绘制图形
history = pd.DataFrame(qpso.gHistory)
history["epoch"] = range(1, history.shape[0] + 1)
print(history)
history.to_excel("Static/History.xlsx")


# 对比下默认参数模型和PSO调优模型的准确率
fig, ax = plt.subplots()
x = np.arange(2)
a = [defaultValAcc, bestValAcc]
b = [defaultTestAcc, bestTestAcc]
total_width, n = 0.8, 2
width = total_width / n
x = x - (total_width - width) / 2
ax.bar(x, a, width=width, label='验证集', color="b")
for x1, y1 in zip(x, a):
    plt.text(x1, y1 + 0.01, '%.3f' % y1, ha='center', va='bottom')
ax.bar(x + width, b, width=width, label='测试集', color="green")
for x1, y1 in zip(x, b):
    plt.text(x1 + width, y1 + 0.01, '%.3f' % y1, ha='center', va='bottom')
ax.legend()
ax.set_xticks([0, 1])
ax.set_ylim([0, 1.2])
ax.set_ylabel("准确度")
ax.set_xticklabels(["默认网络", "改进QPSO优化后的网络"])
plt.grid(True)
fig.savefig("../图片/对比.png", dpi=250)
