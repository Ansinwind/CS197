# Coding

## Basic

### Data Processing

```python
import torch

x = torch.arange(12)
x.shape
x.numel()

X = x.reshape(3,4)

torch.zeros((2,3,4))
# 关于维度：从后往前：列、行、三维、以此类推
torch.ones((2,3,4))

torch.tensor([[2,1,4,3],[1,2,3,4]])
# 几维就有几层中括号

torch.cat((X,Y),dim=0)
# 第0维保持列不变，行数增加；第1维保持行不变，列数增加

# 广播机制 行列自动扩展

# 可以用[-1]选择最后一个元素，[a:b]选择a到b-1几个元素

X[0:2,:]=12
# 赋值，这里选到的是0、1两行，所有列

A = X.numpy()
B = torch.tensor(A)
# 转化为NumPy张量
```

### Data Preprocessing

```python
import os

os.makedirs(os.path.join('..','data'),exist_ok=True)
datafile = os.path.join('..','data','xxx.csv')
with open(data_file, 'w') as f:
    f.write('xxxxxxx,xxxx,xx\n') #列名
# 这是写文件

import pandas as pd
data = pd.read_csv(data_file)
print(data)

# 预处理

# 常用删除

inputs = inputs.fillna(inputs.mean())
# 插值

inputs = pd.get_dummies(inputs.dummy_na=True)
# 处理离散缺值

inport torch
X, y = torch.tensor(inputs.values), torch.tensor(outputs.values)
```

### Q & A

**reshape 和 view** : reshape之类很多函数不改地址，改着改着老东西也变了

tensor 数学 张量 array 计算机 数组

### Auto grad

```python
x.requires_grad_(True)

x = torch.arange(4.0, requires_grad=True)

y = 2 * torch.dot(x,x)

# y = tensor(28., grad_fn=<MulBackward0>)

y.backward()
x.grad
# 反向y之后我们得到y关于x的导

# 默认情况下，PyTorch累积梯度，需要清除之前的值
x.grad.zero_()
y = x.sum()
y.backward()
x.grad

# 对非标量调用backward需要传入gradient参数，该参数指定微分函数
x.grad.zero_()
y = x * x
gradient = torch.ones_like(y) # 创建与 y 同形状的全1张量
y.backward(gradient)
# 等价于y.backward(torch.ones(len(x)))
y.sum().backward()
x.grad

# 将某些计算移动到计算图外（这里u不参与求导）
x.grad.zero_()
y = x * x
u = y.detach()
z = u * x

z.sum().backward()
x.grad == u
```



## Linear Regression

### 基本实现

#### 生成数据集

```python
def synthetic_data(w, b, num_examples):
    """生成y=Xw+b+噪声"""
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))

true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)
```

通过这种方式得到一个简单的数据集

#### 读取数据集

```python
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    # 这些样本是随机读取的，没有特定的顺序 indices是包含样本索引的列表
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(
            indices[i: min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]
        
batch_size = 10

for X, y in data_iter(batch_size, features, labels):
    print(X, '\n', y)
    break
```

1. **打乱索引**：`random.shuffle(indices)` 实现随机采样
2. **分批遍历**：`range(0, num_examples, batch_size)` 控制每批起始位置
3. **防止越界**：`min(i + batch_size, num_examples)` 处理最后一个不足 batch 的情况
4. **生成器模式**：使用 `yield` 按需返回 batch，节省内存

当我们运行迭代时，我们会连续地获得不同的小批量，直至遍历完整个数据集。上面实现的迭代执行效率很低，可能会在实际问题上陷入麻烦。

在深度学习框架中实现的内置迭代器效率要高得多，可以处理存储在文件中的数据和数据流提供的数据。

#### 初始化模型参数

```python
w = torch.normal(0, 0.01, size=(2,1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)
```

#### 定义模型

```python
def linreg(X, w, b):  #@save
    """线性回归模型"""
    return torch.matmul(X, w) + b
```

#### 定义损失函数

```python
def squared_loss(y_hat, y):  #@save
    """均方损失"""
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2
```

#### 定义优化算法

```python
def sgd(params, lr, batch_size):  #@save
    """小批量随机梯度下降"""
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()
```

#### 训练

```python
lr = 0.03
num_epochs = 3
net = linreg
loss = squared_loss
for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y)  # X和y的小批量损失
        # 因为l形状是(batch_size,1)，而不是一个标量。l中的所有元素被加到一起，
        # 并以此计算关于[w,b]的梯度
        l.sum().backward()
        sgd([w, b], lr, batch_size)  # 使用参数的梯度更新参数
    with torch.no_grad():
        train_l = loss(net(features, w, b), labels)
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')
```

### 简洁实现

**调API**

#### 读取数据集

我们将`features`和`labels`作为API的参数传递，并通过数据迭代器指定`batch_size`。

此外，布尔值`is_train`表示是否希望数据迭代器对象在每个迭代周期内打乱数据。

```python
import numpy as np
import torch

true_w = torch.tensor([2,-3,4])
true_b = 4.2
features, labels = d2l.synthetic_data(true_w, true_b, 1000)
#造点集

def load_array(data_arrays, batch_size, is_train=True):  #@save
    """构造一个PyTorch数据迭代器"""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)
# dataloader每次随机挑选batch_size个样本
batch_size = 10
data_iter = load_array((features, labels), batch_size)

from torch import nn
net = nn.Sequential(nn.Linear(2, 1))
# 全连接层/线性层 两个参数：输入数和输出数
# Sequential是大容器

net[0].weight.data.normal_(0,0.01)
net[0].bias.data.fill_(0)
# 初始化模型参数

loss = nn.MSELoss()

trainer = torch.optim.SGD(net.parameters(),lr=0.03)
# 实例化SGD，更新模型参数

num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(X), y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
    l = loss(net(features), labels)
    print(f'epoch {epoch+1}, loss {l:f}')55
```

实例化 SGD (随机梯度下降) 随机在于每次更新参数随机使用一小批样本

- 指创建一个 `torch.optim.SGD` 对象，用于更新模型参数。`

- SGD 需要损失函数通过 `backward()` 提供的梯度来更新参数。



## Softmax Regression

$$
softmax(\textbf X)_{ij} = \frac{\exp(\textbf X_{ij})}{\sum_k\exp(\textbf X_{ik})}
$$

### 完整实现

```python
import torch

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
# 迭代器，按批处理，支持打乱，可循环

num_inputs = 784
num_outputs = 10

W = torch.normal(0,0.01,size=(num_inputs, num_outputs),requires_grad=True)
# 行数为输入个数，列数为输出个数
b = torch.zeros(num_outputs, requires_grad=True)

def softmax(X):
	X_exp = torch.exp(X)
	partition = X_exp.sum(1,keepdim=True)
	# 按行求和
	return X_exp / partition # 这里应用了广播机制 第i行除第i个元素

def net(X):
    return softmax(torch.matmul(X.reshape((-1,W.shape[0])), W) + b)


'''交叉熵引入语法
y = torch.tensor([0,2])
y_hat = torch.tensor([[0.1,0.3,0.6],[0.3,0.2,0.5]])
y_hat[[0,1],y]
得到输出 tensor([0.1, 0.5])
'''

def cross_entropy(y_hat, y):
    return -torch.log(y_hat[range(len(y_hat)),y])

def accuracy(y_hat, y):
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())

def evaluate_accuracy(net. data_iter):
    if isinstance(net, torch.nn.Module):
        net.eval() # 将模型设置为评估模式
    metric = Accumulator(2) # 正确预测数、预测总数
    for X, y in data_iter:
        metric.add(accuracy(net(X), y),y.numel())
    return metric[0] / metric[1]

class Accumulator:
    def __init__(self, n):
        self.data = [0,0] * n
    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]
    def reset(self):
        self.data = [0.0] * len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]
    
def train_epoch(net, train_iter, loss, updater):
    if isinstance(net, torch.nn.Module):
        net.train()
    metric = Accumulator(3)
    for X, y in train_iter:
        y_hat = net(X)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.backward()
            updater.step()
            metric.add(
            	float(l) * len(y), accuracy(y_hat, y),y.size().numel())
        else:
            l.sum().backward()
            updater(X.shape[0])
            metric.add(float(l.sum()),accuracy(y_hat, y), y.numel())
    return metric[0] / metric[2], metric[1] / metric[2]

def train(net, train_iter, test_iter, loss, num_epochs, updater):
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                        legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        animator.add(epoch + 1, train_metrics + (test_acc,))
    train_loss, train_acc = train_metrics
    assert train_loss < 0.5, train_loss
    assert train_acc <= 1 and train_acc > 0.7, train_acc
    assert test_acc <= 1 and test_acc > 0.7, test_acc
```

### 简洁实现

```python
import torch

batch_size = 256
train_iter....

# PyTorch 不会隐式地调整输入的形状
# 因此定义展平层，在线性层前调整网络输入的形状

net = nn.Sequential(nn.Flatten(), nn.Linear(784,10))

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)
net.apply(init_weights)

loss = nn.CrossEntropyLoss()

trainer = torch.optim.SGD(net.parameters(),lr=0.1)

num_epochs = 10
d2l.train(net,train_iter,test_iter,loss,num_epochs,trainer)
```





