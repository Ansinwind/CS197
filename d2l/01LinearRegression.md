# 线性回归

![image-20250803181422763](C:\Users\chy20\AppData\Roaming\Typora\typora-user-images\image-20250803181422763.png)

## 原理

### 随机梯度下降

$$\begin{aligned} \mathbf{w} &\leftarrow \mathbf{w} -  \frac{\eta}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} \partial_{\mathbf{w}} l^{(i)}(\mathbf{w}, b) = \mathbf{w} - \frac{\eta}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} \mathbf{x}^{(i)} \left(\mathbf{w}^\top \mathbf{x}^{(i)} + b - y^{(i)}\right),\\ b &\leftarrow b -  \frac{\eta}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} \partial_b l^{(i)}(\mathbf{w}, b)  = b - \frac{\eta}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} \left(\mathbf{w}^\top \mathbf{x}^{(i)} + b - y^{(i)}\right). \end{aligned}$$

可以通过求$L(w,b)$关于w和b的偏导得到

矢量加速：把x、w等做成矢量，加速在python中的运算

### 极大似然估计

噪声假设：正态分布
在这种情况下用极大似然估计偏导得到的w和b与随机梯度下降一致



## 基本实现

### 生成数据集

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

### 读取数据集

```python
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    # 这些样本是随机读取的，没有特定的顺序
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

当我们运行迭代时，我们会连续地获得不同的小批量，直至遍历完整个数据集。上面实现的迭代执行效率很低，可能会在实际问题上陷入麻烦。

在深度学习框架中实现的内置迭代器效率要高得多，可以处理存储在文件中的数据和数据流提供的数据。

### 初始化模型参数

```python
w = torch.normal(0, 0.01, size=(2,1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)
```

### 定义模型

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

### 训练

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

## 简洁实现

**调API**

### 读取数据集

我们将`features`和`labels`作为API的参数传递，并通过数据迭代器指定`batch_size`。

此外，布尔值`is_train`表示是否希望数据迭代器对象在每个迭代周期内打乱数据。

```python
def load_array(data_arrays, batch_size, is_train=True):  #@save
    """构造一个PyTorch数据迭代器"""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)
batch_size = 10
data_iter = load_array((features, labels), batch_size)
```

### 调用

1. import `nn`包

2. 首先定义一个模型变量`net`，它是一个`Sequential`类的实例。`Sequential`类将多个层串联在一起。当给定输入数据时，`Sequential`实例将数据传入到第一层，然后将第一层的输出作为第二层的输入，以此类推。

   *在下面的例子中，我们的模型只包含一个层，因此实际上不需要`Sequential`。但是由于以后几乎所有的模型都是多层的，在这里使用`Sequential`熟悉“标准的流水线”。*

3. 通过`net[0]`选择网络中的第一个图层，然后使用`weight.data`和`bias.data`方法访问参数。我们还可以使用替换方法`normal_`和`fill_`来重写参数值。

4. 用均方误差/L2范数

5. PyTorch在`optim`模块中实现了该算法的许多变种。当我们(***实例化一个***`SGD`实例)时，我们要指定优化的参数（可通过`net.parameters()`从我们的模型中获得）以及优化算法所需的超参数字典。小批量随机梯度下降只需要设置`lr`值，这里设置为0.03。

6. 在每个迭代周期里，我们将完整遍历一次数据集（`train_data`），不停地从中获取一个小批量的输入和相应的标签。对于每一个小批量，我们会进行以下步骤:

   1. 通过调用`net(X)`生成预测并计算损失`l`（前向传播）。
   2. 通过进行反向传播来计算梯度。
   3. 通过调用优化器来更新模型参数。

```python
# nn是神经网络的缩写
from torch import nn
net = nn.Sequential(nn.Linear(2, 1))

net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.fill_(0)

loss = nn.MSELoss()

trainer = torch.optim.SGD(net.parameters(), lr=0.03)

num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(X) ,y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
    l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {l:f}')
    
w = net[0].weight.data
print('w的估计误差：', true_w - w.reshape(true_w.shape))
b = net[0].bias.data
print('b的估计误差：', true_b - b)
```

通过深度学习框架的高级API来实现我们的模型只需要相对较少的代码。不必单独分配参数、不必定义我们的损失函数，也不必手动实现小批量随机梯度下降。当我们需要更复杂的模型时，高级API的优势将大大增加。

