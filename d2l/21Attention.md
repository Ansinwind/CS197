# Attention Mechanism 注意力机制



- 卷积、全连接、池化层都只考虑不随意线索
- 注意力机制则显示地考虑随意线索
  - 随意线索被称之为查询（query）
  - 每个输入是一个值（value）和不随意线索（key）的对
  - 通过注意力池化层来有偏向性地选择某些输入



## 非参注意力池化层

- 给定数据
- 平均池化是最简单的方案
- Nadarajah-Watson核回归

$$
f(x)=\sum_{i=1}^n\frac{K(x-x_i)}{\sum_{j=1}^nK(x-x_j)}y_i
$$

- 使用高斯核

$$
K(u)=\frac1{\sqrt{2\pi}}\exp(-\frac{u^2}2)
$$

那么
$$
f(x)=\sum_{i=1}^n softmax(-\frac12(x-x_i)^2)y_i
$$


## 参数化的注意力机制

- 在之前基础上引入可学习的w

$$
f(x)=\sum_{i=1}^n softmax(-\frac12((x-x_i)w)^2)y_i
$$

## 注意力分数

> [!tip]
>
> **注意力权重** $ \alpha (x,x_i)$
>
> **注意力分数** $ -\frac12(x-x_i)^2 $
> $$
> \sum_i\alpha(x,x_i)y_i = \sum_{i=1}^nsoftmax(-\frac12(x-x_i)^2)y_i
> $$

### 拓展到高维

 

> [!important]
>
> - 假设query $ \bf q \in \mathbb R^q$, 有$ m $对key-value$ (\bf k_i,\bf v_i) $,其中 $ \bf k_i \in \mathbb R^k, \bf v_i \in \mathbb R^n $
> - 注意力池化层：
>
> $$
> f(\textbf q,(\textbf k_1,\textbf v_1),...,(\textbf k_m,\textbf v_m))=\sum_{i=1}^m\alpha(\textbf q,\textbf k_i)\textbf v_i \in \mathbb R^v
> $$
>
> $$
> \alpha(\textbf q,\textbf k_i) = softmax(a(\textbf q,\textbf k_i))=\frac{\exp(a(\textbf q,\textbf k_i))}{\sum_{j=1}^m\exp(a(\textbf q,\textbf k_j)} \in \mathbb R
> $$

### Additive Attention

$$
a(\textbf k,\textbf q)=\textbf v^T \tanh(\textbf W_k\textbf k+\textbf W_q\textbf q)
$$



- k一个，q一个，v一个

- 等价于将key核value合并起来后放入一个隐藏大小为h输出大小为1的但隐藏层MLP

### Scaled Dot-Product Attention



- 注意力分散是query和key的相似度，注意力权重是分数softmax的结果
- 常见的分数计算：
  - 将query和key合并起来进入一个单输出单隐藏层的MLP
  - 直接将query和key做内积

