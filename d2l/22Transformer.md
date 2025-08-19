# Transformer



## 自注意力



- 给定序列 $\bf x_1,...,x_n, \forall x_i \in \mathbb R^d $
- 自注意力池化层把序列元素当作k,v,q来对序列抽取特征得到 $\bf y_1,...,y_n$

$$
\textbf y_i = f(\textbf x_i,(\textbf x_1,\textbf x_1),...,(\textbf x_n,\textbf x_n)) \in \mathbb R^d
$$

和 CNN, RNN 对比

|            | CNN          | RNN         | 自注意力    |
| ---------- | ------------ | ----------- | ----------- |
| 计算复杂度 | $O(knd^2) $ | $O(nd^2) $ | $O(n^2d) $ |
| 并行度     | $O(n) $     | $O(1) $    | $O(n) $    |
| 最长路径   | $O(n/k) $   | $O(n) $    | $O(1) $    |



### 位置编码

- 自注意力并没有记录位置信息
- 位置编码将位置信息注入到输入里

> [!tip]
>
> 假设长度为 $n$的序列是 $\textbf X \in \mathbb R^{n\times d} $，那么使用位置编码矩阵 $\textbf P \in \mathbb R^{n\times d} $来输出 $\textbf X +\textbf P$作为自编码输入
>
> $$
> p_{i,2j}=\sin (\frac i{10000^\frac{2j}d}),p_{i,zj+1}=\cos(\frac i{10000^\frac{2j}d})
> $$

> [!important]
>
> **相对位置信息**
>
> - 位置于 $i+\delta $处的位置编码可以线性投影位置 $i $处的位置编码来表示
> - 记 $\omega_j=\frac1{10000^\frac{2j}d} $
>
> $$
> [\begin{matrix} {\cos(\delta\omega_j)}&{\sin(\delta\omega_j)} \\ {-\sin(\delta\omega_j)}&{\cos(\delta\omega_j)}\end{matrix}][\begin{matrix} {p_{i,2j}} \\ {p_{i,2j+1}}\end{matrix}] = [\begin{matrix} {p_{i+\delta,2j}} \\ {p_{i+\delta,2j+1}}\end{matrix}]
> $$



- 完全并行、最长序列为1、对长序列计算复杂度高
- 位置编码在输入中加入位置信息，使得自注意力能够记忆位置信息



## Transformer



- 基于编码器-解码器架构（见seq2seq）来处理序列对
- 跟使用注意力的seq2seq不同，Transformer纯基于注意力（没RNN啦）



### 多头注意力 Multi-head Attention

- 对同一key，value，query希望抽取不同的信息
  - 例如短距离关系和长距离关系
- 多头注意力使用h个独立的注意力池化
  - 合并各个头输出得到最终输出

> [!tip]
>
> **有掩码的多头注意力**
>
> - 解码器对序列中一个元素输出时，不应该考虑该元素之后的元素
> - 可以通过掩码来实现



### 基于位置的前馈网络

- 全连接 （Positionwise FFN）
- 将输入形状由 $(b,n,d) $转化为 $(bn,d) $
- 作用两个全连接层
- 输出形状由 $(bn,d) $变化为 $(b,n,d) $
- 等价于两层核窗口为1的一维卷积层



### 层归一化

- 批量归一化对每个特征/通道里元素进行归一化
  - 不适合序列长度会变的NLP
- 层归一化对每个样本里的元素进行归一化

（对变化长度而言更稳定）



### 信息传递

- 编码器种的输出 $\bf y_1,...,y_n $
- 将其作为解码中第$ i $个Transformer块中多头注意力的key和value
  - 它的query来自目标序列
- 意味着编码器和解码器中输出维度是一样的



### 预测

- 预测第 $t+1 $个输出时
- 解码器中输入前 $t $个预测值
  - 在自注意力中，前 $t $个预测值作为key和value，第 $t $个预测值还作为query





- Transformer是一个纯使用注意力的编码-解码器
- 编码器和解码器都有n个transformer块
- 每个块里使用多头自注意力，基于位置的前馈网络，和层归一化

