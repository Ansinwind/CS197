# Attention is all you need

## Sequence Transduction Model 序列转导模型

序列转导任务

- 文本翻译
- 文本生成
- 语音转文字

Transformer之前主流结构：

1. RNN / CNN 循环神经网络 / 卷积神经网络
2. Encoder-Decoder 编码器-解码器结构
3. Attention mechanism 使用注意力机制增强

Transformer 结构创新：

1. 完全摒弃 RNN / CNN
2. 仍然使用Encoder-Decoder
3. 完全基于注意力机制



## Introduction & Background

### Feedforward Neural Network 前馈神经网络

![image-20250727192529745](C:\Users\chy20\AppData\Roaming\Typora\typora-user-images\image-20250727192529745.png)

W和V为两个权重矩阵

**步骤**

1. 分词(Tokenization)
2. 词向量表示(Embedding)
3. 合并词向量(平均或拼接)

平均会丢失顺序，拼接需要非常长的输入层

### Recurrent Neural Network 循环神经网络

解决的问题

- 能建模词序 RNN是按时间顺序逐个处理输入的
- 能建模上下文依赖
- 支持不定长输入 不再需要FNN固定长度输入格式

![image-20250727201239770](C:\Users\chy20\AppData\Roaming\Typora\typora-user-images\image-20250727201239770.png)![image-20250727201344091](C:\Users\chy20\AppData\Roaming\Typora\typora-user-images\image-20250727201344091.png)

### Encoder - Decoder

处理上下文不等长的情况

![image-20250727201725309](C:\Users\chy20\AppData\Roaming\Typora\typora-user-images\image-20250727201725309.png)

![image-20250727201913495](C:\Users\chy20\AppData\Roaming\Typora\typora-user-images\image-20250727201913495.png)相当于只有h的RNN，得到一个向量C，即上下文语义信息

### Attention Mechanism

解决的问题：

1. 解决模型处理长序列的**遗忘**问题
2. 解决不同时间步输入对当前时刻输出的**重要性**问题

![image-20250727202138980](C:\Users\chy20\AppData\Roaming\Typora\typora-user-images\image-20250727202138980.png)

### 串行化计算

仍有问题 类比于一个不能并行化的加法器



## Transformer架构

### 输入端

1. Input Embedding 把字转化成向量 **编码语义信息**
2. Positional Encoding 通过傅里叶等方法搞出一个位置向量，和原向量相加（相当于打tag） **编码位置信息**

***单头注意力：***

三个权重矩阵 $$W_q，W_k，W_v$$ （512*512）得到Q K V 三个矩阵

query key value 问 键 值

***多头注意力***：

八个512*64 通过线性层 得到和单头一样大小的矩阵

Q：为什么用多个头？ A：每个头关注不同的信息

Q：为什么用线形层？ A：再做一个映射

（性能和计算量的平衡）

3. 注意力机制（自注意力） **编码上下文信息**
4. Add & Norm 残差连接 归一化 （防止面目全非）
5. FNN 增强非信息表达能力
6. 残差链接归一化

### 输出端

基本一致

***带掩码的多头注意力机制***

只给它看前文信息

$$Softmax$$ 得分概率

## 总结

1. 计算复杂度降低
2. **并行计算能力**
3. 模型内部学习**长距离依赖**的能力