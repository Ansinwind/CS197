# 循环神经网络 RNN



- 更新隐藏状态

$$
\textbf h_t=\phi(\textbf W_{hh}\textbf h_{t-1} + \textbf W_{hx}\textbf x_{t-1}+\textbf b_h)
$$

$$
\textbf o_t = \phi(\textbf W_{ho}\textbf h_t +\textbf b_o)
$$

去掉第一项，其实更新隐藏状态就成为了MLP



- 困惑度

平均交叉熵 

- 梯度裁剪



- RNN的输出取决于当下输入和前一时间的隐变量
- 应用语言模型中时，循环神经网络根据当前词预测下一时刻词
- 通常使用困惑度来衡量语言模型的好坏



**Deep RNN**

- 深度循环神经网络使用多个隐藏层来获得更多的非线性性

**Bidirectional RNN**

- 一个前向RNN隐层
- 一个方向RNN隐层
- 合并两个隐状态得到输出



- 通过反向更新的隐藏层来利用方向时间信息
- 通常用来对序列抽取特征、填空，而不是预测未来