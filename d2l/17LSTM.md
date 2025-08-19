# LSTM 长短期记忆网络



- 忘记门 将值朝0减少
- 输入门 决定是不是忽略输入数据
- 输出门 决定是不是使用隐状态



> [!tip]
>
> **Forget Gate**
>
> $$
> F_t=\sigma (X_tW_{xf}+H_{t-1}W_{hf}+b_f)
> $$
>
> **Input Gate**
>
> $$
> I_t=\sigma (X_tW_{xi}+H_{t-1}W_{hi}+b_i)
> $$
>
> **Output Gate**
>
> $$
> O_t=\sigma (X_tW_{xo}+H_{t-1}W_{ho}+b_o)
> $$
> 

上面几个0到1

C是-1到1



**候选记忆单元**

$$
\tilde C_t =\tanh(X_tW_{xc}+H_{t-1}W_{hc}+b_c)
$$

**记忆单元**

$$
C_t=F\odot C_{t-1} + I_t\odot \tilde C_t
$$

**隐状态**

$$
H_t=O_t\odot \tanh(C_t)
$$

加tanh只是限制到-1到1




