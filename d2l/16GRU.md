# 门控循环 GRU



关注一个序列不是每个观察值都同等重要



>  [!tip]
>
> **想只记住相关的观察需要**
>
> - 能关注的机制（更新门）
> - 能遗忘的机制（重置门）


$$
R_t=\sigma(X_tW_{xr}+H_{t-1}W_{hr}+b_r)
$$

$$
Z_t=\sigma(X_tW_{xz}+H_{t-1}W_{hz}+b_z)
$$



**候选隐状态(由R进行控制)**
$$
\tilde H_t = \tanh (X_tW_{xh}+(R_t\odot H_{t-1})W_{hh}+b_h)
$$


**隐状态**
$$
H_t=Z_t\odot H_{t-1}+(1-Z_t)\odot \tilde H_t
$$
