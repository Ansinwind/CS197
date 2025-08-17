# 使用块的网络 VGG

## VGG块

- 深度宽度
  - 深 而 窄
- VGG块
  - 3x3卷积 n层m通道
  - 2x2最大池化层 步幅2



![image-20250816135350941](C:\Users\chy20\AppData\Roaming\Typora\typora-user-images\image-20250816135350941.png)

三个参数：

卷积层的数量，输入通道的数量，输出通道的数量

```python
def vgg_block(num_convs, in_channels, out_channels):
    layers = []
    for _ in range(num_convs):
        layers.append(nn.Conv2d(in_channels, out_channels,
                                kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        in_channels = out_channels
    layers.append(nn.MaxPool2d(kernel_size=2,stride=2))
    return nn.Sequential(*layers)
```



## VGG架构

多个VGG块后接全连接层

VGG-16, VGG-19

下展示VGG-11

```python
conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))
def vgg(conv_arch):
    conv_blks = []
    in_channels = 1
    # 卷积层部分
    for (num_convs, out_channels) in conv_arch:
        conv_blks.append(vgg_block(num_convs, in_channels, out_channels))
        in_channels = out_channels

    return nn.Sequential(
        *conv_blks, nn.Flatten(),
        # 全连接层部分
        nn.Linear(out_channels * 7 * 7, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, 10))

net = vgg(conv_arch)
```

