# 息肉分割
为了重新评估,所有的模型均重新训练测试：
CPU: Intel i9-9900K
GPU: NVIDIA GeForce RTX 2080 Ti

# 模型比较
U_Net
MIA_Net
Polyp_Segmentation

# 依赖安装
pip install -r requirements.txt

# 数据下载
以下文件权重过大，劳烦手动下载感谢~~
[![数据集](https://colab.research.google.com/assets/colab-badge.svg)](https://pan.baidu.com/s/1ESD2xBrMHC3OA7QVbsbEkw)[code:zyh1]
[![骨干网络](https://colab.research.google.com/assets/colab-badge.svg)](https://pan.baidu.com/s/1glbeLchNfhaFzZuEST656A)[code:zyh2]
[![模型结果](https://colab.research.google.com/assets/colab-badge.svg)](https://pan.baidu.com/s/1GTdqKJg6AZG4aJJXHa54XQ)[code:zyh3]

# 项目结构
```
│  config.py 配置文件
│  readme 说明文件
│  requirements.txt 环境依赖
│  test.py 测试代码
│  train.py 训练代码
├─backbone 网络结构
│  │  MDCA_Net.py
│  │  MIA_Net.py
│  │  Polyp_PVT.py
│  │  U_Net.py
│  ├─cswin
│  ├─p2t
│  ├─pvt
│  ├─rest
│  ├─segformer
│  ├─swin
│  ├─twins
├─dataset 数据
│  ├─TestDataset
│  └─TrainDataset
├─dice dice对比图
│  │   plot_dice.py
├─edge 边缘对比图
│  │   show_edge.py
├─hotmap 注意力对比图
│  │  plot_hotmap.py
├─log 训练日志
├─model_pth 训练模型
├─result_map 分割结果
├─utils 工具文件
│  │  dataloader.py
│  │  utils.py
```