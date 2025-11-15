# Polarized Low-Light Reflection Removal

本项目提供了一个结合低光增强和去反射的深度学习训练脚本，适用于使用偏振数据集（Mono8灰度）进行训练。模型基于PyTorch框架，在一块NVIDIA RTX 3090上即可高效训练。

## 特性
- **双分支架构**：首先通过照明增强分支提升暗光图像亮度，再通过反射抑制分支去除玻璃反射。
- **注意力机制**：在编码-解码过程中引入通道和空间注意力（CBAM）提升细节恢复能力。
- **复合损失函数**：结合L1、SSIM、梯度与反射一致性损失，兼顾结构与纹理。
- **AMP支持**：可选的自动混合精度，充分利用3090显卡。

## 环境准备
```bash
conda create -n polar python=3.9 -y
conda activate polar
pip install -r requirements.txt
```

## 数据集组织
```
dataset/
├── inputs/        # 经过玻璃拍摄的暗光图像
├── targets/       # 去掉玻璃直接拍摄的参考图像
└── reflections/   # 用黑布遮挡后获得的反射图
```
> 三个文件夹中的文件名需一一对应。

## 训练
```bash
bash scripts/train.sh
```
或自行指定参数：
```bash
python train.py \
  --input_dir dataset/inputs \
  --target_dir dataset/targets \
  --reflection_dir dataset/reflections \
  --output_dir runs/polar_reflection \
  --epochs 300 \
  --batch_size 8 \
  --learning_rate 2e-4 \
  --amp
```

### 关键参数
- `--val_split`：划分验证集比例，默认0.1。
- `--amp`：启用自动混合精度，推荐在3090上开启。
- `--resume`：从现有检查点继续训练。

训练过程中将自动保存每个epoch的权重，并在`output_dir`下保留`best.pth`。

## 推理
```bash
python infer.py \
  --input_dir dataset/inputs \
  --reflection_dir dataset/reflections \
  --weights runs/polar_reflection/best.pth \
  --output_dir results
```

输出图像为去反射且增强亮度的单通道PNG/JPG。

## 项目结构
```
.
├── train.py                 # 训练入口
├── infer.py                 # 推理脚本
├── requirements.txt
├── scripts/
│   └── train.sh             # 训练示例脚本
└── src/
    ├── data/
    │   └── polar_dataset.py
    ├── models/
    │   └── polar_reflection_net.py
    ├── modules/
    │   └── losses.py
    └── utils/
        └── train_utils.py
```

## 许可证
MIT
