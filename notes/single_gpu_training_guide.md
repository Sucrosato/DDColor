# DDColor 单 GPU 训练完整指南

> 适用场景：单卡机器 + ConvNeXt-Tiny 编码器 + 自定义小数据集

---

## 一、环境准备

```bash
# 克隆项目 & 安装
git clone <repo-url> DDColor
cd DDColor
pip install -e .

# 确认 PyTorch 版本 >= 1.13 且支持 CUDA
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

---

## 二、准备预训练权重

训练需要 2 个预训练文件，放入 `pretrain/` 目录：

| 文件 | 用途 | 大小 |
|---|---|---|
| `convnext_tiny_22k_224.pth` | 生成器 Encoder 骨干权重 | ~110 MB |
| `inception_v3_google-1a9a5a14.pth` | FID 验证指标（可选） | ~104 MB |

### 下载 convnext_tiny

```bash
# 方式 1：从 OpenMMLab 下载（推荐）
wget -P pretrain/ https://download.openmmlab.com/mmclassification/v0/convnext/convnext-tiny_3rdparty_32xb128-noema_in1k_20220622-753c473c.pth -O pretrain/convnext_tiny_22k_224.pth

# 方式 2：torchvision 导出
python -c "
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
m = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
torch.save({'model': m.state_dict()}, 'pretrain/convnext_tiny_22k_224.pth')
"
```

### 下载 Inception V3（用于 FID）

```bash
wget -P pretrain/ https://github.com/mseitzer/pytorch-fid/releases/download/fid_weights/pt_inception-2015-12-05-6726825d.pth -O pretrain/inception_v3_google-1a9a5a14.pth
```

> 如果不需要 FID 指标，可以从配置中删除 `val.metrics.fid`，跳过此步。

---

## 三、准备数据集

### 3.1 数据目录结构

```
datasets/
  └── custom/
      ├── images/              # 训练图像
      │   ├── img001.jpg
      │   ├── img002.jpg
      │   └── ...
      └── val/                 # 验证图像（可选）
          ├── img101.jpg
          └── ...
```

支持格式：jpg, jpeg, png, JPG, JPEG, PNG。

### 3.2 生成数据列表文件

```bash
python scripts/get_meta_file.py \
    --output-name data_list/custom_train.txt \
    --data-path datasets/custom/images

python scripts/get_meta_file.py \
    --output-name data_list/custom_val.txt \
    --data-path datasets/custom/val
```

`data_list/custom_train.txt` 中每行是一个绝对路径：

```
/home/user/DDColor/datasets/custom/images/img001.jpg
/home/user/DDColor/datasets/custom/images/img002.jpg
...
```

### 3.3 关于数据集大小

| 数据集规模 | 建议 `total_iter` | 说明 |
|---|---|---|
| < 1,000 张 | 20,000 ~ 50,000 | 容易过拟合，建议降低 `save_checkpoint_freq` |
| 1,000 ~ 10,000 张 | 50,000 ~ 100,000 | 本指南默认值 |
| > 10,000 张 | 100,000 ~ 200,000 | 可适当增加 milestones 数量 |

---

## 四、配置文件

训练配置文件已创建：**`options/train/train_ddcolor_tiny_single.yml`**

### 与原版大型配置的差异

| 参数 | 原版（large） | 新版（tiny） | 原因 |
|---|---|---|---|
| `encoder_name` | `convnext-l` | `convnext-t` | Tiny 模型参数量减少约 4 倍 |
| `num_gpu` | 4 | 1 | 单卡 |
| `total_iter` | 400,000 | 100,000 | 小数据集无需过长训练 |
| `milestones` | 8 个 | 4 个 | 适配 shorter training |
| `save_checkpoint_freq` | 10,000 | 5,000 | 小数据集需更频繁保存 |
| `val_freq` | 10,000 | 2,000 | 更频繁验证 |
| `print_freq` | 100 | 50 | |
| `cutmix / fmix` | True | False | 小数据集下可能引入噪声 |
| `wandb` | 需配置 | 默认关闭 | 简化设置 |

### 关键参数说明

```yaml
# --- 如需修改 ---
datasets.train.dataroot_gt      # 你的训练图片目录
datasets.train.meta_info_file    # 你的数据列表文件
train.total_iter                 # 总迭代轮数
train.scheduler.milestones       # lr 衰减节点
train.optim_g.lr                 # 生成器学习率 (推荐 1e-4)
train.optim_d.lr                 # 判别器学习率 (推荐 1e-4，不大于 G)
train.pixel_opt.loss_weight      # L1 loss 权重
train.perceptual_opt.perceptual_weight  # 感知 loss 权重
```

---

## 五、启动训练

### 5.1 基础命令

```bash
# 确保在项目根目录
cd DDColor

# 启动训练
python basicsr/train.py -opt options/train/train_ddcolor_tiny_single.yml
```

### 5.2 通过命令行覆盖配置

```bash
python basicsr/train.py -opt options/train/train_ddcolor_tiny_single.yml \
    --name my_first_experiment \
    --datasets.train.dataroot_gt /data/my_images \
    --datasets.train.meta_info_file "['data_list/my_train.txt']" \
    --train.total_iter 50000 \
    --train.optim_g.lr 5e-5
```

### 5.3 训练输出结构

训练开始后会在 `experiments/<name>/` 下生成：

```
experiments/train_ddcolor_tiny/
├── log/
│   └── train_train_ddcolor_tiny_20260513_120000.log    # 训练日志
├── models/
│   ├── net_g_10000.pth      # 生成器 checkpoint
│   ├── net_d_10000.pth      # 判别器 checkpoint
│   ├── net_g_latest.pth     # 最新模型
│   └── ...
├── training_states/
│   ├── 10000.state           # 优化器 + 调度器状态（断点续训用）
│   └── ...
├── visualization/            # 验证输出图像
│   └── ...
└── train_ddcolor_tiny.yml    # 使用的配置副本
```

`tb_logger/` 目录存放 TensorBoard 事件文件：

```bash
tensorboard --logdir tb_logger/
```

---

## 六、训练过程详解

每个 iteration 的完整计算图：

```
┌─────────────────────────────────────────────────────────┐
│  输入: lq_rgb (伪灰度 RGB, 3×256×256)                      │
│        gt     (ab 色度通道, 2×256×256)                      │
│        gt_rgb (真彩色 RGB, 3×256×256)                       │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ① 更新 G (net_d 梯度冻结)                                 │
│     output_ab = net_g(lq_rgb)          # Transformer 解码  │
│     l_g = λ1·L1(output_ab, gt)                            │
│          + λ2·Perceptual(output_rgb, gt_rgb)               │
│          + λ3·GAN(net_d(output_rgb), is_real=True)         │
│          + λ4·Colorfulness(output_rgb)                     │
│     backward → optimizer_g.step()                          │
│                                                         │
│  ② 更新 D (net_d 梯度解冻)                                 │
│     real_score = net_d(gt_rgb)                             │
│     fake_score = net_d(output_rgb.detach())                 │
│     l_d = GANLoss(real, True) + GANLoss(fake, False)       │
│     backward → optimizer_d.step()                          │
│                                                         │
│  ③ 更新 EMA (如果 ema_decay > 0)                           │
│     net_g_ema = decay × net_g_ema + (1-decay) × net_g      │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

**Encoder 始终处于 eval 模式**，不更新参数。只有 Decoder、ColorDecoder、RefineNet 和 Discriminator 参与训练。

---

## 七、监控与验证

### 7.1 TensorBoard

```bash
tensorboard --logdir tb_logger/ --port 6006
```

关注指标：
- `l_g_pix` + `l_g_percep` + `l_g_gan` + `l_g_color`：各项 loss 应平稳下降
- `l_d`：判别器 loss，应在 ~0.5~2.0 之间（vanilla GAN 理论值为 ln4 ≈ 1.38）
- `real_score` / `fake_score`：不应极端靠近 1 或 0（D 过强 / G 过强）
- `metrics/cf`：Colorfulness 指标，越高越好
- `metrics/psnr`：PSNR 指标（如有 GT）

### 7.2 训练快照

每 `save_snapshot_freq` 次迭代，训练图像会自动保存到：
```
experiments/<name>/training_images_snapshot/
```

包含 `lq`（灰度输入）、`result`（上色结果）、`gt`（真实彩色）。

### 7.3 损失不收敛排查

| 现象 | 可能原因 | 解决 |
|---|---|---|
| `l_d` 接近 0 | D 过强，G 无法欺骗 | 降低 `optim_d.lr`，或增大 `gan_opt.loss_weight` |
| `l_g_gan` 持续高位 | D 过强 | 同上 |
| `fake_score` 接近 0 | G 生成质量差 | 检查预训练 encoder 是否正确加载 |
| `l_g_pix` 不下降 | 学习率过大或过小 | 调整 `optim_g.lr` |
| loss 全为 NaN | 学习率过大 | 降低 lr，增加 warmup |

---

## 八、断点续训

训练意外中断后，自动续训：

```bash
# auto_resume 会自动找到最新的 .state 文件
python basicsr/train.py -opt options/train/train_ddcolor_tiny_single.yml --auto_resume
```

或手动指定：

```bash
python basicsr/train.py -opt options/train/train_ddcolor_tiny_single.yml \
    --path.resume_state experiments/train_ddcolor_tiny/training_states/10000.state
```

---

## 九、推理测试

训练完成后，用 checkpoint 做推理：

```bash
python scripts/infer.py \
    --model_path experiments/train_ddcolor_tiny/models/net_g_latest.pth \
    --input assets/test_images \
    --output results/my_test \
    --input_size 512 \
    --model_size tiny
```

或在 Python 中调用：

```python
import cv2
from ddcolor import DDColor, ColorizationPipeline, build_ddcolor_model

model = build_ddcolor_model(
    DDColor,
    model_path="experiments/train_ddcolor_tiny/models/net_g_latest.pth",
    input_size=512,
    model_size="tiny",
    device="cuda",
)
colorizer = ColorizationPipeline(model, input_size=512)

img = cv2.imread("test.jpg")
result = colorizer.process(img)
cv2.imwrite("result.jpg", result)
```

---

## 十、性能参考

| 组件 | Tiny | Large |
|---|---|---|
| Encoder 参数量 | ~28M | ~198M |
| Encoder 特征维度 | [96, 192, 384, 768] | [192, 384, 768, 1536] |
| 单卡 256×256 训练显存 | ~8 GB | ~16 GB |
| 推理速度 (512px) | ~0.3s | ~0.8s |

> 显存占用 = Encoder 激活 + Decoder + Discriminator + 优化器状态。Tiny 版本单卡 8GB 显存即可训练。

---

## 十一、常见问题

**Q: 图像有严重的紫色/绿色伪影？**
A: 训练不充分或数据集太小。增加 `total_iter`，或增大 `pixel_opt.loss_weight`。

**Q: 生成结果色彩过于平淡？**
A: 增大 `colorfulness_opt.loss_weight`，或开启 `color_enhance`。

**Q: 训练后期 loss 震荡？**
A: 正常现象，GAN 训练天然不稳定。降低 lr 或增加 `save_checkpoint_freq`，选择验证指标最优的 checkpoint。

**Q: 我只想做推理，不需要训练？**
A: 直接使用 `scripts/infer.py`，从 HuggingFace 加载预训练模型即可：

```bash
python scripts/infer.py --model_name ddcolor_modelscope --input ./images
```
