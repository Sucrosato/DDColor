"""Plot publication-quality training curves from CSV data.

Usage:
  python scripts/plot_curves.py
"""

import os
import csv

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============================================================
# Configuration
# ============================================================
CSV_DIR = 'E:/work/GraduationProject/data/ds0517'
OUTPUT_DIR = 'results/curves'
BIN_SIZE = 25
DPI = 300
FIGSIZE_WIDE = (10, 5)
FIGSIZE_SQUARE = (7, 5.5)

plt.rcParams.update({
    'font.family': 'SimSun',
    'axes.unicode_minus': False,
    'font.size': 13,
    'axes.labelsize': 15,
    'axes.titlesize': 16,
    'legend.fontsize': 12,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'lines.linewidth': 1.5,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'savefig.bbox': 'tight',
    'savefig.dpi': DPI,
})

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Loss display config: (csv_file, symbol, chinese_name, color)
LOSS_CFG = [
    ('l_g_pix.csv',    r'$L_{pix}$', '像素损失', '#E74C3C'),
    ('l_g_percep.csv', r'$L_{per}$', '感知损失', '#2C3E50'),
    ('l_g_gan.csv',    r'$L_{adv}$', '对抗损失', '#27AE60'),
    ('l_g_color.csv',  r'$L_{col}$', '色彩损失', '#8E44AD'),
]


# ============================================================
# Helpers
# ============================================================
def load_csv(path, convert_to_epochs=True, total_epochs=80):
    steps, values = [], []
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            steps.append(int(row['Step']))
            values.append(float(row['Value']))
    s = np.array(steps)
    if convert_to_epochs:
        max_iter = s[-1]
        s = s * total_epochs / max_iter  # 转换 iter → epoch
    else:
        s = s / 1000  # iter → k
    return s, np.array(values)


def bin_average(s, v, bin_size=BIN_SIZE):
    """每 bin_size 个数据点取平均合并为一个点，末尾不足 bin_size 的丢弃。"""
    n = len(v) // bin_size
    s = s[:n * bin_size]
    v = v[:n * bin_size]
    s_binned = s.reshape(n, bin_size).mean(axis=1)
    v_binned = v.reshape(n, bin_size).mean(axis=1)
    return s_binned, v_binned


def save(fig, name):
    fig.savefig(os.path.join(OUTPUT_DIR, f'{name}.png'), dpi=DPI)
    plt.close(fig)


# ============================================================
# Figure 1: 生成器损失曲线 (4个子图)
# ============================================================
def plot_generator_losses():
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    for ax, (fname, symbol, cname, color) in zip(axes.flat, LOSS_CFG):
        s, v = load_csv(os.path.join(CSV_DIR, fname))
        s_bin, v_bin = bin_average(s, v)
        if 'l_g_color' in fname:
            v_bin = v_bin - 2
        ax.plot(s_bin, v_bin, color=color, linewidth=1.2)
        ax.set_ylabel(f'{cname} {symbol}')
        if ax in (axes[1, 0], axes[1, 1]):
            ax.set_xlabel('训练轮数')

    axes[0, 0].set_title(f'(a) {LOSS_CFG[1][2]} {LOSS_CFG[1][1]}')
    axes[0, 1].set_title(f'(b) {LOSS_CFG[0][2]} {LOSS_CFG[0][1]}')
    axes[1, 0].set_title(f'(c) {LOSS_CFG[2][2]} {LOSS_CFG[2][1]}')
    axes[1, 1].set_title(f'(d) {LOSS_CFG[3][2]} {LOSS_CFG[3][1]}')

    for ax in axes.flat:
        ax.tick_params(axis='both', which='major')
        ax.set_xlim(left=0)

    plt.tight_layout()
    save(fig, 'generator_losses')


# ============================================================
# Figure 2: 判别器损失
# ============================================================
def plot_discriminator():
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)

    s, v = load_csv(os.path.join(CSV_DIR, 'l_d.csv'))
    s_bin, v_bin = bin_average(s, v)
    ax.plot(s_bin, v_bin, color='#E67E22', linewidth=1.2)
    ax.set_xlabel('训练轮数')
    ax.set_ylabel(r'判别器损失 $L_{D}$')
    ax.set_xlim(left=0)

    plt.tight_layout()
    save(fig, 'discriminator_loss')


# ============================================================
# Figure 3: 验证指标 (3个独立图)
# ============================================================
def plot_validation_metrics():
    metrics = [
        ('fid.csv',   'FID',        r'FID $\downarrow$', '#2C3E50'),
        ('psnr.csv',  'PSNR (dB)',  r'PSNR $\uparrow$',  '#27AE60'),
        ('cf.csv',    'Colorfulness', 'Colorfulness',    '#8E44AD'),
    ]

    for fname, ylabel, title, color in metrics:
        fig, ax = plt.subplots(figsize=FIGSIZE_SQUARE)
        s, v = load_csv(os.path.join(CSV_DIR, fname), convert_to_epochs=False)
        ax.plot(s, v, color=color, linewidth=1.2, marker='.', markersize=3)
        ax.set_xlabel('迭代次数 (k)')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xlim(left=0)
        plt.tight_layout()
        save(fig, f'val_{fname.split(".")[0]}')


# ============================================================
# Figure 4: GAN 训练动态 (判别器分数 + GAN 损失对比)
# ============================================================
def plot_gan_dynamics():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 判别器对生成图的评分
    ax = axes[0]
    s, v = load_csv(os.path.join(CSV_DIR, 'fake_score.csv'))
    s_bin, v_bin = bin_average(s, v)
    ax.plot(s_bin, v_bin, color='#2980B9', linewidth=1.2)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=0.7, alpha=0.5)
    ax.set_xlabel('训练轮数')
    ax.set_ylabel('判别器输出')
    ax.set_title('(a) 生成图像的判别器评分')
    ax.set_xlim(left=0)

    # 对抗损失与判别损失
    ax = axes[1]
    s1, v1 = load_csv(os.path.join(CSV_DIR, 'l_g_gan.csv'))
    s2, v2 = load_csv(os.path.join(CSV_DIR, 'l_d.csv'))
    s1_bin, v1_bin = bin_average(s1, v1)
    s2_bin, v2_bin = bin_average(s2, v2)
    ax.plot(s1_bin, v1_bin, color='#27AE60', linewidth=1.2, label=r'$L_{adv}$')
    ax.plot(s2_bin, v2_bin, color='#E67E22', linewidth=1.2, label=r'$L_{D}$')
    ax.set_xlabel('训练轮数')
    ax.set_ylabel('损失值')
    ax.set_title('(b) 对抗损失与判别损失')
    ax.legend()
    ax.set_xlim(left=0)

    plt.tight_layout()
    save(fig, 'gan_dynamics')


# ============================================================
# Figure 5: 损失汇总 (色彩损失减2)
# ============================================================
def plot_all_losses():
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)

    for fname, symbol, cname, color in LOSS_CFG:
        s, v = load_csv(os.path.join(CSV_DIR, fname))
        s_bin, v_bin = bin_average(s, v)
        if 'l_g_color' in fname:
            v_bin = v_bin - 2
        ax.plot(s_bin, v_bin, color=color, linewidth=1.2, label=symbol)

    ax.set_xlabel('训练轮数')
    ax.set_ylabel('损失值')
    ax.legend(ncol=4, framealpha=0.9)
    ax.set_xlim(left=0)

    plt.tight_layout()
    save(fig, 'all_losses')


# ============================================================
# Figure 6: 加权总损失
# ============================================================
def plot_weighted_total():
    WEIGHTS = {
        'l_g_pix.csv':    0.1,
        'l_g_percep.csv': 5.0,
        'l_g_gan.csv':    1.0,
        'l_g_color.csv':  0.5,
    }

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)

    total = None
    s_ref = None
    for fname, symbol, cname, color in LOSS_CFG:
        s, v = load_csv(os.path.join(CSV_DIR, fname))
        s_bin, v_bin = bin_average(s, v)
        if 'l_g_color' in fname:
            v_bin = v_bin - 2
        weighted = v_bin * WEIGHTS[fname]
        if total is None:
            s_ref = s_bin
            total = weighted
        else:
            total += weighted

    ax.plot(s_ref, total, color='#2C3E50', linewidth=1.5)
    ax.set_xlabel('训练轮数')
    ax.set_ylabel('加权总损失')
    ax.set_xlim(left=0)

    plt.tight_layout()
    save(fig, 'weighted_total')


# ============================================================
# Main
# ============================================================
def main():
    print('Plotting training curves...')

    plot_generator_losses()
    print('  [1/5] 生成器损失')

    plot_discriminator()
    print('  [2/8] 判别器损失')

    plot_validation_metrics()
    print('  [3/8] 验证指标 (FID / PSNR / CF)')

    plot_gan_dynamics()
    print('  [4/8] GAN 训练动态')

    plot_all_losses()
    print('  [5/8] 损失汇总')

    plot_weighted_total()
    print('  [6/8] 加权总损失')

    print(f'\nDone. Output: {os.path.abspath(OUTPUT_DIR)}/')


if __name__ == '__main__':
    main()
