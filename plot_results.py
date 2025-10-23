import json
import matplotlib.pyplot as plt
import argparse
import os
import numpy as np


def plot_curves(log_path, save_dir):
    """
    从 JSON 日志文件加载损失数据并使用 Matplotlib 绘制曲线图。
    """

    # 检查日志文件是否存在
    if not os.path.exists(log_path):
        print(f"错误: 日志文件未找到: {log_path}")
        return

    # 加载数据
    try:
        with open(log_path, 'r') as f:
            history = json.load(f)
    except Exception as e:
        print(f"错误: 无法读取 JSON 文件: {e}")
        return

    # 检查数据是否为空
    if not history['train_total_loss']:
        print("错误: 日志文件为空，没有数据可供绘图。")
        return

    # 获取 Epoch 数量
    # 验证集可能比训练集少一个点（如果在第一个 epoch 就评估）
    val_epochs_offset = len(history['train_total_loss']) - len(history['val_total_loss'])
    train_epochs = np.arange(1, len(history['train_total_loss']) + 1)
    val_epochs = np.arange(1 + val_epochs_offset, len(history['val_total_loss']) + 1 + val_epochs_offset)

    # --- 开始绘图 ---
    plt.style.use('ggplot')  # 使用 'ggplot' 风格让图表更美观

    # 创建一个 1x2 的子图布局
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

    # --- 子图 1: 总损失 (Total Loss) ---
    ax1.plot(train_epochs, history['train_total_loss'], 'b-o', label='Training Total Loss', markersize=4)
    ax1.plot(val_epochs, history['val_total_loss'], 'r-s', label='Validation Total Loss', markersize=4)
    ax1.set_title('Training vs. Validation Total Loss', fontsize=16)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.legend(fontsize=12)
    ax1.grid(True)

    # --- 子图 2: Dice 损失 (Dice Loss) ---
    ax2.plot(train_epochs, history['train_dice_loss'], 'b-o', label='Training Dice Loss', markersize=4)
    ax2.plot(val_epochs, history['val_dice_loss'], 'r-s', label='Validation Dice Loss', markersize=4)
    ax2.set_title('Training vs. Validation Dice Loss', fontsize=16)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Dice Loss', fontsize=12)
    ax2.legend(fontsize=12)
    ax2.grid(True)

    # 调整布局
    plt.tight_layout()

    # 保存图表
    save_path = os.path.join(save_dir, "loss_curves.png")
    try:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存到: {save_path}")
    except Exception as e:
        print(f"错误: 无法保存图表: {e}")

    # 可选：显示图表
    # plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="从 JSON 日志绘制训练曲线。")
    parser.add_argument('--log', type=str, required=True,
                        help='指向 "training_losses.json" 文件的路径。')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='保存 PNG 图表的目录。')

    args = parser.parse_args()

    # 确保输出目录存在
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    plot_curves(args.log, args.output_dir)