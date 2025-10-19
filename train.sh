#!/bin/bash
# =============================================================================
# 脚本名称: 深度学习训练启动脚本
# 功能描述: 配置Swin-UNet模型的训练参数，并启动训练过程。
# 参数说明: 支持通过环境变量动态配置参数，未设置时使用默认值。
# =============================================================================

# 设置训练轮数，如果未设置环境变量epoch_time，则使用默认值150[6,8](@ref)
if [ $epoch_time ]; then
    EPOCH_TIME=$epoch_time
else
    EPOCH_TIME=150
fi

# 设置模型输出目录，如果未设置环境变量out_dir，则默认输出到'./models'
if [ $out_dir ]; then
    OUT_DIR=$out_dir
else
    OUT_DIR='./models'
fi

# 设置模型配置文件路径，如果未设置环境变量cfg，则使用Swin-Tiny架构的默认配置
if [ $cfg ]; then
    CFG=$cfg
else
    CFG='./configs/swin_tiny_patch4_window7_224_lite.yaml'
fi

# 设置数据集路径，如果未设置环境变量data_dir，则默认使用'datasets/Synapse'
if [ $data_dir ]; then
    DATA_DIR=$data_dir
else
    DATA_DIR='./data/ACDC'
fi

# 设置学习率，如果未设置环境变量learning_rate，则使用默认值0.05
if [ $learning_rate ]; then
    LEARNING_RATE=$learning_rate
else
    LEARNING_RATE=0.05
fi

# 设置输入图像尺寸，如果未设置环境变量img_size，则默认使用224x224像素
if [ $img_size ]; then
    IMG_SIZE=$img_size
else
    IMG_SIZE=224
fi

# 设置批量大小，如果未设置环境变量batch_size，则默认使用24
if [ $batch_size ]; then
    BATCH_SIZE=$batch_size
else
    BATCH_SIZE=6
fi

# 输出开始训练信息，并调用Python训练脚本启动训练过程[4](@ref)
echo "start train model"
python train.py --dataset Synapse --cfg $CFG --root_path $DATA_DIR --max_epochs $EPOCH_TIME --output_dir $OUT_DIR --img_size $IMG_SIZE --base_lr $LEARNING_RATE --batch_size $BATCH_SIZE