#!/bin/bash

# 确保脚本出错时停止
set -e

echo "🚀 开始 UIE 模型实战训练 (Debug 模式)..."

# 检查是否安装了必要的库
# pip install -r uie_pytorch/requirements.txt

# 设置环境变量 (可选)
export CUDA_VISIBLE_DEVICES=0

# 运行训练脚本
# 参数说明:
# --train_path: 训练数据路径
# --dev_path: 验证数据路径
# --save_dir: 模型保存路径
# --learning_rate: 学习率 (Debug模式设大一点以便快速收敛，或者保持默认)
# --batch_size: 批大小 (Debug数据少，设小一点)
# --max_seq_len: 最大序列长度
# --num_epochs: 训练轮数 (Debug模式跑几轮看看效果)
# --model: 预训练模型名称 (这里使用 uie_base_pytorch)
# --device: 使用 cpu 还是 gpu (默认 gpu, 如果报错请改为 cpu)

python3 uie_pytorch/finetune.py \
    --train_path "debug_data/train_converted.jsonl" \
    --dev_path "debug_data/dev_converted.jsonl" \
    --save_dir "./checkpoint_debug" \
    --learning_rate 1e-4 \
    --batch_size 4 \
    --max_seq_len 512 \
    --num_epochs 10 \
    --model "bert-base-chinese" \
    --logging_steps 2 \
    --valid_steps 5 \
    --device "cpu" 

echo "✅ 训练完成！模型已保存到 ./checkpoint_debug"
