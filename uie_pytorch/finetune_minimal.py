"""
一个尽量“教学化”的 UIE 训练脚本，比原始 finetune.py 简单很多，
方便你理解从 jsonl 数据到模型训练的完整流程。

使用方式示例（在项目根目录）：

python3 uie_pytorch/finetune_minimal.py \
  --train_path debug_data/train_converted.jsonl \
  --dev_path debug_data/dev_converted.jsonl \
  --model bert-base-chinese \
  --save_dir checkpoint_debug_minimal \
  --device cpu \
  --num_epochs 3 \
  --batch_size 4 \
  --learning_rate 1e-4
"""

import argparse
import os
import time

import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast

from model import UIE
from utils import IEDataset, SpanEvaluator, set_seed


def build_dataloader(path, tokenizer, max_seq_len, batch_size, shuffle):
    """
    从 UIE 格式的 jsonl 构建 DataLoader：
    - IEDataset 负责把一行 json 转成模型需要的张量
    - DataLoader 负责按 batch 打包数据
    """
    dataset = IEDataset(path, tokenizer=tokenizer, max_seq_len=max_seq_len)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def move_batch_to_device(batch, device):
    """
    将一个 batch 中的所有张量移动到指定 device（cpu 或 cuda）。
    batch 的结构为：
      input_ids, token_type_ids, attention_mask, start_ids, end_ids
    """
    input_ids, token_type_ids, att_mask, start_ids, end_ids = batch
    if device == "cuda":
        input_ids = input_ids.cuda()
        token_type_ids = token_type_ids.cuda()
        att_mask = att_mask.cuda()
        start_ids = start_ids.cuda()
        end_ids = end_ids.cuda()
    return input_ids, token_type_ids, att_mask, start_ids, end_ids


def evaluate(model, data_loader, device):
    """
    一个简化版的评估函数：
    - 不做 early stopping
    - 只在整个 dev 集上跑一遍，给出 loss / P / R / F1
    """
    model.eval()
    metric = SpanEvaluator()
    loss_fn = torch.nn.functional.binary_cross_entropy

    total_loss = 0.0
    total_steps = 0

    with torch.no_grad():
        for batch in data_loader:
            input_ids, token_type_ids, att_mask, start_ids, end_ids = move_batch_to_device(
                batch, device
            )

            outputs = model(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=att_mask,
            )
            start_prob, end_prob = outputs[0], outputs[1]

            start_ids = start_ids.type(torch.float32)
            end_ids = end_ids.type(torch.float32)
            loss_start = loss_fn(start_prob, start_ids)
            loss_end = loss_fn(end_prob, end_ids)
            loss = (loss_start + loss_end) / 2.0

            total_loss += float(loss)
            total_steps += 1

            # 评估器需要的是 numpy 数组
            metric.update(
                *metric.compute(
                    start_prob.cpu().numpy(),
                    end_prob.cpu().numpy(),
                    start_ids.cpu(),
                    end_ids.cpu(),
                )
            )

    avg_loss = total_loss / max(total_steps, 1)
    precision, recall, f1 = metric.accumulate()
    return avg_loss, precision, recall, f1


def do_train(args):
    """
    极简训练主流程：
    1. 准备 tokenizer / 模型 / DataLoader
    2. 循环 epoch，做前向 + 反向 + 更新
    3. 每个 epoch 结束后在 dev 上评估一次
    """
    set_seed(args.seed)

    # 1. 加载 tokenizer 和 UIE 模型
    tokenizer = BertTokenizerFast.from_pretrained(args.model)
    model = UIE.from_pretrained(args.model)

    device = "cuda" if args.device == "gpu" and torch.cuda.is_available() else "cpu"
    if device == "cuda":
        model = model.cuda()
    print(f"[INFO] use device = {device}")

    # 2. 准备训练 / 验证 DataLoader
    train_loader = build_dataloader(
        args.train_path, tokenizer, args.max_seq_len, args.batch_size, shuffle=True
    )
    dev_loader = build_dataloader(
        args.dev_path, tokenizer, args.max_seq_len, args.batch_size, shuffle=False
    )

    # 3. 优化器与损失函数
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    loss_fn = torch.nn.functional.binary_cross_entropy

    global_step = 0
    best_f1 = 0.0

    for epoch in range(1, args.num_epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_steps = 0
        start_time = time.time()

        print(f"\n===== Epoch {epoch}/{args.num_epochs} =====")

        for step, batch in enumerate(train_loader, start=1):
            input_ids, token_type_ids, att_mask, start_ids, end_ids = move_batch_to_device(
                batch, device
            )

            outputs = model(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=att_mask,
            )
            start_prob, end_prob = outputs[0], outputs[1]

            start_ids = start_ids.type(torch.float32)
            end_ids = end_ids.type(torch.float32)
            loss_start = loss_fn(start_prob, start_ids)
            loss_end = loss_fn(end_prob, end_ids)
            loss = (loss_start + loss_end) / 2.0

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            epoch_loss += float(loss)
            epoch_steps += 1
            global_step += 1

            if global_step % args.logging_steps == 0:
                avg_loss = epoch_loss / max(epoch_steps, 1)
                elapsed = time.time() - start_time
                print(
                    f"[Train] step={global_step} epoch={epoch} "
                    f"loss={avg_loss:.4f} speed={args.logging_steps/elapsed:.2f} step/s"
                )
                start_time = time.time()

        avg_train_loss = epoch_loss / max(epoch_steps, 1)
        print(f"[Train] Epoch {epoch} finished, avg_loss={avg_train_loss:.4f}")

        # 每个 epoch 结束，在 dev 集上评估一次
        dev_loss, precision, recall, f1 = evaluate(model, dev_loader, device)
        print(
            f"[Eval ] Epoch {epoch} dev_loss={dev_loss:.4f} "
            f"P={precision:.4f} R={recall:.4f} F1={f1:.4f}"
        )

        # 只在 F1 提升时保存模型
        if f1 > best_f1:
            best_f1 = f1
            save_dir = os.path.join(args.save_dir, "model_best")
            os.makedirs(save_dir, exist_ok=True)
            model.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)
            print(f"[INFO] best model updated, F1={best_f1:.4f}, saved to {save_dir}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, required=True,
                        help="UIE 格式训练集 jsonl 路径")
    parser.add_argument("--dev_path", type=str, required=True,
                        help="UIE 格式验证集 jsonl 路径")
    parser.add_argument("--model", type=str, default="bert-base-chinese",
                        help="作为 UIE 编码器的预训练模型名或本地路径")
    parser.add_argument("--save_dir", type=str, default="./checkpoint_debug_minimal",
                        help="模型保存目录")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="训练 batch 大小")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="学习率")
    parser.add_argument("--num_epochs", type=int, default=3,
                        help="训练轮数")
    parser.add_argument("--max_seq_len", type=int, default=512,
                        help="最大序列长度")
    parser.add_argument("--device", type=str, choices=["cpu", "gpu"], default="cpu",
                        help="训练设备，cpu 或 gpu")
    parser.add_argument("--seed", type=int, default=1000,
                        help="随机种子")
    parser.add_argument("--logging_steps", type=int, default=10,
                        help="每多少个 step 打一条训练日志")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    do_train(args)

