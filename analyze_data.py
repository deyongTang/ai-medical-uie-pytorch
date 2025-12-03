#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CMeIE 数据集分析脚本
用于统计数据集中的关系类型、实体类型等信息
"""

import json
from collections import Counter
from pathlib import Path


def analyze_cmeie_dataset(file_path):
    """分析 CMeIE 数据集统计信息"""
    
    print(f"📊 开始分析数据集: {file_path}\n")
    
    # 统计变量
    total_samples = 0
    total_spo = 0
    predicates = []  # 关系类型
    subject_types = []  # 主体类型
    object_types = []  # 客体类型
    text_lengths = []  # 文本长度
    
    # 读取数据
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            total_samples += 1
            data = json.loads(line)
            text = data['text']
            text_lengths.append(len(text))
            
            spo_list = data['spo_list']
            total_spo += len(spo_list)
            
            for spo in spo_list:
                predicates.append(spo['predicate'])
                subject_types.append(spo['subject_type'])
                object_types.append(spo['object_type']['@value'])
    
    # 打印基本统计
    print("=" * 60)
    print("📈 基本统计信息")
    print("=" * 60)
    print(f"样本总数: {total_samples:,}")
    print(f"三元组总数: {total_spo:,}")
    print(f"平均每个样本的三元组数: {total_spo / total_samples:.2f}")
    print(f"平均文本长度: {sum(text_lengths) / len(text_lengths):.1f} 字")
    print(f"最短文本: {min(text_lengths)} 字")
    print(f"最长文本: {max(text_lengths)} 字")
    
    # 关系类型统计
    print("\n" + "=" * 60)
    print("🔗 关系类型统计 (Top 15)")
    print("=" * 60)
    predicate_counter = Counter(predicates)
    for i, (pred, count) in enumerate(predicate_counter.most_common(15), 1):
        percentage = count / len(predicates) * 100
        print(f"{i:2d}. {pred:20s}: {count:5,} ({percentage:5.2f}%)")
    
    print(f"\n关系类型总数: {len(predicate_counter)}")
    
    # 实体类型统计
    print("\n" + "=" * 60)
    print("🏷️  实体类型统计")
    print("=" * 60)
    all_types = subject_types + object_types
    type_counter = Counter(all_types)
    for i, (etype, count) in enumerate(type_counter.most_common(), 1):
        percentage = count / len(all_types) * 100
        print(f"{i:2d}. {etype:20s}: {count:5,} ({percentage:5.2f}%)")
    
    # 文本长度分布
    print("\n" + "=" * 60)
    print("📏 文本长度分布")
    print("=" * 60)
    length_ranges = [
        ("0-50字", 0, 50),
        ("51-100字", 51, 100),
        ("101-200字", 101, 200),
        ("201-500字", 201, 500),
        ("500字以上", 501, float('inf'))
    ]
    
    for label, min_len, max_len in length_ranges:
        count = sum(1 for l in text_lengths if min_len <= l <= max_len)
        percentage = count / len(text_lengths) * 100
        print(f"{label:12s}: {count:5,} ({percentage:5.2f}%)")


def sample_data_examples(file_path, n=3):
    """展示数据样例"""
    
    print("\n" + "=" * 60)
    print(f"📝 数据样例 (随机抽取 {n} 条)")
    print("=" * 60)
    
    import random
    
    # 读取所有数据
    samples = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            samples.append(json.loads(line))
    
    # 随机抽取
    selected = random.sample(samples, min(n, len(samples)))
    
    for i, sample in enumerate(selected, 1):
        print(f"\n【样例 {i}】")
        print(f"文本: {sample['text'][:100]}{'...' if len(sample['text']) > 100 else ''}")
        print(f"三元组数量: {len(sample['spo_list'])}")
        
        for j, spo in enumerate(sample['spo_list'][:3], 1):  # 只显示前3个
            print(f"\n  三元组 {j}:")
            print(f"    Subject: {spo['subject']} ({spo['subject_type']})")
            print(f"    Predicate: {spo['predicate']}")
            print(f"    Object: {spo['object']['@value']} ({spo['object_type']['@value']})")
        
        if len(sample['spo_list']) > 3:
            print(f"\n  ... 还有 {len(sample['spo_list']) - 3} 个三元组")


if __name__ == "__main__":
    # 数据文件路径
    data_file = Path(__file__).parent / "data" / "annotated_data" / "CMeIE-V2.jsonl"
    
    if not data_file.exists():
        print(f"❌ 数据文件不存在: {data_file}")
        print("请确保数据文件在正确的位置")
    else:
        # 执行分析
        analyze_cmeie_dataset(data_file)
        
        # 显示样例
        sample_data_examples(data_file, n=3)
        
        print("\n" + "=" * 60)
        print("✅ 分析完成！")
        print("=" * 60)
