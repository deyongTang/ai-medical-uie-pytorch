#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
手动数据转换示例
将 CMeIE 格式转换为 UIE 格式
"""

import json


def cmeie_to_uie_samples(cmeie_data):
    """
    将一条 CMeIE 数据转换为多条 UIE 训练样本
    
    Args:
        cmeie_data: CMeIE 格式的数据（dict）
    
    Returns:
        list: UIE 格式的样本列表
    """
    text = cmeie_data['text']
    spo_list = cmeie_data['spo_list']
    
    uie_samples = []
    
    # 用于去重的集合
    seen_entities = set()
    seen_relations = set()
    
    for spo in spo_list:
        subject = spo['subject']
        subject_type = spo['subject_type']
        predicate = spo['predicate']
        obj = spo['object']['@value']
        obj_type = spo['object_type']['@value']
        
        # 1. Subject 实体抽取样本
        subject_key = (subject_type, subject)
        if subject_key not in seen_entities:
            seen_entities.add(subject_key)
            
            start = text.find(subject)
            if start != -1:
                uie_sample = {
                    "content": text,
                    "prompt": subject_type,
                    "result_list": [
                        {
                            "text": subject,
                            "start": start,
                            "end": start + len(subject)
                        }
                    ]
                }
                uie_samples.append(uie_sample)
        
        # 2. Object 实体抽取样本
        obj_key = (obj_type, obj)
        if obj_key not in seen_entities:
            seen_entities.add(obj_key)
            
            start = text.find(obj)
            if start != -1:
                uie_sample = {
                    "content": text,
                    "prompt": obj_type,
                    "result_list": [
                        {
                            "text": obj,
                            "start": start,
                            "end": start + len(obj)
                        }
                    ]
                }
                uie_samples.append(uie_sample)
        
        # 3. 关系抽取样本
        relation_key = (subject, predicate, obj)
        if relation_key not in seen_relations:
            seen_relations.add(relation_key)
            
            start = text.find(obj)
            if start != -1:
                uie_sample = {
                    "content": text,
                    "prompt": f"{subject}的{predicate}",
                    "result_list": [
                        {
                            "text": obj,
                            "start": start,
                            "end": start + len(obj)
                        }
                    ]
                }
                uie_samples.append(uie_sample)
    
    return uie_samples


def main():
    """主函数：演示转换过程"""
    
    # 示例 CMeIE 数据
    cmeie_example = {
        "text": "糖尿病患者应该服用二甲双胍控制血糖",
        "spo_list": [
            {
                "subject": "糖尿病",
                "subject_type": "疾病",
                "predicate": "药物治疗",
                "object": {
                    "@value": "二甲双胍"
                },
                "object_type": {
                    "@value": "药物"
                }
            }
        ]
    }
    
    print("=" * 70)
    print("CMeIE 到 UIE 格式转换示例")
    print("=" * 70)
    
    print("\n【输入】CMeIE 格式:")
    print(json.dumps(cmeie_example, ensure_ascii=False, indent=2))
    
    # 执行转换
    uie_samples = cmeie_to_uie_samples(cmeie_example)
    
    print(f"\n【输出】生成了 {len(uie_samples)} 条 UIE 样本:\n")
    
    for i, sample in enumerate(uie_samples, 1):
        print(f"样本 {i}:")
        print(json.dumps(sample, ensure_ascii=False, indent=2))
        print()
    
    print("=" * 70)
    
    # 更复杂的例子
    complex_example = {
        "text": "类癌综合征@类癌综合征患者手术前应该开始输注奥曲肽以防止类癌瘤危象。",
        "spo_list": [
            {
                "subject": "类癌综合征",
                "subject_type": "疾病",
                "predicate": "相关（导致）",
                "object": {"@value": "类癌瘤危象"},
                "object_type": {"@value": "疾病"}
            },
            {
                "subject": "类癌瘤危象",
                "subject_type": "疾病",
                "predicate": "药物治疗",
                "object": {"@value": "奥曲肽"},
                "object_type": {"@value": "药物"}
            }
        ]
    }
    
    print("\n\n【复杂示例】包含多个三元组:")
    print(json.dumps(complex_example, ensure_ascii=False, indent=2))
    
    uie_samples_complex = cmeie_to_uie_samples(complex_example)
    
    print(f"\n生成了 {len(uie_samples_complex)} 条 UIE 样本:\n")
    
    for i, sample in enumerate(uie_samples_complex, 1):
        print(f"样本 {i}: Prompt=\"{sample['prompt']}\"")
        for result in sample['result_list']:
            print(f"  → \"{result['text']}\" (位置: {result['start']}-{result['end']})")
    
    print("\n" + "=" * 70)
    print("✅ 转换完成！")


if __name__ == "__main__":
    main()
