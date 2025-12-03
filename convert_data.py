import json
import argparse
import os
from tqdm import tqdm

def cmeie_to_uie_samples(cmeie_data):
    """
    将一条 CMeIE 数据转换为多条 UIE 训练样本
    """
    text = cmeie_data['text']
    spo_list = cmeie_data.get('spo_list', [])
    
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

def convert_file(input_path, output_path):
    print(f"Converting {input_path} to {output_path}...")
    with open(input_path, 'r', encoding='utf-8') as f_in, \
         open(output_path, 'w', encoding='utf-8') as f_out:
        
        for line in tqdm(f_in):
            try:
                cmeie_data = json.loads(line)
                uie_samples = cmeie_to_uie_samples(cmeie_data)
                for sample in uie_samples:
                    f_out.write(json.dumps(sample, ensure_ascii=False) + '\n')
            except Exception as e:
                print(f"Error processing line: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    args = parser.parse_args()
    
    convert_file(args.input_file, args.output_file)
