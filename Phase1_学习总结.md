# Phase 1 学习总结：项目概览与环境准备

> 📍 **学习目标**：理解 UIE 的核心思想、项目架构和环境配置

---

## ✅ 1. 环境检查结果

你的环境已经就绪：

| 组件 | 版本 | 状态 |
|------|------|------|
| Python | 3.9.6 | ✅ 正常 |
| PyTorch | 2.8.0 | ✅ 正常 |
| Transformers | 4.57.3 | ✅ 正常 |
| NumPy | 2.0.2 | ✅ 正常 |

**结论**：可以直接开始学习和训练！

---

## 📊 2. 数据集概况

### 你的数据集
- **文件位置**: `data/annotated_data/CMeIE-V2.jsonl`
- **样本总数**: **14,338 条**
- **文件大小**: 约 15.6 MB
- **任务类型**: 中文医疗信息抽取

### 数据示例（第一条）

**原始医疗文本**：
```
溶血性贫血
* 获得性溶血性贫血可分为免疫性和非免疫性：
* 自身抗体是免疫介导的溶血性贫血的病因，通常包含在其他自身免疫性疾病
（例如系统性红斑狼疮、类风湿关节炎、硬皮病）中或与淋巴组织增生性疾病
（例如非霍奇金淋巴瘤、慢性淋巴细胞白血病）有关。
```

**包含的知识关系**（SPO三元组）：
```json
{
  "subject": "溶血性贫血",           // 主体
  "predicate": "病理分型",           // 关系类型
  "object": "获得性溶血性贫血"       // 客体
}
```

这条数据包含了 10 个这样的三元组关系！

---

## 🎯 3. UIE 核心概念（重点！）

### 什么是 UIE？

**UIE = Universal Information Extraction（通用信息抽取）**

### 传统方法 vs UIE

#### ❌ 传统方法的问题

```
任务1：命名实体识别（NER） → 需要训练 NER 模型
任务2：关系抽取（RE）      → 需要训练 RE 模型  
任务3：事件抽取（EE）      → 需要训练 EE 模型

问题：
• 每个任务需要独立的模型
• 训练成本高
• 难以迁移到新任务
```

#### ✅ UIE 的创新

```
一个统一的模型 + 不同的 Prompt = 完成所有任务！

[Prompt: 疾病] + UIE模型 → 抽取疾病实体
[Prompt: 症状] + UIE模型 → 抽取症状实体  
[Prompt: XX的病因] + UIE模型 → 抽取病因关系

优势：
• 一个模型搞定所有抽取任务
• 通过改变 Prompt 控制抽取内容
• 零样本/少样本学习能力强
```

### Prompt-based Learning 实例

#### 例子1：抽取疾病实体

```
输入文本: "患者确诊为2型糖尿病"
Prompt: "疾病"

完整输入: [CLS] 疾病 [SEP] 患者确诊为2型糖尿病 [SEP]
                ↑ Prompt

模型输出: "2型糖尿病" (起始位置: 5, 结束位置: 10)
```

#### 例子2：抽取症状实体

```
输入文本: "临床表现为多饮、多尿、体重下降"
Prompt: "症状"

完整输入: [CLS] 症状 [SEP] 临床表现为多饮、多尿、体重下降 [SEP]
                ↑ Prompt

模型输出: 
  - "多饮" (位置: 6-7)
  - "多尿" (位置: 9-10)
  - "体重下降" (位置: 12-15)
```

#### 例子3：抽取关系（更高级）

```
输入文本: "患者确诊为糖尿病，临床表现为多饮多尿"
Prompt: "糖尿病的临床表现"  ← 注意：Prompt包含了关系信息

完整输入: [CLS] 糖尿病的临床表现 [SEP] 患者确诊为糖尿病，临床表现为多饮多尿 [SEP]
                ↑ 关系型Prompt

模型输出: 
  - "多饮" (位置: 16-17)  ← 这些是"糖尿病"的临床表现
  - "多尿" (位置: 17-19)
```

**关键理解**：同样的文本，不同的 Prompt，能抽取出不同的信息！

---

## 🧠 4. 核心技术：指针网络

### 传统 NER 方法（BIO 标注）

```
文本:  患  者  确  诊  为  糖  尿  病
标签:  O   O   O   O   O   B   I   I
      ↑ Outside        ↑ Begin  ↑ Inside

问题：
• 不能很好处理嵌套实体
• 不能处理重叠实体
• 标签序列复杂
```

### UIE 的指针网络

```
文本:  患  者  确  诊  为  糖  尿  病
位置:  0   1   2   3   4   5   6   7

起始指针预测: [0.01, 0.01, 0.02, 0.01, 0.03, 0.95, 0.02, 0.01]
                                              ↑ 
                                    位置5概率最高 → "糖"

结束指针预测: [0.01, 0.01, 0.01, 0.02, 0.01, 0.05, 0.04, 0.92]
                                                          ↑
                                                位置7概率最高 → "病"

抽取结果: text[5:8] = "糖尿病" ✅
```

**优势**：
- ✅ 支持嵌套实体（一个实体包含另一个）
- ✅ 支持重叠实体（共享部分文字）
- ✅ 更灵活、更准确

---

## 🏗️ 5. 项目架构

### 完整数据流

```
1. 原始文本 
      ↓
2. Tokenizer 分词（文字 → 数字）
      ↓
3. 添加 Prompt
      ↓
4. 格式化: [CLS] Prompt [SEP] Text [SEP]
      ↓
5. ERNIE 编码器（获得向量表示）
      ↓
6. 向量序列: [h0, h1, h2, ..., hn]  (每个hi是768维)
      ↓
7. 起始指针网络 + 结束指针网络
      ↓
8. 起始概率 + 结束概率
      ↓
9. 提取实体 Span
```

### 核心模块文件

```
uie_pytorch/
├── model.py              ← UIE 模型定义
│   └── class UIE          • 起始指针网络: linear_start
│                          • 结束指针网络: linear_end
│
├── ernie.py              ← ERNIE 编码器（基于 BERT）
│                          • 将文本转为向量表示
│
├── tokenizer.py          ← 分词器
│                          • 文本 → Token IDs
│
├── utils.py              ← 工具函数
│   ├── IEDataset          • 数据加载
│   ├── SpanEvaluator      • 评估指标（P/R/F1）
│   └── convert_example    • 数据格式转换
│
├── finetune.py           ← 训练脚本
│                          • 定义训练循环
│                          • 保存最佳模型
│
└── uie_predictor.py      ← 推理预测器
                           • 多阶段推理
                           • Schema 解析
```

### UIE 模型代码结构

```python
class UIE(ErniePreTrainedModel):
    def __init__(self, config):
        # 1. ERNIE 编码器
        self.encoder = ErnieModel(config)
        
        # 2. 起始指针网络
        self.linear_start = nn.Linear(768, 1)
        
        # 3. 结束指针网络
        self.linear_end = nn.Linear(768, 1)
        
        # 4. 激活函数
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, input_ids, ...):
        # 编码
        sequence_output = self.encoder(input_ids)
        
        # 预测起始位置
        start_logits = self.linear_start(sequence_output)
        start_prob = self.sigmoid(start_logits)
        
        # 预测结束位置
        end_logits = self.linear_end(sequence_output)
        end_prob = self.sigmoid(end_logits)
        
        return start_prob, end_prob
```

---

## 📁 6. 项目文件导航

### 数据文件

```
ai-medical/
├── data/
│   └── annotated_data/
│       └── CMeIE-V2.jsonl        ← 14,338条医疗数据（主数据集）
│
└── debug_data/                   ← 小数据集（快速测试用）
    ├── train.jsonl
    └── dev.jsonl
```

### 预训练模型

```
ai-medical/
└── pretrained/
    └── uie_base_pytorch/         ← 预训练的 UIE 模型
        ├── pytorch_model.bin     ← 模型权重
        ├── config.json           ← 配置文件
        └── vocab.txt             ← 词表
```

---

## 💡 7. 知识自测

### 必答题（确保理解）

**Q1**: UIE 相比传统方法的最大优势是什么？

<details>
<summary>点击查看答案</summary>

**答案**：一个模型通过改变 Prompt 就能完成所有信息抽取任务（NER、RE、EE等），不需要为每个任务训练单独的模型。这大大降低了训练成本，提高了模型的通用性。
</details>

---

**Q2**: 什么是指针网络？它解决了什么问题？

<details>
<summary>点击查看答案</summary>

**答案**：指针网络通过预测实体的起始位置和结束位置来定位实体，而不是给每个字打标签。它解决了传统 BIO 标注方法难以处理嵌套实体和重叠实体的问题。
</details>

---

**Q3**: 如果我想抽取"高血压的治疗药物"，应该如何设计？

<details>
<summary>点击查看答案</summary>

**答案**：
- **方法1**：Prompt = "高血压的治疗药物"（直接描述关系）
- **方法2**：两阶段推理
  - 阶段1：Prompt = "疾病" → 抽取"高血压"
  - 阶段2：Prompt = "高血压的治疗药物" → 抽取药物

UIE 支持两种方式！
</details>

---

**Q4**: `linear_start` 和 `linear_end` 在模型中的作用是什么？

<details>
<summary>点击查看答案</summary>

**答案**：
- `linear_start`：起始指针网络，预测实体开始位置的概率分布
- `linear_end`：结束指针网络，预测实体结束位置的概率分布

两者结合，就能准确定位实体在文本中的位置。
</details>

---

## 🎯 8. 下一步学习计划

### Phase 2 预告：数据格式与处理

在 Phase 2 中，我们将学习：

1. **数据格式转换**
   - CMeIE 格式 → UIE 格式
   - 理解 `convert.py` 的转换逻辑

2. **数据处理流程**
   - 如何使用 `IEDataset` 加载数据
   - 如何构造训练样本
   - 标签生成机制

3. **Prompt 设计**
   - 实体抽取的 Prompt
   - 关系抽取的 Prompt
   - 如何设计有效的 Prompt

4. **动手实践**
   - 手动转换一条 CMeIE 数据
   - 编写数据统计脚本
   - 可视化标签分布

---

## ✅ Phase 1 完成检查清单

在进入 Phase 2 之前，请确保你理解：

- [ ] UIE 是什么？为什么它是"通用"的？
- [ ] Prompt 在 UIE 中的作用
- [ ] 指针网络的工作原理
- [ ] SPO 三元组的含义（Subject-Predicate-Object）
- [ ] 项目的核心文件和目录结构
- [ ] ERNIE 编码器的作用

**如果这些都清楚了，你就可以继续 Phase 2 了！** 🎉

---

## 📚 参考资源

- **UIE 论文**: [Universal Information Extraction](https://arxiv.org/abs/2203.12277)
- **ERNIE 论文**: [ERNIE: Enhanced Representation through kNowledge IntEgration](https://arxiv.org/abs/1904.09223)
- **项目学习指南**: `学习指南.md`（项目根目录）

---

> 💪 **恭喜完成 Phase 1！** 你已经掌握了 UIE 的核心思想和项目架构。
> 
> 准备好了就告诉我，我们开始 Phase 2：数据格式与处理！
