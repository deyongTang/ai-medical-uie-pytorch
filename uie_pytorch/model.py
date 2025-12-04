# Copyright (c) 2022 Heiheiyoyo. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
from dataclasses import dataclass
from transformers import PretrainedConfig
from transformers.utils import ModelOutput
from typing import Optional, Tuple

from uie_pytorch.ernie import ErnieModel, ErniePreTrainedModel
from uie_pytorch.ernie_m import ErnieMModel, ErnieMPreTrainedModel


@dataclass
class UIEModelOutput(ModelOutput):
    """
    UIE 模型的统一输出结构。

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*):
            训练时的总损失（起始 + 结束两个 BCE loss 的平均）。
        start_prob (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
            每个 token 作为“实体起始位置”的概率。
        end_prob (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
            每个 token 作为“实体结束位置”的概率。
        hidden_states (`tuple(torch.FloatTensor)`, *optional*):
            编码器每一层的 hidden states，只有在 `output_hidden_states=True` 时才会返回。
        attentions (`tuple(torch.FloatTensor)`, *optional*):
            注意力矩阵，只有在 `output_attentions=True` 时才会返回。
    """
    loss: Optional[torch.FloatTensor] = None
    start_prob: torch.FloatTensor = None
    end_prob: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class UIE(ErniePreTrainedModel):
    """
    UIE 模型（中文版），底层编码器使用 ERNIE。

    结构概要：
    - encoder: ErnieModel，将输入序列编码成 hidden states
    - linear_start: 线性层，预测每个 token 为“起始”的 logits
    - linear_end: 线性层，预测每个 token 为“结束”的 logits
    - sigmoid: 将 logits 映射到 0~1 概率空间
    """

    def __init__(self, config: PretrainedConfig):
        super(UIE, self).__init__(config)
        # ERNIE 编码器，负责上下文建模
        self.encoder = ErnieModel(config)
        self.config = config
        hidden_size = self.config.hidden_size

        # 起始 / 结束指针网络，本质是两个线性层
        self.linear_start = nn.Linear(hidden_size, 1)
        self.linear_end = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

        # if hasattr(config, 'use_task_id') and config.use_task_id:
        #     # Add task type embedding to BERT
        #     task_type_embeddings = nn.Embedding(
        #         config.task_type_vocab_size, config.hidden_size)
        #     self.encoder.embeddings.task_type_embeddings = task_type_embeddings

        #     def hook(module, input, output):
        #         input = input[0]
        #         return output+task_type_embeddings(torch.zeros(input.size(), dtype=torch.int64, device=input.device))
        #     self.encoder.embeddings.word_embeddings.register_forward_hook(hook)

        # 调用父类的初始化逻辑（权重初始化等）
        self.post_init()

    def forward(self, input_ids: Optional[torch.Tensor] = None,
                token_type_ids: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                head_mask: Optional[torch.Tensor] = None,
                inputs_embeds: Optional[torch.Tensor] = None,
                start_positions: Optional[torch.Tensor] = None,
                end_positions: Optional[torch.Tensor] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                return_dict: Optional[bool] = None
                ):
        """
        前向计算。

        Args:
        input_ids (`torch.LongTensor` of shape `({0})`):
            Indices of input sequence tokens in the vocabulary.
            Indices can be obtained using [`BertTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.
            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.FloatTensor` of shape `({0})`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
            [What are attention masks?](../glossary#attention-mask)
        token_type_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0,
            1]`:
            - 0 corresponds to a *sentence A* token,
            - 1 corresponds to a *sentence B* token.
            [What are token type IDs?](../glossary#token-type-ids)
        position_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.max_position_embeddings - 1]`.
            [What are position IDs?](../glossary#position-ids)
        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        inputs_embeds (`torch.FloatTensor` of shape `({0}, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        start_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        end_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        # 如果调用时没有显式指定，就使用 config 中的默认设置
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # 1. 编码器前向传播，得到每个 token 的上下文表示
        outputs = self.encoder(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )
        sequence_output = outputs[0]  # [batch_size, seq_len, hidden_size]

        # 2. 指针网络：把 hidden states 映射到“是否为起始/结束”的概率
        start_logits = self.linear_start(sequence_output)
        start_logits = torch.squeeze(start_logits, -1)
        start_prob = self.sigmoid(start_logits)
        end_logits = self.linear_end(sequence_output)
        end_logits = torch.squeeze(end_logits, -1)
        end_prob = self.sigmoid(end_logits)

        # 3. 如果提供了标签，则计算 BCE 损失
        total_loss = None
        if start_positions is not None and end_positions is not None:
            loss_fct = nn.BCELoss()
            start_loss = loss_fct(start_prob, start_positions)
            end_loss = loss_fct(end_prob, end_positions)
            total_loss = (start_loss + end_loss) / 2.0

        if not return_dict:
            output = (start_prob, end_prob) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return UIEModelOutput(
            loss=total_loss,
            start_prob=start_prob,
            end_prob=end_prob,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class UIEM(ErnieMPreTrainedModel):
    """
    UIE 多语言版本，底层编码器使用 ERNIE-M。

    结构与 UIE 完全一致，只是 encoder 换成了 ErnieMModel，
    可以支持中英等多语言场景。
    """

    def __init__(self, config: PretrainedConfig):
        super(UIEM, self).__init__(config)
        # ERNIE-M 编码器
        self.encoder = ErnieMModel(config)
        self.config = config
        hidden_size = self.config.hidden_size

        # 与 UIE 相同的指针头
        self.linear_start = nn.Linear(hidden_size, 1)
        self.linear_end = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

        self.post_init()

    def forward(self, input_ids: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                head_mask: Optional[torch.Tensor] = None,
                start_positions: Optional[torch.Tensor] = None,
                end_positions: Optional[torch.Tensor] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                return_dict: Optional[bool] = None
                ):
        """
        前向计算，接口形式与 UIE 保持一致（但不使用 token_type_ids）。

        Args:
        input_ids (`torch.LongTensor` of shape `({0})`):
            Indices of input sequence tokens in the vocabulary.
            Indices can be obtained using [`BertTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.
            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.FloatTensor` of shape `({0})`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
            [What are attention masks?](../glossary#attention-mask)
        position_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.max_position_embeddings - 1]`.
            [What are position IDs?](../glossary#position-ids)
        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        inputs_embeds (`torch.FloatTensor` of shape `({0}, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        start_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        end_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        # 同样，根据配置决定是否返回字典形式
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # 1. 前向编码
        outputs = self.encoder(
            input_ids=input_ids,
            position_ids=position_ids,
            # attention_mask=attention_mask,
            # head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )
        sequence_output = outputs[0]

        # 2. 指针网络
        start_logits = self.linear_start(sequence_output)
        start_logits = torch.squeeze(start_logits, -1)
        start_prob = self.sigmoid(start_logits)
        end_logits = self.linear_end(sequence_output)
        end_logits = torch.squeeze(end_logits, -1)
        end_prob = self.sigmoid(end_logits)

        # 3. 计算损失（如果提供标签）
        total_loss = None
        if start_positions is not None and end_positions is not None:
            loss_fct = nn.BCELoss()
            start_loss = loss_fct(start_prob, start_positions)
            end_loss = loss_fct(end_prob, end_positions)
            total_loss = (start_loss + end_loss) / 2.0

        if not return_dict:
            output = (start_prob, end_prob) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return UIEModelOutput(
            loss=total_loss,
            start_prob=start_prob,
            end_prob=end_prob,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
