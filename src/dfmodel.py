"""
@Time   :   2021-01-12 15:08:01
@File   :   models.py
@Author :   Abtion
@Email  :   abtion{at}outlook.com
"""
import operator
import os
from collections import OrderedDict

import torch, copy, math, json
from torch import nn
import pytorch_lightning as pl
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from transformers import BertConfig, AutoConfig, AutoModel
from transformers import BertModel, BertPreTrainedModel
from transformers.models.bert.modeling_bert import BertEmbeddings, BertEncoder, BertPooler, BertOnlyMLMHead, BertModel, BertForMaskedLM
from transformers.modeling_utils import ModuleUtilsMixin

from .utils import compute_corrector_prf, compute_sentence_level_prf, score_f_sent, score_f
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['KaiTi']
plt.rcParams['axes.unicode_minus'] = False

ACT2FN = {
    "relu": nn.ReLU(),
    "gelu": nn.GELU(),
    "tanh": nn.Tanh(),
    "sigmoid": nn.Sigmoid(),
}


class CustomRNN(nn.GRU):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, hidden_staes, length):
        package = nn.utils.rnn.pack_padded_sequence(hidden_staes, length, batch_first=True, enforce_sorted=False)
        packout, _ = super(CustomRNN, self).forward(package)
        out, _ = nn.utils.rnn.pad_packed_sequence(packout, batch_first=True)
        return out
#
class DetectionNetwork(nn.Module, ModuleUtilsMixin):
    def __init__(self, config, args):
        super().__init__()
        self.config = config
        self.args = args
        self.gru = CustomRNN(
            input_size=self.config.hidden_size,
            hidden_size=self.args.hidden_size // 2,
            # hidden_size=self.config.hidden_size // 2,
            num_layers=2,
            batch_first=True,
            dropout=self.config.hidden_dropout_prob,
            bidirectional=True)

        self.decoder = nn.Linear(self.args.hidden_size, 1)
        self.bias = nn.Parameter(torch.zeros(1))
        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias
        # self.softmax = nn.Softmax(-1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, hidden_states, lens):

        lens = torch.IntTensor(lens)
        out_l = self.gru(hidden_states, lens)
        prob = self.sigmoid(self.decoder(out_l))
        return prob

class DetectionNetwork_Trans(nn.Module):

    def __init__(self, config, device):
        """
        :param position_embeddings: bert的position_embeddings，本质是一个nn.Embedding
        :param transformer: BERT的前两层transformer_block，其是一个ModuleList对象
        """
        super(DetectionNetwork_Trans, self).__init__()
        # self.transformer_blocks = transformer_blocks
        self.transformer_blocks = nn.ModuleList([copy.deepcopy(BertLayer(config)) for _ in range(2)])
        # 定义最后的预测层，预测哪个token是错误的
        self.dense_layer = nn.Sequential(
            nn.Linear(config.hidden_size, 1),
            nn.Sigmoid()
        )
        self._device = device

    def forward(self, embed):
        # sequence_length = embed.size(1)
        # position_embeddings = self.position_embeddings(torch.LongTensor(range(sequence_length)).to(self._device))
        # 融合work_embedding和position_embedding
        x = embed
        # 将x一层一层的使用transformer encoder进行向后传递
        for transformer_layer in self.transformer_blocks:
            x = transformer_layer(x)[0]

        # 最终返回Detection Network输出的hidden states和预测结果
        hidden_states = x
        return self.dense_layer(hidden_states)


class TeacherModel(torch.nn.Module, ModuleUtilsMixin):
    def __init__(self, config, tokenizer, args, device, add_pooling_layer=False):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.corrector = BertModel(config, add_pooling_layer=add_pooling_layer)
        self.corrector = BertForMaskedLM.from_pretrained(args.bert_checkpoint)
        self.embeddings = self.corrector.embeddings
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.mask_token_id = self.tokenizer.mask_token_id
        self.pooler = BertPooler(self.config)
        self.cls = BertOnlyMLMHead(self.config)
        # self.dropout = nn.Dropout(0.1)
        # self.cls = self.corrector.cls

        self._device = device

    def forward(self, texts, prob, embed=None, cor_labels=None, attention_mask=None, residual_connection=False):
        if cor_labels is not None:
            text_labels = self.tokenizer(cor_labels, padding=True, return_tensors='pt')['input_ids']
            text_labels = text_labels.to(self._device)
            # torch的cross entropy loss 会忽略-100的label
            text_labels[text_labels == 0] = -100
        else:
            text_labels = None
        encoded_texts = self.tokenizer(texts, padding=True, return_tensors='pt')
        encoded_texts.to(self._device)
        if embed is None:
            embed = self.embeddings(input_ids=encoded_texts['input_ids'],
                                    token_type_ids=encoded_texts['token_type_ids'],
                                    position_ids=encoded_texts['position_ids'])
        # 此处较原文有一定改动，做此改动意在完整保留type_ids及position_ids的embedding。
        mask_embed = self.embeddings(torch.ones_like(prob.squeeze(-1)).long() * self.mask_token_id).detach()
        # # 此处为原文实现
        # # mask_embed = self.embeddings(torch.tensor([[self.mask_token_id]], device=self._device)).detach()
        cor_embed = prob * mask_embed + (1 - prob) * embed
        # return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        input_shape = encoded_texts['input_ids'].size()
        device = encoded_texts['input_ids'].device

        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask,
                                                                                 input_shape, device)
        head_mask = self.get_head_mask(None, self.config.num_hidden_layers)

        encoder_outputs = self.corrector.encoder(
            cor_embed,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            return_dict=False,
        )
        sequence_outputs = encoder_outputs[0]

        sequence_out = self.dropout(sequence_outputs + embed) if residual_connection else sequence_outputs
        prediction_scores = self.cls(sequence_out)

        return prediction_scores




class BertCorrectionModel(torch.nn.Module, ModuleUtilsMixin):
    def __init__(self, config, tokenizer, args, device, add_pooling_layer=False):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.corrector = BertModel(config, add_pooling_layer=add_pooling_layer)
        # self.corrector = BertModel.from_pretrained(args.bert_path)
        self.embeddings = self.corrector.embeddings
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.mask_token_id = self.tokenizer.mask_token_id
        self.pooler = BertPooler(self.config)
        # self.pooler = None
        self.cls = BertOnlyMLMHead(self.config)
        # self.cls = self.corrector.cls
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)

        self._device = device

    def get_prompt(self, batch_size, prefix_tokens):
        prefix_tokens = prefix_tokens.unsqueeze(0).expand(batch_size, -1).to(self._device)

        prompts = self.prefix_encoder(prefix_tokens)
        # prompts = self.prefix_encoder(prompt_tokens)
        return prompts

    def forward(self, texts, prob, embed=None, cor_labels=None, teacher_scores=None, attention_mask=None, residual_connection=False):
        if cor_labels is not None:
            text_labels = self.tokenizer(cor_labels, padding=True, return_tensors='pt')['input_ids']
            text_labels = text_labels.to(self._device)
            # torch的cross entropy loss 会忽略-100的label
            text_labels[text_labels == 0] = -100
        else:
            text_labels = None

        encoded_texts = self.tokenizer(texts, padding=True, return_tensors='pt')
        encoded_texts.to(self._device)

        if embed is None:
            embed = self.embeddings(input_ids=encoded_texts['input_ids'],
                                    token_type_ids=encoded_texts['token_type_ids'],
                                    position_ids=encoded_texts['position_ids'])

        input_shape = encoded_texts['input_ids'].size()
        device = encoded_texts['input_ids'].device

        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask,
                                                                                 input_shape, device)
        head_mask = self.get_head_mask(None, self.config.num_hidden_layers)

        encoder_outputs = self.corrector.encoder(
            embed,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            return_dict=False,
        )
        sequence_outputs = encoder_outputs[0]
        pooled_output = self.pooler(sequence_outputs) if self.pooler is not None else None
        sequence_out = self.dropout(sequence_outputs + embed) if residual_connection else sequence_outputs
        
        prediction_scores = self.cls(sequence_out)

        if teacher_scores is not None:
            kl_loss = self.compute_kl_loss(prediction_scores, teacher_scores, pad_mask=attention_mask)
        else:
            kl_loss = 0
            
        out = (prob, prediction_scores, sequence_out.squeeze(0), kl_loss)

        # Masked language modeling softmax layer
        if text_labels is not None:

            loss_fct = nn.CrossEntropyLoss(reduction='sum')  # -100 index = padding token
            cor_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), text_labels.view(-1))
            out = (cor_loss,) + out
        return out

    def log_softmax_t(self, X, T):
        X = X / T
        X = nn.LogSoftmax(-1)(X)

        return X

    def load_from_transformers_state_dict(self, gen_fp):
        state_dict = OrderedDict()
        gen_state_dict = torch.load(gen_fp)
        # name1 = None
        for k, v in gen_state_dict.items():
            name = copy.deepcopy(k)
            if name.startswith('bert'):
                name = f'corrector.{name[5:]}'
            if name.startswith('encoder'):
                name = f'corrector.{name}'
            # if name.startswith('embedding'):
            #     name = f'corrector.{name}'
            if 'gamma' in name:
                name = name.replace('gamma', 'weight')
            if 'beta' in name:
                name = name.replace('beta', 'bias')


            state_dict[name] = v
            # if isinstance(name1, str):
            #     state_dict[name1] = v
            #     name1 = None

        self.load_state_dict(state_dict, strict=False)
        # print(0)
#         for k in state_dict.keys():
#             print(k)

    def compute_kl_loss(self, p, q, reduction='sum', pad_mask=None):

        p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='none')

        # pad_mask is for seq-level tasks
        if pad_mask is not None:
            pad_mask = pad_mask.bool()
            p_loss.masked_fill_(~(pad_mask.unsqueeze(-1)), 0.)

        # You can choose whether to use function "sum" and "mean" depending on your task
        if reduction == 'sum':
            p_loss = p_loss.sum()
        if reduction == 'mean':
            p_loss = p_loss.mean()

        # loss = self.prob * p_loss + (1 - self.prob) * q_loss
        return p_loss

class BaseCorrectorTrainingModel(pl.LightningModule):
    """
    用于CSC的BaseModel, 定义了训练及预测步骤
    """

    def __init__(self, arguments, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.validation_step_outputs = []
        self.args = arguments
        self.w = arguments.loss_weight
        # self.Regularizer_Loss = nn.CrossEntropyLoss(reduction='sum')
        self.min_loss = float('inf')
        self.best_f1 = 0.0
        self.best_epoch = None

    def training_step(self, batch, batch_idx):
        ori_text, cor_text, det_labels = batch
        outputs = self.forward(ori_text, cor_text, det_labels)
        loss = self.w * (outputs[1]) + (1 - self.w) * outputs[0]
        
        return loss

    def validation_step(self, batch, batch_idx):
        ori_text, cor_text, det_labels = batch
        outputs = self.forward(ori_text, cor_text, det_labels)
        loss = self.w * (outputs[1]) + (1 - self.w) * outputs[0]
        
        cor_y_hat = torch.argmax((outputs[4]), dim=-1)
        encoded_x = self.tokenizer(cor_text, max_length=128, padding=True, return_tensors='pt')
        encoded_x.to(self._device)
        cor_y = encoded_x['input_ids']
        cor_y_hat *= encoded_x['attention_mask']

        results = []
        det_acc_labels = []
        cor_acc_labels = []
        for src, tgt, predict, det_label in zip(ori_text, cor_y, cor_y_hat, det_labels):
            _src = self.tokenizer(src, max_length=128, add_special_tokens=False)['input_ids']
            _tgt = tgt[1:len(_src) + 1].cpu().numpy().tolist()
            _predict = predict[1:len(_src) + 1].cpu().numpy().tolist()
            cor_acc_labels.append(1 if operator.eq(_tgt, _predict) else 0)
            results.append((_src, _tgt, _predict,))
        self.validation_step_outputs.append([loss.cpu().item(), det_acc_labels, cor_acc_labels, results])
        return loss.cpu().item(), det_acc_labels, cor_acc_labels, results

    def on_validation_epoch_end(self) -> None:
        print('Valid End.')
        det_acc_labels = []
        cor_acc_labels = []
        results = []
        for out in self.validation_step_outputs:
            results += out[3]
        loss = np.mean([out[0] for out in self.validation_step_outputs])
        print(f'loss: {loss}')
        print('Char Level:')
        score_f_sent(results)
        _, _, _, _ = compute_sentence_level_prf(results, self.tokenizer)
        # if self.args.mode == "train" and (len(outputs) > 6) and (loss < self.min_loss):
        if self.args.mode == "train" and (f1 > self.best_f1) and len(self.validation_step_outputs) > 6:
            self.min_loss = loss
            self.best_f1 = f1
            torch.save(self.state_dict(),
                       os.path.join(self.args.model_save_path, f'{self.__class__.__name__}_model.bin'))
            print('model saved.')
            self.best_epoch = self.current_epoch
        print(self.best_epoch, self.best_f1)
        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def on_test_epoch_end(self) -> None:
        print('Test.')
        self.on_validation_epoch_end()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()), lr=self.args.lr)
        # scheduler = LambdaLR(optimizer,
        #                      lr_lambda=lambda step: min((step + 1) ** -0.5,
        #                                                 (step + 1) * self.args.warmup_epochs ** (-1.5)),
        #                      last_epoch=-1)
        return [optimizer]#, [scheduler]

    def predict(self, texts):
        inputs = self.tokenizer(texts, padding=True, return_tensors='pt')
        inputs.to(self._device)
        with torch.no_grad():
            outputs = self.forward(texts)
            y_hat = torch.argmax(outputs[1], dim=-1)
            expand_text_lens = torch.sum(inputs['attention_mask'], dim=-1) - 1
        rst = []
        for t_len, _y_hat in zip(expand_text_lens, y_hat):
            rst.append(self.tokenizer.decode(_y_hat[1:t_len]).replace(' ', ''))
        return rst


class SoftMaskedBertModel(BaseCorrectorTrainingModel):
    def __init__(self, args, tokenizer):
        super().__init__(args)
        self.args = args
        self.config = BertConfig.from_pretrained(self.args.bert_checkpoint)
        # self.detector = DetectionNetwork(self.config, self.args)
        self.tokenizer = tokenizer
        self.embedding = BertEmbeddings(self.config)
        self.corrector = BertCorrectionModel(self.config, tokenizer, args, args.device)
        self.teacher = TeacherModel(self.config, tokenizer, args, args.device)
        for name, param in self.teacher.named_parameters():
            param.requires_grad = False
        self._device = args.device

    def forward(self, texts, cor_labels=None, det_labels=None):
        encoded_texts = self.tokenizer(texts, padding=True, max_length=128, return_tensors='pt')
        encoded_texts.to(self._device)
        
        embed = self.corrector.embeddings(input_ids=encoded_texts['input_ids'],
                                          token_type_ids=encoded_texts['token_type_ids'])
                                          
        # lens = [len(t) for t in encoded_texts['attention_mask'].sum(1)]
        
        prob = None
        # prob = self.detector(embed)
        if self.args.mode == 'train':
            teacher_prob = det_labels.unsqueeze(-1)
            teacher_scores = self.teacher(texts, teacher_prob, embed, cor_labels, encoded_texts['attention_mask'], residual_connection=True)
            cor_out = self.corrector(texts, prob, embed, cor_labels, teacher_scores, encoded_texts['attention_mask'], residual_connection=True)
        else:
            cor_out = self.corrector(texts, prob, embed, cor_labels, attention_mask=encoded_texts['attention_mask'], residual_connection=True)
        # prob = cor_out[-1]
        # 用于带有detection模块的CSC模型
        if det_labels is not None:
            # det_loss_fct = nn.BCELoss(reduction='sum')
            # # # pad部分不计算损失
            # active_loss = encoded_texts['attention_mask'].view(-1, prob.shape[1]) == 1
            # active_probs = prob.view(-1, prob.shape[1])[active_loss]
            # active_labels = det_labels[active_loss]
            # det_loss = det_loss_fct(active_probs, active_labels.float())
            det_loss = None
            outputs = (det_loss, cor_out[0], prob) + cor_out[1:]
        else:
            outputs = (prob,) + cor_out

        return outputs

    def load_from_transformers_state_dict(self, gen_fp):
        """
        从transformers加载预训练权重
        :param gen_fp:
        :return:
        """
        self.corrector.load_from_transformers_state_dict(gen_fp)
        self.teacher.load_from_transformers_state_dict(gen_fp)
