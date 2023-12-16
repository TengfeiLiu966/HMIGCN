import os
import tqdm
import torch.nn as nn
import copy
import torch
import math
import torch.nn.functional as F
import numpy as np
from datasets.bert_processors.abstract_processor import BertProcessor, InputExample
def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class PositionwiseFeedForward(nn.Module):
     "Implements FFN equation."

     def __init__(self, d_model, d_ff, dropout=0.1):
         super(PositionwiseFeedForward, self).__init__()
         self.w_1 = nn.Linear(d_model, d_ff)
         self.w_2 = nn.Linear(d_ff, d_model)
         self.dropout = nn.Dropout(dropout)

     def forward(self, x):
         return self.w_2(self.dropout(F.relu(self.w_1(x))))

def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    # query,key,value:torch.Size([30, 8, 10, 64])
    # decoder mask:torch.Size([30, 1, 9, 9])
    d_k = query.size(-1)
    key_ = key.transpose(-2, -1)  # torch.Size([30, 8, 64, 10])
    # torch.Size([30, 8, 10, 10])
    scores = torch.matmul(query, key_) / math.sqrt(d_k)
    if mask is not None:
        # decoder scores:torch.Size([30, 8, 9, 9]),
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        #Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h  # 48=768//16
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        # query,key,value:torch.Size([2, 10, 768])
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)    #2
        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linears, (query, key, value))]  # query,key,value:torch.Size([30, 8, 10, 64])
        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask,
                                          dropout=self.dropout)
         # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(
                  nbatches, -1, self.h * self.d_k)
        ret = self.linears[-1](x)  # torch.Size([2, 10, 768])
        return (ret,self.attn)
#layer normalization [(cite)](https://arxiv.org/abs/1607.06450). do on
class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
class customizedModule(nn.Module):
    def __init__(self):
        super(customizedModule, self).__init__()

    # linear transformation (w/ initialization) + activation + dropout
    def customizedLinear(self, in_dim, out_dim, activation=None, dropout=False):
        cl = nn.Sequential(nn.Linear(in_dim, out_dim))
        nn.init.xavier_uniform(cl[0].weight)
        nn.init.constant(cl[0].bias, 0)

        if activation is not None:
            cl.add_module(str(len(cl)), activation)
        if dropout:
            cl.add_module(str(len(cl)), nn.Dropout(p=self.args.dropout))

        return cl
# That is, the output of each sub-layer is $\mathrm{LayerNorm}(x + \mathrm{Sublayer}(x))$, where $\mathrm{Sublayer}(x)$ is the function implemented by the sub-layer itself.  We apply dropout [(cite)](http://jmlr.org/papers/v15/srivastava14a.html) to the output of each sub-layer, before it is added to the sub-layer input and normalized.
# To facilitate these residual connections, all sub-layers in the model, as well as the embedding layers, produce outputs of dimension $d_{\text{model}}=512$.
class SublayerConnection(customizedModule):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)
        self.init_mBloSA()

    def init_mBloSA(self):
        self.g_W1 = self.customizedLinear(768, 768)
        self.g_W2 = self.customizedLinear(768, 768)
        self.g_b = nn.Parameter(torch.zeros(768))

        self.g_W1[0].bias.requires_grad = False
        self.g_W2[0].bias.requires_grad = False

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        ret = x + self.dropout(sublayer(self.norm(x)))
        # G = F.sigmoid(self.g_W1(x) + self.g_W2(self.dropout(sublayer(self.norm(x)))) + self.g_b)
        # # (batch, m, word_dim)
        # ret = G * x + (1 - G) * self.dropout(sublayer(self.norm(x)))
        return ret
class SublayerConnection(customizedModule):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)
        self.init_mBloSA()

    def init_mBloSA(self):
        self.g_W1 = self.customizedLinear(768, 768)
        self.g_W2 = self.customizedLinear(768, 768)
        self.g_b = nn.Parameter(torch.zeros(768))

        self.g_W1[0].bias.requires_grad = False
        self.g_W2[0].bias.requires_grad = False

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        # ret = x + self.dropout(sublayer(self.norm(x)))
        ret = x + self.dropout(sublayer(self.norm(x))[0])
        return ret,sublayer(self.norm(x))[1]

class SublayerConnection1(customizedModule):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection1, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)
        self.init_mBloSA()

    def init_mBloSA(self):
        self.g_W1 = self.customizedLinear(768, 768)
        self.g_W2 = self.customizedLinear(768, 768)
        self.g_b = nn.Parameter(torch.zeros(768))

        self.g_W1[0].bias.requires_grad = False
        self.g_W2[0].bias.requires_grad = False

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        # ret = x + self.dropout(sublayer(self.norm(x)))
        ret = x + self.dropout(sublayer(self.norm(x)))
        return ret
# Each layer has two sub-layers. The first is a multi-head self-attention mechanism, and the second is a simple, position-wise fully connected feed-forward network.
class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn      #多头注意力机制
        self.feed_forward = feed_forward    #前向神经网络
        # self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.sublayer = SublayerConnection(size, dropout)
        self.sublayer1 = SublayerConnection1(size, dropout)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x, attention_score = self.sublayer(x, lambda x: self.self_attn(x, x, x, mask))
        # torch.Size([30, 10, 512])
        ret = self.sublayer1(x, self.feed_forward)
        return ret,attention_score

def top_k_graph(scores, g1,g2,g3, h, k):
    num_nodes = g1.shape[1]                                       #一共几个节点
    values, idx = torch.topk(scores, max(2, k))                   #最大值得索引以及值
    new_h = []
    for i in range(4):
        new_h.append(h[i, idx[i, :], :].unsqueeze(0))
    new_h = torch.cat([new_h[0], new_h[1],new_h[2], new_h[3]], 0)                   #组合的新特征
    values = torch.unsqueeze(values, -1)                         #新增一个维度   2*512*738
    new_h = torch.mul(new_h, values)         #                   #新的特征       2*512*1     2*512*768
    #下面是对权重的选择
    g_sentence = []
    g_section = []
    g_mask = []
    for i in range(4):
        g11 = g1[i,idx[i, :],:]
        g11 = g11[:, idx[i,:]]
        g_section.append(g11.unsqueeze(0))
        g22 = g2[i,idx[i,:],:]
        g22 = g22[:, idx[i,:]]
        g_sentence.append(g22.unsqueeze(0))
        g33 = g3[i, idx[i, :], :]
        g_mask.append(g33.unsqueeze(0))
    return torch.cat([g_section[0],g_section[1],g_section[2],g_section[3]],0),torch.cat([g_sentence[0],g_sentence[1],g_sentence[2],g_sentence[3]],0),torch.cat([g_mask[0],g_mask[1],g_mask[2],g_mask[3]],0), new_h

def top_k_graph1(scores, g1, h, k):
    num_nodes = h.shape[1]                                       #一共几个节点
    values, idx = torch.topk(scores, max(2, k))                   #最大值得索引以及值
    new_h = []
    for i in range(4):
        new_h.append(h[i, idx[i, :], :].unsqueeze(0))
    new_h = torch.cat([new_h[0], new_h[1],new_h[2], new_h[3]], 0)                   #组合的新特征
    values = torch.unsqueeze(values, -1)                         #新增一个维度   2*512*738
    new_h = torch.mul(new_h, values)         #                   #新的特征       2*512*1     2*512*768
    g_mask_sen = []
    for i in range(4):
        g11 = g1[i, idx[i, :], :]
        g_mask_sen.append(g11.unsqueeze(0))
    return  torch.cat([g_mask_sen[0],g_mask_sen[1],g_mask_sen[2],g_mask_sen[3]],0),new_h

def norm_g(g):
    degrees = torch.sum(g, 1)
    g = g / degrees
    return g

class Pool(nn.Module):
    def __init__(self, k, in_dim, p):
        super(Pool, self).__init__()
        self.k = k
        self.sigmoid = nn.Sigmoid()
        self.drop = nn.Dropout(p=p) if p > 0 else nn.Identity()

    def forward(self, g1,g2,g3,h, section_feature):
        Z = self.drop(h)
        weights = torch.matmul(h, section_feature.transpose(1, 2)).squeeze(2)
        scores = self.sigmoid(weights)
        return top_k_graph(scores, g1,g2,g3,h, self.k)

class Pool1(nn.Module):
    def __init__(self, k, in_dim, p):
        super(Pool1, self).__init__()
        self.k = k
        self.sigmoid = nn.Sigmoid()
        self.drop = nn.Dropout(p=p) if p > 0 else nn.Identity()

    def forward(self,g1, h, section_feature):
        Z = self.drop(h)
        weights = torch.matmul(h, section_feature.transpose(1, 2)).squeeze(2)
        scores = self.sigmoid(weights)
        return top_k_graph1(scores,g1, h, self.k)

class Rea(nn.Module):
    def __init__(self,device):
        super(Rea, self).__init__()

        self.f1 = nn.Linear(768, 768)

        self.f2 = nn.Linear(768, 768)
        self.f3 = nn.Linear(768, 768)
        self.f4 = nn.Linear(768, 768)
        self.dropout1 = nn.Dropout(0.1)

    def forward(self, x,section_mask_full,sentence_mask_full):                        #所有的mask都为  4*1016*1016
        att = torch.matmul(self.f1(x), self.f2(x).transpose(1, 2)) / 28               #这是随机初始化的原始权重4*1016*1016
        #生成一个对角mask,然后softmax处理
        dia_att = F.softmax(att.masked_fill(section_mask_full == 0, -1e9), dim=2)     #通过段落mask，mask掉padding的部分
        #先验知识
        prior_att = F.softmax((2*section_mask_full + 4*sentence_mask_full).masked_fill(section_mask_full == 0, -1e9), dim=2)
        #知识的融合
        fusion_att =  F.softmax((dia_att * prior_att).masked_fill(section_mask_full == 0, -1e9), dim=2)
        graph_r = self.dropout1(fusion_att)
        g_x1 = self.f3(torch.matmul(graph_r, x))
        #把权重矩阵一分为二
        no_self_att = F.softmax(att.masked_fill(section_mask_full == 1, -1e9), dim=2)  # 通过段落mask，mask掉padding的部分
        graph_r1 = self.dropout1(no_self_att)
        g_x2 = self.f4(torch.matmul(graph_r1, g_x1))
        g_x = g_x1  + g_x2
        return g_x

class Rea_sec(nn.Module):
    def __init__(self):
        super(Rea_sec, self).__init__()

        self.f1 = nn.Linear(768, 768)
        self.f2 = nn.Linear(768, 768)
        self.f3 = nn.Linear(768, 768)

        self.dropout1 = nn.Dropout(0.1)

    def forward(self, x):                        #所有的mask都为  4*1016*1016
        att = torch.matmul(self.f1(x), self.f2(x).transpose(1, 2)) / 28               #这是随机初始化的原始权重4*1016*1016
        #生成一个对角mask,然后softmax处理
        att = F.softmax(att, dim=2)     #通过段落mask，mask掉padding的部分
        graph_r = self.dropout1(att)
        g_x = self.f3(torch.matmul(graph_r, x))
        return g_x

class Rea_sentence(nn.Module):
    def __init__(self,pooling_node,device):
        super(Rea_sentence, self).__init__()

        self.f1 = nn.Linear(768, 768)
        self.f2 = nn.Linear(768, 768)
        self.f3 = nn.Linear(768, 768)
        self.pooling_node = pooling_node
        self.b = torch.zeros(4,16,pooling_node,pooling_node).to(device)
        self.dropout1 = nn.Dropout(0.1)

    def forward(self, x, section_mask_full,attention_score,threhold):
        attention_score = torch.where(attention_score < threhold, self.b, attention_score)
        final_mask = section_mask_full.unsqueeze(0).unsqueeze(0).expand(4, 16, self.pooling_node,self.pooling_node) + attention_score
        fusion_att = F.softmax(final_mask.masked_fill(final_mask == 0, -1e9), dim=3)
        x = torch.matmul(fusion_att, x.view(4, self.pooling_node, 16, 48).transpose(1, 2))
        g_x = self.f3(x.transpose(1, 2).contiguous().view(4, -1, 768))
        return g_x

class DecoupledGraphPooling(customizedModule):
    def __init__(self,pooling_node1,pooling_node2,feature_dim,device,threhold,dropout = None):
        super(DecoupledGraphPooling, self).__init__()
        self.Rea = Rea(device)
        self.poolings = Pool(pooling_node1, feature_dim, dropout)
        self.poolings1 = Pool1(pooling_node2, feature_dim, dropout)
        self.init_mBloSA()
        self.pooling_node2 = pooling_node2
        self.Rea_sec = Rea_sec()
        self.device = device
        self.threhold = threhold
        self.Rea_sentence = Rea_sentence(pooling_node2,device)
        self.MultiHeadedAttention = MultiHeadedAttention(16, 768)
        self.PositionwiseFeedForward = PositionwiseFeedForward(768, 3072)
        self.Encoder = EncoderLayer(768, self.MultiHeadedAttention,self.PositionwiseFeedForward, 0.1)

    def init_mBloSA(self):
        self.g_W1 = self.customizedLinear(768, 768)
        self.g_W2 = self.customizedLinear(768, 768)
        self.g_b = nn.Parameter(torch.zeros(768))
        self.f_W2 = self.customizedLinear(768 * 2, 768)
        self.f_W3 = self.customizedLinear(768 * 2, 768)
        self.f_W4 = self.customizedLinear(768 * 2, 768)
        self.g_W1[0].bias.requires_grad = False
        self.g_W2[0].bias.requires_grad = False

    def forward(self, tran_mask, train_mask_sen, sentence_feature, token_feature, section_feature, section_mask,sentence_mask):
        # 先进行池化
        section_mask, sentence_mask, tran_mask, new_word = self.poolings(section_mask, sentence_mask,tran_mask, token_feature,torch.max(section_feature, dim=1)[0].unsqueeze(1))  # 2*14*768
        train_mask_sen, new_sentence = self.poolings1(train_mask_sen, sentence_feature,torch.max(section_feature, dim=1)[0].unsqueeze(1))  # 2*14*768
        # 然后进行单词特征进行学习
        DGPN_output = self.Rea(new_word, section_mask, sentence_mask)
        # 然后进行特征级别的注意力机制---门控单元
        G = F.sigmoid(self.g_W1(new_word) + self.g_W2(DGPN_output) + self.g_b)
        final_word_feature = G * new_word + (1 - G) * DGPN_output
        # 再进行段落的特征学习
        final_section_feature = self.Rea_sec(section_feature)
        # 最后进行句子特征的学习
        global_output, attention_scre = self.Encoder(new_sentence, mask=None)  # 4*120*50   全局交互
        section_mask_full = torch.triu(torch.ones(self.pooling_node2, self.pooling_node2), diagonal=-3).to(
            self.device) - torch.triu(torch.ones(self.pooling_node2, self.pooling_node2), diagonal=4).to(self.device)
        local_output = self.Rea_sentence(new_sentence, section_mask_full, attention_scre,self.threhold)  # 4*100*100

        G1 = F.sigmoid(self.f_W2(torch.cat([global_output, local_output], dim=2)))
        final_sentence_feature = G1 * global_output + (1 - G1) * local_output

        final_word_feature = self.f_W3(torch.cat([final_word_feature, torch.matmul(tran_mask, final_section_feature)], dim=2))
        final_sentence_feature = self.f_W4(torch.cat([final_sentence_feature, torch.matmul(train_mask_sen, final_section_feature)], dim=2))

        return section_mask, sentence_mask, tran_mask, train_mask_sen, final_word_feature, final_sentence_feature, final_section_feature

class AAPDProcessor(BertProcessor):
    NAME = 'AAPD'
    NUM_CLASSES = 100
    IS_MULTILABEL = True

    def get_train_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir,'Wiki 10-31k', 'train100.tsv')), 'train')

    def get_test_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, 'Wiki 10-31k', 'test100.tsv')), 'test')

    def _create_examples(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):

            guid = "%s-%s" % (set_type, i)
            text_a = line[1]
            label = line[0]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples