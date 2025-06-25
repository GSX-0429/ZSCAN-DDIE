import math
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn.init as init
from torch import nn
from torch.nn import Parameter
from .builder import build_structural, build_textual, CLASSIFIERS
from models.Loss.FocalLoss import FocalLoss
from models.Loss.LabelSmoothing import LabelSmoothingCrossEntropy


@CLASSIFIERS.register_module()
class classifier(nn.Module):
    def __init__(self, structuralmodule, textualmodule, temperature=0.9,seen_labels=None, device=None, separability=False, UniformityLoss=True,
                 train_textualinput=None, val_zsl_textualinput=None, val_gzsl_textualinput=None,uni_lambda=0.7, use_w=False,instanse_uni_loss=True,
                 class_uni_loss=True,use_attention=True,attributlabel=None,zsl_labels=None,gzsl_labels=None,enable_dynamic_loss=True):
        super(classifier, self).__init__()
        self.Structuralmodule = build_structural(structuralmodule)
        self.Textualmodule = build_textual(textualmodule)
        self.temperature = temperature
        self.device = device
        self.seen_labels = seen_labels
        self.zsl_labels = zsl_labels
        self.gzsl_labels = gzsl_labels
        self.train_textualinput = train_textualinput
        self.val_zsl_textualinput = val_zsl_textualinput
        self.val_gzsl_textualinput = val_gzsl_textualinput
        self.uni_lambda = uni_lambda
        self.instanse_uni_loss = instanse_uni_loss
        self.class_uni_loss = class_uni_loss
        self.use_attention = use_attention
        self.attributlabel = attributlabel
        self.enable_dynamic_loss = enable_dynamic_loss

        # global
        self.UniformityLoss = UniformityLoss
        self.separability = separability

        # local
        if self.use_attention:
            self.W_q = Parameter(torch.Tensor(300, 256))
            self.W_k = Parameter(torch.Tensor(self.Textualmodule.output_dim, self.Textualmodule.output_dim))
            self.W_v = Parameter(torch.Tensor(self.Textualmodule.output_dim, self.Textualmodule.output_dim))
            init.xavier_normal_(self.W_q)
            init.xavier_normal_(self.W_k)
            init.xavier_normal_(self.W_v)

        self.loss = self.set_loss_function()


        # CAN
        self.hidden_dim = 256
        self.num_heads = 2
        self.head_size = self.hidden_dim // self.num_heads
        self.query1 = nn.Linear(300, self.hidden_dim, bias=False)
        self.key1 = nn.Linear(300, self.hidden_dim, bias=False)
        self.value1 = nn.Linear(300, self.hidden_dim, bias=False)
        self.query2 = nn.Linear(256, self.hidden_dim, bias=False)
        self.key2 = nn.Linear(256, self.hidden_dim, bias=False)
        self.value2 = nn.Linear(256, self.hidden_dim, bias=False)

    def set_loss_function(self, loss_type='CrossEntropy'):
        if self.enable_dynamic_loss:
            if loss_type == 'CrossEntropy':
                self.loss = nn.CrossEntropyLoss()
            elif loss_type == 'Focal':
                self.loss = FocalLoss()
            elif loss_type == 'LabelSmoothing':
                self.loss = LabelSmoothingCrossEntropy()
            else:
                raise ValueError("Unsupported Loss function type")
        else:
            self.loss = nn.CrossEntropyLoss()
        return self.loss
    # 计算两种嵌入的均匀性损失
    def UniformityLoss_twoemb(self, drugembeddings, proto_embedding, device):
        center_proto_embedding = torch.mean(proto_embedding, dim=1)

        normalize_proto_embedding = F.normalize(proto_embedding - center_proto_embedding.unsqueeze(1))
        cos_dist_matrix = torch.matmul(normalize_proto_embedding,
                                       normalize_proto_embedding.transpose(1, 2))
        unit_matrix = torch.eye(cos_dist_matrix.shape[1]).to(device)
        cos_dist_matrix = cos_dist_matrix - unit_matrix
        loss1 = torch.mean(torch.mean(torch.max(cos_dist_matrix, 2).values))

        n_d = F.normalize(drugembeddings - center_proto_embedding)
        d_cos_dist_matrix = torch.matmul(n_d, n_d.transpose(1, 0))
        d_unit_matrix = torch.eye(d_cos_dist_matrix.shape[0]).to(device)
        d_cos_dist_matrix = d_cos_dist_matrix - d_unit_matrix
        loss2 = torch.mean(torch.max(d_cos_dist_matrix, 1).values)

        if self.class_uni_loss == True and self.instanse_uni_loss == False:
            return loss1
        elif self.class_uni_loss == False and self.instanse_uni_loss == True:
            return loss2
        elif self.class_uni_loss == True and self.instanse_uni_loss == True:
            return loss1 + loss2
        # 计算单个嵌入的均匀性损失
    def single_UniformityLoss_twoemb(self, drugembeddings, proto_embedding, device):
        center_proto_embedding = torch.mean(proto_embedding, dim=0)
        normalize_proto_embedding = F.normalize(proto_embedding - center_proto_embedding.unsqueeze(0))
        cos_dist_matrix = torch.matmul(normalize_proto_embedding,
                                       normalize_proto_embedding.transpose(0, 1))

        unit_matrix = torch.eye(cos_dist_matrix.shape[0]).to(device)
        cos_dist_matrix = cos_dist_matrix - unit_matrix

        loss1 = torch.mean(torch.max(cos_dist_matrix, 1).values)

        n_d = F.normalize(drugembeddings - center_proto_embedding.unsqueeze(0))

        d_cos_dist_matrix = torch.matmul(n_d, n_d.transpose(1, 0))
        d_unit_matrix = torch.eye(d_cos_dist_matrix.shape[0]).to(device)
        d_cos_dist_matrix = d_cos_dist_matrix - d_unit_matrix
        loss2 = torch.mean(torch.max(d_cos_dist_matrix, 1).values)

        return loss1 + loss2

    def _heads(self, x, n_heads, n_ch):
        s = list(x.size())[:-1] + [n_heads, n_ch]
        return x.view(*s)

    def CAN(self, struct_output, drugemb, semanticemb, emb_ids):
        d_k = drugemb.size(-1)
        query1 = self._heads(
            self.query1(drugemb).expand((semanticemb.shape[0], drugemb.shape[0], drugemb.shape[1], 256)).transpose(0, 1),
            self.num_heads, self.head_size)
        key1 = self._heads(
            self.key1(drugemb).expand((semanticemb.shape[0], drugemb.shape[0], drugemb.shape[1], 256)).transpose(0,1),
            self.num_heads, self.head_size)
        query2 = self._heads(self.query2(semanticemb).expand(
            (drugemb.shape[0], semanticemb.shape[0], semanticemb.shape[1], semanticemb.shape[2])),
            self.num_heads, self.head_size)
        key2 = self._heads(self.key2(semanticemb).expand(
            (drugemb.shape[0], semanticemb.shape[0], semanticemb.shape[1], semanticemb.shape[2])),
            self.num_heads, self.head_size)

        a1 = F.softmax(torch.matmul(query1.transpose(2, 3), key1.transpose(2, 3).transpose(3, 4)) / math.sqrt(d_k),
                       dim=-1)
        a2 = F.softmax(torch.matmul(query2.transpose(2, 3), key2.transpose(2, 3).transpose(3, 4)) / math.sqrt(d_k),
                       dim=-1)
        a = torch.einsum('bcdll, bcdkk->bclk', a1, a2)

        logits11 = torch.einsum('bclhd, bckhd->bclkh', query1, key1)
        logits12 = torch.einsum('bclhd, bckhd->bclkh', query1, key2)
        logits21 = torch.einsum('bclhd, bckhd->bclkh', query2, key1)
        logits22 = torch.einsum('bclhd, bckhd->bclkh', query2, key2)

        value1 = self._heads(
            self.value1(drugemb).expand((semanticemb.shape[0], drugemb.shape[0], drugemb.shape[1], 256)).transpose(
                0, 1),
            self.num_heads, self.head_size)
        value2 = self._heads(
            self.value2(semanticemb).expand(
                (drugemb.shape[0], semanticemb.shape[0], semanticemb.shape[1], semanticemb.shape[2])),
            self.num_heads, self.head_size)

        drug_atten1 = torch.matmul(a1, value1.transpose(2, 3)).transpose(2, 3)
        drug_atten2 = torch.matmul(a2, value2.transpose(2, 3)).transpose(2, 3)
        drug_atten1 = torch.mean(drug_atten1, dim=2).flatten(-2)
        drug_atten2 = torch.mean(drug_atten2, dim=2).flatten(-2)
        drug_atten_ = (drug_atten1 + drug_atten2) / 2


        drug_atten1 = torch.matmul(struct_output.unsqueeze(dim=1), drug_atten1.transpose(1, 2)).squeeze(1)
        drug_atten2 = torch.matmul(struct_output.unsqueeze(dim=1), drug_atten2.transpose(1, 2)).squeeze(1)


        output1 = (torch.einsum('bclkh, bc->bcl', logits11, drug_atten1) +
                   torch.einsum('bclkh, bc->bcl', logits12, drug_atten2)) / 2
        output2 = (torch.einsum('bclkh, bc->bcl', logits21, drug_atten1) +
                   torch.einsum('bclkh, bc->bcl', logits22, drug_atten2)) / 2

        logits = torch.einsum('bcl, bck->bc', output1, output2)

        loss = self.loss(logits, torch.tensor(emb_ids, dtype=torch.long).to(self.device))
        if self.UniformityLoss:
            loss_uni = self.UniformityLoss_twoemb(struct_output, drug_atten_, self.device)
            loss = loss + self.uni_lambda * loss_uni

        return logits, loss, a, drug_atten_
    # 计算局部损失
    def Local(self, struct_output, drugemb, semanticemb, emb_ids):
        if self.use_attention:
            d_k = drugemb.size(-1)  # 获取药物嵌入的维度300
            Q = torch.matmul(drugemb, self.W_q).expand((semanticemb.shape[0], drugemb.shape[0],
                                                        drugemb.shape[1], 256)).transpose(0,1)

            K = torch.matmul(semanticemb, self.W_k).expand((drugemb.shape[0], semanticemb.shape[0],
                                                            semanticemb.shape[1],
                                                            semanticemb.shape[2]))

            V = torch.matmul(semanticemb, self.W_v).expand((drugemb.shape[0], semanticemb.shape[0],
                                                            semanticemb.shape[1],
                                                            semanticemb.shape[2]))
            a = F.softmax(torch.matmul(Q, K.transpose(2, 3)) / math.sqrt(d_k), dim=-1)
            drug_atten = torch.matmul(a, V)
            drug_atten_ = torch.mean(drug_atten, dim=2)

            logits = (torch.matmul(struct_output.unsqueeze(dim=1), drug_atten_.transpose(1, 2)).squeeze(1)
                      / self.temperature)

            loss = self.loss(logits, torch.tensor(emb_ids, dtype=torch.long).to(self.device))
            if self.UniformityLoss:
                loss_uni = self.UniformityLoss_twoemb(struct_output, drug_atten_, self.device)
                loss = loss + self.uni_lambda * loss_uni

        else:
            if len(semanticemb.shape) == 3:
                drug_atten_ = torch.mean(semanticemb, 1)
            else:
                drug_atten_ = semanticemb

            logits = torch.matmul(struct_output, drug_atten_.transpose(0, 1)) / self.temperature

            loss = self.loss(logits, torch.tensor(emb_ids, dtype=torch.long).to(self.device))
            if self.UniformityLoss:
                loss_uni = self.single_UniformityLoss_twoemb(struct_output, drug_atten_, self.device)
                loss = loss + self.uni_lambda * loss_uni

        return logits, loss, a, drug_atten_
    def forward(self, input):
        struct_output, sub_structure, d1_att, d2_att ,drug1_batch, drug2_batch, drug1_mols, drug2_mols = self.Structuralmodule(input[0])

        if input[2][0] == "train":
            textual_output_all, emb_ids, bertemb = self.Textualmodule(input[0],
                                                                 self.train_textualinput)

        elif input[2][0] == "zsl":
            textual_output_all, emb_ids, _ = self.Textualmodule(input[0], self.val_zsl_textualinput)
        elif input[2][0] == "gzsl":
            textual_output_all, emb_ids, _ = self.Textualmodule(input[0], self.val_gzsl_textualinput)

        # logits, loss_g, cross_att, proto = self.Local(struct_output, sub_structure, textual_output_all, emb_ids)  # 计算局部损失
        logits, loss_g, cross_att, proto = self.CAN(struct_output, sub_structure, textual_output_all, emb_ids)

        if input[2][0] == "train" :
            loss_g = loss_g

        if len(textual_output_all.shape) == 3:
            eventemb_mean = torch.mean(textual_output_all, 1)
        else:
            eventemb_mean = textual_output_all

        prototype = eventemb_mean[emb_ids, :]

        return loss_g, logits, emb_ids, struct_output, prototype, cross_att, sub_structure, d1_att, d2_att, drug1_batch, drug2_batch, drug1_mols, drug2_mols

