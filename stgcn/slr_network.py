import pdb
import copy
import utils
import torch
import types
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from modules.criterions import SeqKD
from modules import BiLSTMLayer, TemporalConv
from stgcn.graphcueagcnsim import MultiCueGraphNet

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class NormLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(NormLinear, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_dim, out_dim))
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))

    def forward(self, x):
        outputs = torch.matmul(x, F.normalize(self.weight, dim=0))
        return outputs

class ConfCrossEntropyLoss(nn.Module):
    def __init__(self, classes, dim=-1):
        super(ConfCrossEntropyLoss, self).__init__()
        self.cls = classes
        self.dim = dim

    def forward(self, pred, seq_pred, label, x_len, label_len):
        with torch.no_grad():
            target = self.pseudo_gen(seq_pred.log_softmax(-1), label, x_len, label_len)
            # pred (time, batch, cls)
            # target (time, batch)
            conf = torch.gather(seq_pred.softmax(-1), 2, target.unsqueeze(2)).squeeze(2)
            # conf (time, batch)
            smoonthing = (1 - conf) / (self.cls - 1)
            # smoonthing (time, batch)
            true_dist = smoonthing.data.unsqueeze(2).repeat(1, 1, self.cls)
            # true_dist (time, batch, cls)
            true_dist.scatter_(2, target.data.unsqueeze(2), conf.data.unsqueeze(2))
        # sum over 'cls' and then mean over 'time' and 'batch'
            mask = torch.ones(pred.shape)
            for i, len in enumerate(x_len):
                mask[len:, i, :].zero_()

        return torch.mean(torch.sum(-true_dist * pred.log_softmax(-1) * mask, dim=self.dim))

    def gloss_max_probability_path(self, data, label):
        ninf = float("-inf")
        N, M = data.shape
        pb_max = np.zeros((len(label), N), dtype=float)
        R, C = pb_max.shape
        path = list()

        for i in range(R):
            path.append([(-1, -1) for _ in range(C)])   

        for i in range(R):
            pb_max[i, :] = data[:, label[i]]
            pb_max[i, 0:i] = ninf
            pb_max[i, C-R+i+1:C] = ninf

        dp = np.zeros((R, C), dtype=float)

        dp[0, 0] = pb_max[0, 0]
        dp[1:R, 0] = ninf
        for i in range(1, C):
            dp[0, i] = dp[0, i-1] + pb_max[0, i]
            path[0][i] = (0, i-1)
        
        for i in range(1, R):
            for j in range(1, C):
                if dp[i-1, j-1] >  dp[i, j-1]:
                    dp[i, j] = dp[i-1, j-1] + pb_max[i, j]
                    path[i][j] = (i-1, j-1)
                else:
                    dp[i, j] = dp[i, j-1] + pb_max[i, j]
                    path[i][j] = (i, j-1) 
        
        outcome = list()

        outcome.append(label[R-1])
        back = path[R-1][C-1]
        
        while back != (-1, -1):
            outcome.append(label[back[0]])
            back = path[back[0]][back[1]]
        outcome.reverse()
        return outcome
    
    def pseudo_gen(self, pred, label, x_len, label_len):
        mx = pred.size(0)
        pseudo = [self.gloss_max_probability_path(pred[:x_len[i] ,i ,:], label[i, :label_len[i]]) + ([0] * (mx - x_len[i])) for i in range(pred.size(1))]
        return torch.tensor(pseudo, dtype = torch.int64).permute(1, 0)
            

class BaseModel(nn.Module):
    def __init__(
            self, num_classes, c2d_type, conv_type, use_bn=False,
            hidden_size=1024, gloss_dict=None, loss_weights=None,
            weight_norm=True, share_classifier=True, stgcn_args = None
    ):
        super(BaseModel, self).__init__()
        self.decoder = None
        self.loss = dict()
        self.criterion_init(num_classes)
        self.num_classes = num_classes
        self.loss_weights = loss_weights
        self.conv2d = getattr(models, c2d_type)(pretrained=True)
        self.conv2d.fc = Identity()
        self.conv1d = TemporalConv(input_size=512,
                                   hidden_size=hidden_size,
                                   conv_type=conv_type,
                                   use_bn=use_bn,
                                   num_classes=num_classes)
        self.decoder = utils.Decode(gloss_dict, num_classes, 'beam')
        self.temporal_model = BiLSTMLayer(rnn_type='LSTM', input_size=hidden_size, hidden_size=hidden_size,
                                          num_layers=2, bidirectional=True)
        if weight_norm:
            self.classifier = NormLinear(hidden_size, self.num_classes)
            self.conv1d.fc = NormLinear(hidden_size, self.num_classes)
        else:
            self.classifier = nn.Linear(hidden_size, self.num_classes)
            self.classifier = nn.Linear(hidden_size, self.num_classes)
        if share_classifier:
            self.conv1d.fc = self.classifier
        self.register_backward_hook(self.backward_hook)

    def backward_hook(self, module, grad_input, grad_output):
        for g in grad_input:
            g[g != g] = 0

    def masked_bn(self, inputs, len_x):
        def pad(tensor, length):
            return torch.cat([tensor, tensor.new(length - tensor.size(0), *tensor.size()[1:]).zero_()])

        x = torch.cat([inputs[len_x[0] * idx:len_x[0] * idx + lgt] for idx, lgt in enumerate(len_x)])
        x = self.conv2d(x)
        x = torch.cat([pad(x[sum(len_x[:idx]):sum(len_x[:idx + 1])], len_x[0])
                       for idx, lgt in enumerate(len_x)])
        return x

    def forward(self, x, ske, len_x = None, label=None, label_lgt=None):
        batch, temp, channel, height, width = x.shape
        inputs = x.reshape(batch * temp, channel, height, width)
        framewise = self.masked_bn(inputs, len_x)
        framewise = framewise.reshape(batch, temp, -1).transpose(1, 2)

        conv1d_outputs = self.conv1d(framewise, len_x)

        x = conv1d_outputs['visual_feat']
        lgt = conv1d_outputs['feat_len']
        # x (time, batch, C)
        tm_outputs = self.temporal_model(x, lgt)
        outputs = self.classifier(tm_outputs['predictions'])
        pred = None if self.training \
            else self.decoder.decode(outputs, lgt, batch_first=False, probs=False)
        conv_pred = None if self.training \
            else self.decoder.decode(conv1d_outputs['conv_logits'], lgt, batch_first=False, probs=False)

        return {
            "framewise_features": None,
            "visual_features": x,
            "feat_len": lgt,
            "conv_logits": conv1d_outputs['conv_logits'],
            "sequence_logits": outputs,
            "conv_sents": conv_pred,
            "recognized_sents": pred,
        }

    def criterion_calculation(self, ret_dict, label, label_lgt):
        loss = 0
        for k, weight in self.loss_weights.items():
            if k == 'ConvCTC':
                loss += weight * self.loss['CTCLoss'](ret_dict["conv_logits"].log_softmax(-1),
                                                      label.cpu().int(), ret_dict["feat_len"].cpu().int(),
                                                      label_lgt.cpu().int()).mean()
            elif k == 'SeqCTC':
                loss += weight * self.loss['CTCLoss'](ret_dict["sequence_logits"].log_softmax(-1),
                                                      label.cpu().int(), ret_dict["feat_len"].cpu().int(),
                                                      label_lgt.cpu().int()).mean()
            elif k == 'Dist':
                loss += weight * self.loss['distillation'](ret_dict["conv_logits"],
                                                           ret_dict["sequence_logits"].detach(),
                                                           use_blank=False)
            elif k == 'CCELoss':
                loss += weight * self.loss['CCELoss'](ret_dict["conv_logits"].cpu(),
                                                      ret_dict["sequence_logits"].cpu(), 
                                                      label.cpu().int(), 
                                                      ret_dict["feat_len"].cpu().int(),
                                                      label_lgt.cpu().int())
        return loss

    def criterion_init(self, num_classes):
        self.loss['CTCLoss'] = torch.nn.CTCLoss(reduction='none', zero_infinity=False)
        self.loss['distillation'] = SeqKD(T=8)
        self.loss['CCELoss'] = ConfCrossEntropyLoss(num_classes)
        return self.loss

class SLRModel(nn.Module):
    def __init__(
            self, num_classes, c2d_type, conv_type, use_bn=False,
            hidden_size=1024, gloss_dict=None, loss_weights=None,
            weight_norm=True, share_classifier=True, stgcn_args = None
    ):
        super(SLRModel, self).__init__()
        self.decoder = None
        self.loss = dict()
        self.criterion_init(num_classes)
        self.num_classes = num_classes
        self.loss_weights = loss_weights
        self.stgcn = MultiCueGraphNet(**stgcn_args)
        
        self.decoder = utils.Decode(gloss_dict, num_classes, 'beam')
        self.temporal_model = BiLSTMLayer(rnn_type='LSTM', input_size=hidden_size, hidden_size=hidden_size,
                                          num_layers=2, bidirectional=True)
        if weight_norm:
            self.classifier = NormLinear(hidden_size, self.num_classes)
            self.conv_classifier = NormLinear(hidden_size, self.num_classes)
        else:
            self.classifier = nn.Linear(hidden_size, self.num_classes)
            self.conv_classifier = nn.Linear(hidden_size, self.num_classes)

        if share_classifier:
            self.conv_classifier = self.classifier
            
        self.register_backward_hook(self.backward_hook)

    def backward_hook(self, module, grad_input, grad_output):
        for g in grad_input:
            g[g != g] = 0

    def masked_bn(self, inputs, len_x):
        def pad(tensor, length):
            return torch.cat([tensor, tensor.new(length - tensor.size(0), *tensor.size()[1:]).zero_()])

        x = torch.cat([inputs[len_x[0] * idx:len_x[0] * idx + lgt] for idx, lgt in enumerate(len_x)])
        x = self.conv2d(x)
        x = torch.cat([pad(x[sum(len_x[:idx]):sum(len_x[:idx + 1])], len_x[0])
                       for idx, lgt in enumerate(len_x)])
        return x

    def forward(self, x, ske, len_x = None, label=None, label_lgt=None):
        # x (batch, time, C, H, W)
        x, lgt = self.stgcn(x, ske, len_x)
        # x (batch, time, C)
        x = x.permute(1, 0, 2)
        # x (time, batch, C)
        lgt = lgt.cpu()
        conv_output = self.conv_classifier(x)
        tm_outputs = self.temporal_model(x, lgt)
        outputs = self.classifier(tm_outputs['predictions'])
        pred = None if self.training \
            else self.decoder.decode(outputs, lgt, batch_first=False, probs=False)
        conv_pred = None if self.training \
            else self.decoder.decode(conv_output, lgt, batch_first=False, probs=False)

        return {
            "framewise_features": None,
            "visual_features": x,
            "feat_len": lgt,
            "conv_logits": conv_output,
            "sequence_logits": outputs,
            "conv_sents": conv_pred,
            "recognized_sents": pred,
        }

    def criterion_calculation(self, ret_dict, label, label_lgt, CCE_activate):
        loss = 0
        for k, weight in self.loss_weights.items():
            if k == 'ConvCTC':
                loss += weight * self.loss['CTCLoss'](ret_dict["conv_logits"].log_softmax(-1),
                                                      label.cpu().int(), ret_dict["feat_len"].cpu().int(),
                                                      label_lgt.cpu().int()).mean()
            elif k == 'SeqCTC':
                loss += weight * self.loss['CTCLoss'](ret_dict["sequence_logits"].log_softmax(-1),
                                                      label.cpu().int(), ret_dict["feat_len"].cpu().int(),
                                                      label_lgt.cpu().int()).mean()
            elif k == 'Dist':
                loss += weight * self.loss['distillation'](ret_dict["conv_logits"],
                                                           ret_dict["sequence_logits"].detach(),
                                                           use_blank=False)
            elif k == 'CCELoss' and CCE_activate:
                loss += weight * self.loss['CCELoss'](ret_dict["conv_logits"].cpu(),
                                                      ret_dict["sequence_logits"].cpu(), 
                                                      label.cpu().int(), 
                                                      ret_dict["feat_len"].cpu(),
                                                      label_lgt.cpu().int())
        return loss

    def criterion_init(self, num_classes):
        self.loss['CTCLoss'] = torch.nn.CTCLoss(reduction='none', zero_infinity=False)
        self.loss['distillation'] = SeqKD(T=8)
        self.loss['CCELoss'] = ConfCrossEntropyLoss(num_classes)
        return self.loss
