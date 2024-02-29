import torch
import torch.nn as nn
import torchvision
from copy import deepcopy
from stgcn.graph import Graph
#from utils.graph import Graph
import torch.nn.utils.rnn as rnn_utils

max_len = 300
head_grid = 16
hand_grid = 24
relation_grid = 32
channels = 64

class simpleCNN(nn.Module):
    def __init__(self):
        super(simpleCNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(64, 128, 5, 1, 2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(128, 256, 5, 1, 2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 512, 5, 1, 2),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)
        )
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.avgpool(x)
        return x

class FeatureGenNet(nn.Module):
    def __init__(self):
        super(FeatureGenNet, self).__init__()
        self.resnet = torchvision.models.resnet18(weights='DEFAULT')
        self.stem = nn.Sequential(*list(self.resnet.children())[:4])
        self.layer1 = self.resnet.layer1
        self.layer2 = self.resnet.layer2
        self.layer3 = self.resnet.layer3
        self.layer4 = self.resnet.layer4
        self.head_conv = simpleCNN()
        self.hand_conv = simpleCNN()
        self.avg_pool = self.resnet.avgpool
        self.roi_pooling = torch.nn.AdaptiveMaxPool2d((relation_grid, relation_grid))

    def forward(self, x, ctr, xlen):

        # x (frames, channel, H, W)
        # crt (frames, channel, H, W)
        x = self.stem(x)
        x = self.layer1(x)

        head_roi, lhand_roi, rhand_roi, h_lh_re_roi, h_rh_re_roi, rh_lh_re_roi = self.getProposal(x, ctr, xlen)

        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avg_pool(x)



        head_x = self.head_conv(head_roi)

        lhand_x = self.hand_conv(lhand_roi)

        rhand_x = self.hand_conv(rhand_roi)

        h_lh_re_x = self.layer2(h_lh_re_roi)
        h_lh_re_x = self.layer3(h_lh_re_x)
        h_lh_re_x = self.layer4(h_lh_re_x)
        h_lh_re_x = self.avg_pool(h_lh_re_x)

        h_rh_re_x = self.layer2(h_rh_re_roi)
        h_rh_re_x = self.layer3(h_rh_re_x)
        h_rh_re_x = self.layer4(h_rh_re_x)
        h_rh_re_x = self.avg_pool(h_rh_re_x)

        rh_lh_re_x = self.layer2(rh_lh_re_roi)
        rh_lh_re_x = self.layer3(rh_lh_re_x)
        rh_lh_re_x = self.layer4(rh_lh_re_x)
        rh_lh_re_x = self.avg_pool(rh_lh_re_x)

        # x (frames, C, 1, 1)
        result = torch.stack([x, head_x, lhand_x, rhand_x, h_lh_re_x, h_rh_re_x, rh_lh_re_x], dim=1)
        # result (frames ,7 , C, 1, 1)
        result = result.mean(dim=4).mean(dim=3)
        # result (frames ,7 , C)
        return result

    def linearinter(self, ctr, x_len):
        presum = [sum(x_len[: i]) for i in range(len(x_len))] + [sum(x_len)]
        cur = 0
        points = ctr.size(1)
        seq_len = ctr.size(0)
        for i in range(points):
            temp = ctr[:, i, :]
            for seq_index in range(seq_len):
                while(presum[cur + 1] <= seq_index): 
                    cur+=1
                if temp[seq_index, 2] == 0:
                    front = seq_index - 1                    
                    if front < presum[cur]:
                        front = presum[cur]
                    
                    end = seq_index
                    while(temp[end, 2]==0):
                        end = end + 1
                        if end > presum[cur + 1] - 1:
                            end = presum[cur + 1] - 1
                            break
                    
                    if front == presum[cur]:
                        for j in range(presum[cur], end):
                            temp[j, :] = temp[end, :]
                    elif end == presum[cur + 1] - 1:
                        for j in range(front + 1, presum[cur + 1]):
                            temp[j, :] = temp[front, :]
                    else:
                        step = (temp[end, :] - temp[front, :]) / (end - front)
                        for j in range(front + 1, end):
                            temp[j, :] = temp[front, :] + (j - front) * step
            ctr[:, i, :] = temp
        return ctr

    def getProposal(self, x, ctr, x_len):
        h = x.size(2)
        w = x.size(3)
        seq_len = ctr.size(0)
        ctr = self.linearinter(ctr, x_len)
        ctr[:, :, 0] = ctr[:, :, 0] * (w / 224)
        ctr[:, :, 1] = ctr[:, :, 1] * (h / 224)
        ctr = ctr.long()

        h_roi = torch.zeros((seq_len, 4), dtype=torch.long, requires_grad=False, device=ctr.device)
        rh_roi = torch.zeros((seq_len, 4), dtype=torch.long, requires_grad=False, device=ctr.device)
        lh_roi = torch.zeros((seq_len, 4), dtype=torch.long, requires_grad=False, device=ctr.device)
        re_roi = torch.zeros((seq_len, 4), dtype=torch.long, requires_grad=False, device=ctr.device)

        head_roi = torch.zeros((seq_len, channels, head_grid, head_grid), dtype=torch.float32, requires_grad=False, device=x.device)
        rhand_roi = torch.zeros((seq_len, channels, hand_grid, hand_grid), dtype=torch.float32, requires_grad=False, device=x.device)
        lhand_roi = torch.zeros((seq_len, channels, hand_grid, hand_grid), dtype=torch.float32, requires_grad=False, device=x.device)
        h_lh_re_roi = torch.zeros((seq_len, channels, relation_grid, relation_grid), dtype=torch.float32, requires_grad=False, device=x.device)
        h_rh_re_roi = torch.zeros((seq_len, channels, relation_grid, relation_grid), dtype=torch.float32, requires_grad=False, device=x.device)
        rh_lh_re_roi = torch.zeros((seq_len, channels, relation_grid, relation_grid), dtype=torch.float32, requires_grad=False, device=x.device)

        h_roi[:, 0].copy_(ctr[:, 0, 0] - head_grid // 2)
        h_roi[:, 1].copy_(ctr[:, 0, 0] + head_grid // 2)
        h_roi[:, 2].copy_(ctr[:, 0, 1] - head_grid // 2)
        h_roi[:, 3].copy_(ctr[:, 0, 1] + head_grid // 2)
        h_roi.copy_(torch.where(h_roi < 0, torch.full_like(h_roi, 0), h_roi))
        h_roi.copy_(torch.where(h_roi > h, torch.full_like(h_roi, h), h_roi))
        for i in range(seq_len):
            if h_roi[i, 0] == 0:
                head_roi[i, :, h_roi[i, 2] - (h_roi[i, 3] - head_grid): head_grid, 0: h_roi[i, 1] - h_roi[i, 0]].copy_(x[i, :, h_roi[i, 2]: h_roi[i, 3], h_roi[i, 0]: h_roi[i, 1]].detach())
            else:
                head_roi[i, :, h_roi[i, 2] - (h_roi[i, 3] - head_grid): head_grid, head_grid - (h_roi[i, 1] - h_roi[i, 0]): head_grid].copy_(x[i, :, h_roi[i, 2]: h_roi[i, 3], h_roi[i, 0]: h_roi[i, 1]].detach())
            

        rh_roi[:, 0].copy_(ctr[:, 1, 0] - hand_grid // 2)
        rh_roi[:, 1].copy_(ctr[:, 1, 0] + hand_grid // 2)
        rh_roi[:, 2].copy_(ctr[:, 1, 1] - hand_grid // 2)
        rh_roi[:, 3].copy_(ctr[:, 1, 1] + hand_grid // 2)
        rh_roi.copy_(torch.where(rh_roi < 0, torch.full_like(rh_roi, 0), rh_roi))
        rh_roi.copy_(torch.where(rh_roi > h, torch.full_like(rh_roi, h), rh_roi))

        for i in range(seq_len):
            if rh_roi[i, 2] == 0:
                rhand_roi[i, :, hand_grid - (rh_roi[i, 3] - rh_roi[i, 2]): hand_grid, rh_roi[i, 0] - (rh_roi[i, 1] - hand_grid): hand_grid].copy_(x[i, :, rh_roi[i, 2]: rh_roi[i, 3], rh_roi[i, 0]: rh_roi[i, 1]].detach())
            else:
                rhand_roi[i, :, 0: rh_roi[i, 3] - rh_roi[i, 2], rh_roi[i, 0] - (rh_roi[i, 1] - hand_grid): hand_grid].copy_(x[i, :, rh_roi[i, 2]: rh_roi[i, 3], rh_roi[i, 0]: rh_roi[i, 1]].detach())

        lh_roi[:, 0].copy_(ctr[:, 2, 0] - hand_grid // 2)
        lh_roi[:, 1].copy_(ctr[:, 2, 0] + hand_grid // 2)
        lh_roi[:, 2].copy_(ctr[:, 2, 1] - hand_grid // 2)
        lh_roi[:, 3].copy_(ctr[:, 2, 1] + hand_grid // 2)
       
        lh_roi.copy_(torch.where(lh_roi > h, torch.full_like(lh_roi, h), lh_roi))
        lh_roi.copy_(torch.where(lh_roi < 0, torch.full_like(lh_roi, 0), lh_roi))
    
        for i in range(seq_len):
            if lh_roi[i, 2] == 0:
                lhand_roi[i, :, hand_grid - (lh_roi[i, 3]- lh_roi[i, 2]): hand_grid, 0: lh_roi[i, 1] - lh_roi[i, 0]].copy_(x[i, :, lh_roi[i, 2]: lh_roi[i, 3], lh_roi[i, 0]: lh_roi[i, 1]].detach())
            else:
                lhand_roi[i, :, 0: lh_roi[i, 3] - lh_roi[i, 2], 0: lh_roi[i, 1] - lh_roi[i, 0]].copy_(x[i, :, lh_roi[i, 2]: lh_roi[i, 3], lh_roi[i, 0]: lh_roi[i, 1]].detach())

        re_roi[:, 0].copy_(torch.min(h_roi[:, 0], lh_roi[:, 0]))
        re_roi[:, 1].copy_(torch.max(h_roi[:, 1], lh_roi[:, 1]))
        re_roi[:, 2].copy_(torch.min(h_roi[:, 2], lh_roi[:, 2]))
        re_roi[:, 3].copy_(torch.max(h_roi[:, 3], lh_roi[:, 3]))
    
        for i in range(seq_len):
            h_lh_re_roi[i, :, :, :].copy_(self.roi_pooling(x[i, :, re_roi[i, 2]: re_roi[i, 3], re_roi[i, 0]: re_roi[i, 1]]).detach()) 

        re_roi[:, 0].copy_(torch.min(h_roi[:, 0], rh_roi[:, 0]))
        re_roi[:, 1].copy_(torch.max(h_roi[:, 1], rh_roi[:, 1]))
        re_roi[:, 2].copy_(torch.min(h_roi[:, 2], rh_roi[:, 2]))
        re_roi[:, 3].copy_(torch.max(h_roi[:, 3], rh_roi[:, 3]))

        for i in range(seq_len):
            h_rh_re_roi[i, :, :, :].copy_(self.roi_pooling(x[i, :, re_roi[i, 2]: re_roi[i, 3], re_roi[i, 0]: re_roi[i, 1]]).detach())

        re_roi[:, 0].copy_(torch.min(rh_roi[:, 0], lh_roi[:, 0]))
        re_roi[:, 1].copy_(torch.max(rh_roi[:, 1], lh_roi[:, 1]))
        re_roi[:, 2].copy_(torch.min(rh_roi[:, 2], lh_roi[:, 2]))
        re_roi[:, 3].copy_(torch.max(rh_roi[:, 3], lh_roi[:, 3]))

        for i in range(seq_len):
            rh_lh_re_roi[i, :, :, :].copy_(self.roi_pooling(x[i, :, re_roi[i, 2]: re_roi[i, 3], re_roi[i, 0]: re_roi[i, 1]]).detach())

        return head_roi, lhand_roi, rhand_roi, h_lh_re_roi, h_rh_re_roi, rh_lh_re_roi
        
        
class RGCNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, num_relation, bias=False, dropout=0):
        super().__init__()
        self.self_loop_weight = nn.Parameter(torch.Tensor(in_channels, out_channels))
        nn.init.xavier_uniform_(self.self_loop_weight, gain=nn.init.calculate_gain('relu'))

        self.rel_weight = nn.Parameter(torch.Tensor(num_relation, in_channels, out_channels))
        nn.init.xavier_uniform_(self.rel_weight, gain=nn.init.calculate_gain('relu'))

        if in_channels != out_channels:
            self.down = nn.Linear(in_channels, out_channels)
        else:
            self.down = lambda x: x

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
            nn.init.zeros_(self.bias)

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout, inplace=True)

    def forward(self, x, rel):
        # x (time, 7, C)
        # rel (1, num_relation, num_node, num_node) = (1,2,7,7)
        T = x.size(0)
        _, num_relation, num_node, num_node = rel.size()
        temp_rel = rel.expand(T, num_relation, num_node, num_node)
        # temp_rel (time, num_relation, num_node, num_node)
        self_loop_message = torch.einsum('bsi,ij->bsj', x, self.self_loop_weight)
        #self_loop_message = self.dropout(self_loop_message)
        neighbor_message = torch.einsum('brts,bsi,rij->btj', temp_rel, x, self.rel_weight)

        n_feat = self_loop_message + neighbor_message
        n_feat += self.down(x)
        #n_feat = n_feat + self.bias (time, 7, C)
        n_feat = self.relu(n_feat)

        return n_feat

class ST_RGCN(nn.Module):
    def __init__(self, in_channels, out_channels, num_relation, t_kernel_size, stride, dropout=0):
        super().__init__()
        self.t_kernel_size = t_kernel_size
        self.stride = stride
        padding = ((t_kernel_size - 1) // 2, 0)
        self.t_padding = padding[0]
        self.rgcn = RGCNLayer(in_channels, out_channels, num_relation, dropout=dropout)
        self.tgcn = nn.Conv2d(
                out_channels,
                out_channels,
                (t_kernel_size, 1),
                (stride, 1),
                padding)
        self.tgcn_bn = nn.BatchNorm1d(out_channels)
            

        if (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
            self.res_bn = lambda x: x

        else:
            self.residual = nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(stride, 1))
            self.res_bn = nn.BatchNorm1d(out_channels)

        self.relu = nn.ReLU(inplace=True)

    def tconv(self, inputs, len_x, new_len, conv_layer, bn_layer):
        def pad(tensor, length):
            return torch.cat([tensor, tensor.new(length - tensor.size(0), *tensor.size()[1:]).zero_()])
        # input (time, 7, C)
        mx_len = max(len_x)
        x = torch.stack([pad(inputs[sum(len_x[: i]): sum(len_x[: i + 1])], mx_len) for i in range(len(len_x))], axis = 0)
        # x (batch, time, 7, C)
        x = x.permute(0, 3, 1, 2)
        # x (batch, C, time, 7)
        x = conv_layer(x)
        x = x.permute(0, 2, 3, 1)
        # x (batch, time, 7, C)
        x = torch.cat([x[i,: new_len[i]] for i in range(len(new_len))], axis = 0)
        # x (time, 7, C)
        x = x.permute(0, 2, 1)
        # x (time, C, 7)
        x = bn_layer(x)
        x = x.permute(0, 2, 1)
        # x (time, 7, C)
        return x
        
    def forward(self, x, rel, xlen):
        # x (frames ,7 , C)
        new_len = torch.div(xlen + 2 * self.t_padding - self.t_kernel_size + self.stride, self.stride, rounding_mode='floor')
        res = self.tconv(x, xlen, new_len, self.residual, self.res_bn)

        x = self.rgcn(x, rel)
        # x (time, 7, C)
        x = self.tconv(x, xlen, new_len, self.tgcn, self.tgcn_bn)
        # x (time, 7, C)
        x = x + res
        x = self.relu(x)
        return x, new_len

class RGCNEmbedding(nn.Module):
    def __init__(self, num_relation, dim):
        super().__init__()
        self.node_embedding = nn.Embedding(num_relation, dim)

    def forward(self, x, node_types):
        # x (frames ,7 , C)
        # node_type (1, 7)

        N = x.size(0)
        _, V = node_types.size()
        temp_node_types = node_types.expand(N, V)
        # temp_node_types (time, 7)
        node_embeds = self.node_embedding(temp_node_types)
        x = x + node_embeds
        return x



class MultiCueGraphNet(nn.Module):
    def __init__(self, in_channels, out_channels, t_kernel_size, edge_importance_weighting, dropout=0, model='union'):
        super().__init__()
        self.feature_extracter = FeatureGenNet()
        self.graph = Graph()
        self.model = model
        rel = torch.tensor(self.graph.rel, dtype=torch.float32, requires_grad=False)
        # rel (num_relation, num_node, num_node) = (2,7,7)
        rel = rel.unsqueeze(0)
        # rel (1,2,7,7)
        self.register_buffer('rel', rel)
        node_rel = torch.tensor(self.graph.node_rel, dtype=torch.long, requires_grad=False)
        # node_rel (7)
        node_rel = node_rel.unsqueeze(0)
        # node_rel (1,7)
        self.register_buffer('node_rel', node_rel)
        self.data_bn = nn.BatchNorm1d(in_channels * rel.size(2))
        self.num_relation = rel.size(1)
        # num_relation = 2
        self.ST_RGCN_net = nn.ModuleList((
            ST_RGCN(in_channels, out_channels, self.num_relation, t_kernel_size, 1, dropout=dropout),
            ST_RGCN(out_channels, out_channels, self.num_relation, t_kernel_size, 2, dropout=dropout),
            ST_RGCN(out_channels, out_channels, self.num_relation, t_kernel_size, 1, dropout=dropout),
            ST_RGCN(out_channels, out_channels, self.num_relation, t_kernel_size, 2, dropout=dropout),
            ST_RGCN(out_channels, out_channels, self.num_relation, t_kernel_size, 1, dropout=dropout),
        ))

        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(self.rel.size()))
                for i in self.ST_RGCN_net
            ])
        else:
            self.edge_importance = [1] * len(self.ST_RGCN_net)

        self.embedding = RGCNEmbedding(self.num_relation, in_channels)
        self.linear = nn.Linear(in_channels, 1295)
        self.dropout = nn.Dropout(dropout, inplace=True)

    def forward(self, x, ctr, xlen):
        # x (batch, time, channel, H, W)
        # ctr (batch, time, num_object, num_relation) = (batch, time, 3, 2)
        x = torch.cat([x[i, :xlen[i]] for i in range(len(xlen))], axis = 0)
        ctr = torch.cat([ctr[i, :xlen[i]] for i in range(len(xlen))], axis = 0)
        # x (frames, channel, H, W)
        # ctr (frames, num_object, num_relation)
        x = self.feature_extracter(x, ctr, xlen)
        # x (frames ,7 , C)
        # 7: node_type [x, head_x, lhand_x, rhand_x, h_lh_re_x, h_rh_re_x, rh_lh_re_x]
        x = self.embedding(x, self.node_rel)
        # x (frames ,7 , C)
        T, V, C = x.size()
        x = x.reshape(T, V * C)        
        # x (time, 7*C)
        x = self.data_bn(x)   
        x = x.reshape(T, V, C)
        # x (frames ,7 , C)
        for gcn, importance in zip(self.ST_RGCN_net, self.edge_importance):
            x, xlen = gcn(x, self.rel * importance, xlen)
        # x (time, 7, C)
        x = x.mean(dim=1)
        # x (time, C)
        if self.model != 'union':
            x = self.dropout(x)
            x = self.linear(x)

        def pad(tensor, length):
            return torch.cat([tensor, tensor.new(length - tensor.size(0), *tensor.size()[1:]).zero_()])
        x = torch.stack([pad(x[sum(xlen[: i]): sum(xlen[: i + 1])], max(xlen)) for i in range(len(xlen))], axis = 0)
        # x (batch, time, C)
        return x, xlen

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, batch_first=True):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.output_size = output_size
        self.bidirection = True
        if num_layers == 1:
            self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, 
                                num_layers=num_layers, batch_first=batch_first, 
                                bidirectional=self.bidirection)
        else:
            self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, 
                                num_layers=num_layers, batch_first=batch_first, 
                                dropout=0.5, bidirectional=self.bidirection)
        self.in_size = 2 if self.bidirection else 1
        self.linear = nn.Linear(hidden_size * self.in_size, output_size)

    def forward(self, x, h, c):
        x, (h, c) = self.lstm(x, (h, c))
        x, out_len = rnn_utils.pad_packed_sequence(x, batch_first=self.batch_first)
        linear_out = torch.zeros(x.size(0), x.size(1), self.output_size)
        for i in range(x.size(0)):
            len_i = out_len[i]
            feat = x[i, :len_i, :]
            linear_out[i, : len_i, :] = self.linear(feat)
        return linear_out, out_len
    
    def init_hidden(self, batch_size):
        h0 = torch.zeros(self.num_layers * self.in_size, batch_size, self.hidden_size)
        c0 = torch.zeros(self.num_layers * self.in_size, batch_size, self.hidden_size)
        return h0, c0

class ConfCrossEntropyLoss(nn.Module):
    def __init__(self, classes, dim=-1):
        super(ConfCrossEntropyLoss, self).__init__()
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target, conf):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            # true_dist = pred.data.clone()
            smoonthing = (1 - conf) / (self.cls - 1)
            true_dist = smoonthing.data.unsqueeze(1).repeat(1, self.cls)
            true_dist.scatter_(1, target.data.unsqueeze(1), conf.data.unsqueeze(1))
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))

if __name__ == "__main__":
    
    x = torch.randn((8, 3, 224, 224), dtype=torch.float32)
    #maxp = torch.nn.AdaptiveMaxPool2d((32, 32))
    #x = x.unsqueeze(0)
    ctr_list = [[[60, 60], [30, 150], [220, 150]] for _ in range(8)]
    ctr = torch.tensor(ctr_list, dtype=torch.float32)
    net = MultiCueGraphNet(512, 3, True)
    lstm = LSTM(512, 512, 30, 2)
    h, c = lstm.init_hidden(1)    
    x = net(x, ctr)
    packed = rnn_utils.pack_sequence([x])
    e, f = lstm(packed, h, c)
    print(e.size())
    print(f.size())
    print(f)
    print(x.size())
    '''
    input1 = torch.randn((3, 3))
    net = mininet()
    re = net(input1)
    '''
