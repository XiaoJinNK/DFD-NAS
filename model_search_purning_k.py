import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from operations_CDC_theta import *
from torch.autograd import Variable
from genotypes_CDC_theta import PRIMITIVES
from genotypes_CDC_theta import Genotype


def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()

    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x



class MixedOp(nn.Module):

    def __init__(self, C, stride, switch, k=4):
        super(MixedOp, self).__init__()
        self.m_ops = nn.ModuleList()
        self.mp = nn.MaxPool2d(2, 2)
        self.k = k
        self.switch = switch
        for i in range(len(self.switch)):
            if self.switch[i]:
                primitive = PRIMITIVES[i]
                op = OPS[primitive](C // self.k, stride, False)
                if 'pool' in primitive:
                    op = nn.Sequential(op, nn.BatchNorm2d(C // self.k, affine=False))
                self.m_ops.append(op)

    def update_switch(self, index):
        del self.m_ops[index]

    def forward(self, x, weights):
        # channel proportion k=4
        dim_2 = x.shape[1]
        xtemp = x[:, :  dim_2 // self.k, :, :]
        xtemp2 = x[:, dim_2 // self.k:, :, :]
        temp1 = sum(w * op(xtemp) for w, op in zip(weights, self.m_ops))
        # reduction cell needs pooling before concat
        if temp1.shape[2] == x.shape[2]:
            ans = torch.cat([temp1, xtemp2], dim=1)
        else:
            ans = torch.cat([temp1, self.mp(xtemp2)], dim=1)
        ans = channel_shuffle(ans, self.k)
        # ans = torch.cat([ans[ : ,  dim_2//4:, :, :],ans[ : , :  dim_2//4, :, :]],dim=1)
        # except channe shuffle, channel shift also works
        return ans

class Cell(nn.Module):

    def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev, switches, k):
        super(Cell, self).__init__()
        self.reduction = reduction
        self.switches = switches

        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)
        self._steps = steps
        self._multiplier = multiplier

        self.cell_ops = nn.ModuleList()
        self._bns = nn.ModuleList()
        switch_count = 0
        for i in range(self._steps):
            for j in range(2 + i):
                stride = 2 if reduction and j < 2 else 1
                op = MixedOp(C, stride, switch=switches[switch_count], k=k)
                self.cell_ops.append(op)
                switch_count = switch_count + 1

    def update_switches(self, indexes):
        count = 0
        for op in self.cell_ops:
            op.switch = self.switches[count]
            op.update_switch(indexes[count])
            count = count + 1



    def forward(self, s0, s1, weights, weights2):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        offset = 0
        for i in range(self._steps):
            s = sum(weights2[offset + j] * self.cell_ops[offset + j](h, weights[offset + j]) for j, h in enumerate(states))
            offset += len(states)
            states.append(s)

        return torch.cat(states[-self._multiplier:], dim=1)


class Network(nn.Module):

    def __init__(self, C, num_classes, layers, criterion, steps=4, multiplier=4, stem_multiplier=3,
                 switches_normal=[], switches_reduce=[], k=16):
        super(Network, self).__init__()
        self._C = C
        self._num_classes = num_classes
        self._layers = layers
        self._criterion = criterion
        self._steps = steps
        self._multiplier = multiplier

        self.switches_normal = switches_normal
        self.switches_reduce = switches_reduce
        self.k = k


        C_curr = stem_multiplier * C


        self.stem = nn.Sequential(
            nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
            nn.BatchNorm2d(C_curr)
        )

        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = nn.ModuleList()
        reduction_prev = False
        for i in range(layers):
            if i in [layers // 2 - 1, 2 * layers // 2 - 1]:
                C_curr *= 2
                reduction = True
                switches = self.switches_reduce
            else:
                reduction = False
                switches = self.switches_normal
            cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev, switches, k=self.k)
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, multiplier * C_curr

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)

        self._initialize_alphas()
        self.prob_normal = self.betas_normal
        self.prob_reduce = self.betas_reduce
        self.prob = self.prob_normal

    def new(self):
        model_new = Network(self._C, self._num_classes, self._layers, self._criterion).cuda()
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.data)
        return model_new

    def forward(self, input):
        s0 = s1 = self.stem(input)
        for i, cell in enumerate(self.cells):
            if cell.reduction:
                weights = F.softmax(self.alphas_reduce, dim=-1)

                n = 3
                start = 2
                weights2 = F.softmax(self.betas_reduce[0:2], dim=-1)
                for i in range(self._steps - 1):
                    end = start + n
                    tw2 = F.softmax(self.betas_reduce[start:end], dim=-1)
                    start = end
                    n += 1
                    weights2 = torch.cat([weights2, tw2], dim=0)

            else:
                weights = F.softmax(self.alphas_normal, dim=-1)
                n = 3
                start = 2
                weights2 = F.softmax(self.betas_normal[0:2], dim=-1)
                for i in range(self._steps - 1):
                    end = start + n
                    tw2 = F.softmax(self.betas_normal[start:end], dim=-1)
                    start = end
                    n += 1
                    weights2 = torch.cat([weights2, tw2], dim=0)

            s0, s1 = s1, cell(s0, s1, weights, weights2)
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))

        weights_np = weights.data.cpu().numpy()
        indexes = []
        for i in range(14):
            idx = np.argmax(weights_np[i])
            indexes.append(idx)

        return logits, indexes

    def update_switches(self, indexes_normal, indexes_reduce):
        for cell in self.cells:
            if cell.reduction:
                cell.switches = self.switches_reduce
                cell.update_switches(indexes_reduce)
            else:
                cell.switches = self.switches_normal
                cell.update_switches(indexes_normal)

    def update_arch_parameters(self, w_drop_normal, w_drop_reduce, column_len):
        w_keep_normal = []
        w_keep_reduce = []

        for i in range(14):
            for j in range(column_len + 1):
                if j != w_drop_normal[i]:
                    w_keep_normal.append(j)
            with torch.no_grad():
                for k in range(column_len):
                    self.alphas_normal[i][k] = self.alphas_normal[i][w_keep_normal[k]]
                    self.alphas_normal.grad[i][k] = self.alphas_normal.grad[i][w_keep_normal[k]]
        for i in range(14):
            for j in range(column_len + 1):
                if j != w_drop_reduce[i]:
                    w_keep_reduce.append(j)
            with torch.no_grad():
                for k in range(column_len):
                    self.alphas_reduce[i][k] = self.alphas_reduce[i][w_keep_reduce[k]]
                    self.alphas_reduce.grad[i][k] = self.alphas_reduce.grad[i][w_keep_reduce[k]]
        with torch.no_grad():
            self.alphas_normal.data = self.alphas_normal[:, 0:column_len]
            self.alphas_reduce.data = self.alphas_reduce[:, 0:column_len]
            self.alphas_normal.grad.data = self.alphas_normal.grad[:, 0:column_len]
            self.alphas_reduce.grad.data = self.alphas_reduce.grad[:, 0:column_len]


    def _loss(self, input, target):
        logits, _ = self(input)
        return self._criterion(logits, target)

    def _initialize_alphas(self):
        k = sum(1 for i in range(self._steps) for n in range(2 + i))  # k=14
        num_ops = len(PRIMITIVES)

        self.alphas_normal = Variable(1e-3 * torch.randn(k, num_ops).cuda(), requires_grad=True)
        self.alphas_reduce = Variable(1e-3 * torch.randn(k, num_ops).cuda(), requires_grad=True)
        self.betas_normal = Variable(1e-3 * torch.randn(k).cuda(), requires_grad=True)
        self.betas_reduce = Variable(1e-3 * torch.randn(k).cuda(), requires_grad=True)
        self._arch_parameters = [
            self.alphas_normal,
            self.alphas_reduce,
            self.betas_normal,
            self.betas_reduce,
        ]

    def arch_parameters(self):
        return self._arch_parameters

    def genotype(self):

        def _parse(weights, weights2, switches):
            print('switches:', switches)
            gene = []
            n = 2
            start = 0
            for i in range(self._steps):
                end = start + n
                W = weights[start:end].copy()
                W2 = weights2[start:end].copy()
                for j in range(n):
                    W[j, :] = W[j, :] * W2[j]
                edges = sorted(range(i + 2),
                               key=lambda x: -max(W[x][k] for k in range(len(W[x])) if k != PRIMITIVES.index('none')
                                                  and switches[x][k]))[
                        :2]

                for j in edges:
                    k_best = None
                    for k in range(len(W[j])):
                        if k != PRIMITIVES.index('none') and switches[j][k]:
                            if k_best is None or W[j][k] > W[j][k_best]:
                                k_best = k
                    gene.append((PRIMITIVES[k_best], j))
                start = end
                n += 1
            # print('gene:', gene)
            return gene
        n = 3
        start = 2
        weightsr2 = F.softmax(self.betas_reduce[0:2], dim=-1)
        weightsn2 = F.softmax(self.betas_normal[0:2], dim=-1)

        for i in range(self._steps - 1):
            end = start + n
            tw2 = F.softmax(self.betas_reduce[start:end], dim=-1)
            tn2 = F.softmax(self.betas_normal[start:end], dim=-1)
            start = end
            n += 1
            weightsr2 = torch.cat([weightsr2, tw2], dim=0)
            weightsn2 = torch.cat([weightsn2, tn2], dim=0)
        gene_normal = _parse(F.softmax(self.alphas_normal, dim=-1).data.cpu().numpy(), weightsn2.data.cpu().numpy(), switches = self.switches_normal)
        gene_reduce = _parse(F.softmax(self.alphas_reduce, dim=-1).data.cpu().numpy(), weightsr2.data.cpu().numpy(), switches = self.switches_reduce)

        concat = range(2 + self._steps - self._multiplier, self._steps + 2)
        genotype = Genotype(
            normal=gene_normal, normal_concat=concat,
            reduce=gene_reduce, reduce_concat=concat
        )
        # print('genotype001:', genotype)
        return genotype

    def genotype_prob(self, normal_prob, reduce_prob):

        def _parse(weights, weights2, switches):
            print('switches:', switches)
            gene = []
            n = 2
            start = 0
            for i in range(self._steps):
                end = start + n
                W = weights[start:end].copy()
                W2 = weights2[start:end].copy()
                for j in range(n):
                    W[j, :] = W[j, :] * W2[j]
                edges = sorted(range(i + 2),
                               key=lambda x: -max(W[x][k] for k in range(len(W[x])) if k != PRIMITIVES.index('none')
                                                  and switches[x][k]))[
                        :2]

                # edges = sorted(range(i + 2), key=lambda x: -W2[x])[:2]
                for j in edges:
                    k_best = None
                    for k in range(len(W[j])):
                        if k != PRIMITIVES.index('none') and switches[j][k]:
                            if k_best is None or W[j][k] > W[j][k_best]:
                                k_best = k
                    gene.append((PRIMITIVES[k_best], j))
                start = end
                n += 1
            # print('gene:', gene)
            return gene
        n = 3
        start = 2
        weightsr2 = F.softmax(self.betas_reduce[0:2], dim=-1)
        weightsn2 = F.softmax(self.betas_normal[0:2], dim=-1)
        # weightsr2 = F.softmax(self.prob_reduce[0:2], dim=-1)
        # weightsn2 = F.softmax(self.prob_normal[0:2], dim=-1)
        for i in range(self._steps - 1):
            end = start + n
            # tw2 = F.softmax(self.prob_reduce[start:end], dim=-1)
            # tn2 = F.softmax(self.prob_normal[start:end], dim=-1)
            tw2 = F.softmax(self.betas_reduce[start:end], dim=-1)
            tn2 = F.softmax(self.betas_normal[start:end], dim=-1)
            start = end
            n += 1
            weightsr2 = torch.cat([weightsr2, tw2], dim=0)
            weightsn2 = torch.cat([weightsn2, tn2], dim=0)
        normal_prob = torch.tensor(normal_prob)
        reduce_prob = torch.tensor(reduce_prob)
        gene_normal = _parse(F.softmax(normal_prob, dim=-1).data.cpu().numpy(), weightsn2.data.cpu().numpy(), switches = self.switches_normal)
        gene_reduce = _parse(F.softmax(reduce_prob, dim=-1).data.cpu().numpy(), weightsr2.data.cpu().numpy(), switches = self.switches_reduce)

        concat = range(2 + self._steps - self._multiplier, self._steps + 2)
        genotype = Genotype(
            normal=gene_normal, normal_concat=concat,
            reduce=gene_reduce, reduce_concat=concat
        )
        # print('genotype001:', genotype)
        return genotype