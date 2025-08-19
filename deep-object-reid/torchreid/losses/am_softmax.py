"""
 Copyright (c) 2019 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

from __future__ import absolute_import
from __future__ import division

import math

import torch
import torch.nn as nn
import numpy as np

import torch.nn.functional as F
from torch.nn import Parameter


def focal_loss(input_values, gamma):
    """Computes the focal loss"""
    p = torch.exp(-input_values)
    loss = (1 - p) ** gamma * input_values
    return loss.mean()


class AngleSimpleLinear(nn.Module):
    """Computes cos of angles between input vectors and weights vectors"""

    def __init__(self, in_features, out_features):
        super(AngleSimpleLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.normal_().renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(self, x):
        cos_theta = F.normalize(x, dim=1).mm(F.normalize(self.weight, p=2, dim=0))
        return cos_theta.clamp(-1, 1)

    def get_centers(self):
        return torch.t(self.weight)


class AngleSimpleLinearV2(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, features):
        cosine = F.linear(F.normalize(features), F.normalize(self.weight))
        return cosine.clamp(-1, 1)


class AdaCos(nn.Module):
    def __init__(self, num_classes, m=0.50):
        super(AdaCos, self).__init__()
        self.s = math.sqrt(2) * math.log(num_classes - 1)
        self.m = m
    
    def forward(self, logits, label, iteration):
        theta = torch.acos(torch.clamp(logits, -1.0 + 1e-7, 1.0 - 1e-7))
        one_hot = torch.zeros_like(logits)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        with torch.no_grad():
            B_avg = torch.where(one_hot < 1, torch.exp(self.s * logits), torch.zeros_like(logits))
            B_avg = torch.sum(B_avg) / logits.size(0)
            theta_med = torch.median(theta[one_hot == 1])
            self.s = torch.log(B_avg) / torch.cos(torch.min(math.pi/4 * torch.ones_like(theta_med), theta_med))
        output = self.s * logits

        return focal_loss(F.cross_entropy(output, label, reduction='none'), 1.0)


class ArcfaceLoss(nn.Module):
    def __init__(self, s=30.0, m=0.5, pr_product=False):
        super().__init__()
        self.s = s
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

        self.gamma = 1.0
        self.pr_product = pr_product

    def forward(self, logits, labels, iteration):
        if self.pr_product:
            pr_alpha = torch.sqrt(1.0 - logits.pow(2.0))
            logits = pr_alpha.detach() * logits + logits.detach() * (1.0 - pr_alpha)
        
        logits = logits.float()
        cosine = logits

        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        
        index = torch.zeros_like(logits, dtype=torch.uint8)
        index.scatter_(1, labels.data.view(-1, 1), 1)
        output = torch.where(index, phi, logits)
        
        output *= self.s

        return focal_loss(F.cross_entropy(output, labels, reduction='none'), self.gamma)


class AMSoftmaxLoss(nn.Module):
    margin_types = ['cos', 'arc']

    def __init__(self, use_gpu=True, conf_penalty=0.0, margin_type='cos',
                 gamma=0.0, m=0.5, s=30, t=1.0, label_smooth=False, epsilon=0.1,
                 end_s=None, duration_s=None, skip_steps_s=None, pr_product=False,
                 class_counts=None):
        super(AMSoftmaxLoss, self).__init__()
        self.use_gpu = use_gpu
        self.conf_penalty = conf_penalty
        self.label_smooth = label_smooth
        self.epsilon = epsilon
        self.pr_product = pr_product

        assert margin_type in AMSoftmaxLoss.margin_types
        self.margin_type = margin_type
        assert gamma >= 0
        self.gamma = gamma
        assert m >= 0
        self.m = m
        assert s > 0
        self.start_s = s
        assert self.start_s > 0.0
        self.end_s = end_s
        self.duration_s = duration_s
        self.skip_steps_s = skip_steps_s
        self.last_scale = self.start_s
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        assert t >= 1
        self.t = t

        if class_counts is not None:
            class_ids = list(class_counts)
            class_ids.sort()

            counts = np.array([class_counts[class_id] for class_id in class_ids], dtype=np.float32)
            class_margins = (self.m / np.power(counts, 1. / 4.)).reshape((1, -1))

            self.register_buffer('class_margins', torch.from_numpy(class_margins).cuda())
            print('[INFO] Enabled adaptive margins for AM-Softmax loss')
        else:
            self.class_margins = self.m

    @staticmethod
    def get_last_info():
        return {}

    def get_last_scale(self):
        return self.last_scale

    @staticmethod
    def get_scale(start_scale, end_scale, duration, skip_steps, iteration, power=1.2):
        def _invalid(_v):
            return _v is None or _v <= 0

        if not _invalid(skip_steps) and iteration < skip_steps:
            return start_scale

        if _invalid(iteration) or _invalid(end_scale) or _invalid(duration):
            return start_scale

        skip_steps = skip_steps if not _invalid(skip_steps) else 0
        steps_to_end = duration - skip_steps
        if iteration < duration:
            factor = (end_scale - start_scale) / (1.0 - power)
            var_a = factor / (steps_to_end ** power)
            var_b = -factor * power / float(steps_to_end)

            iteration -= skip_steps
            out_value = var_a * np.power(iteration, power) + var_b * iteration + start_scale
        else:
            out_value = end_scale

        return out_value

    def forward(self, cos_theta, target, iteration=None):
        """
        Args:
            cos_theta (torch.Tensor): prediction matrix (before softmax) with
                shape (batch_size, num_classes).
            target (torch.LongTensor): ground truth labels with shape (batch_size).
                Each position contains the label index.
            iteration (int): current iteration
        """
        if self.pr_product:
            pr_alpha = torch.sqrt(1.0 - cos_theta.pow(2.0))
            cos_theta = pr_alpha.detach() * cos_theta + cos_theta.detach() * (1.0 - pr_alpha)

        if self.margin_type == 'cos':
            phi_theta = cos_theta - self.class_margins
        else:
            sine = torch.sqrt(1.0 - torch.pow(cos_theta, 2))
            phi_theta = cos_theta * self.cos_m - sine * self.sin_m
            phi_theta = torch.where(cos_theta > self.th, phi_theta, cos_theta - math.sin(math.pi - self.m) * self.m)

        index = torch.zeros_like(cos_theta, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)
        output = torch.where(index, phi_theta, cos_theta)

        self.last_scale = self.get_scale(self.start_s, self.end_s, self.duration_s, self.skip_steps_s, iteration)

        if self.gamma == 0.0 and self.t == 1.0:
            output *= self.last_scale

            if self.label_smooth:
                targets = torch.zeros(output.size()).scatter_(1, target.unsqueeze(1).data.cpu(), 1)
                if self.use_gpu:
                    targets = targets.cuda()

                num_classes = output.size(1)
                targets = (1.0 - self.epsilon) * targets + self.epsilon / float(num_classes)
                losses = (- targets * F.log_softmax(output, dim=1)).sum(dim=1)
            else:
                losses = F.cross_entropy(output, target, reduction='none')

            if self.conf_penalty > 0.0:
                probs = F.softmax(output, dim=1)
                log_probs = F.log_softmax(output, dim=1)
                entropy = torch.sum(-probs * log_probs, dim=1)

                losses = F.relu(losses - self.conf_penalty * entropy)

            with torch.no_grad():
                nonzero_count = max(losses.nonzero().size(0), 1)

            return losses.sum() / nonzero_count

        if self.t > 1:
            h_theta = self.t - 1 + self.t*cos_theta
            support_vecs_mask = (1 - index) * \
                torch.lt(torch.masked_select(phi_theta, index).view(-1, 1).repeat(1, h_theta.shape[1]) - cos_theta, 0)
            output = torch.where(support_vecs_mask, h_theta, output)

            return F.cross_entropy(self.last_scale * output, target)

        return focal_loss(F.cross_entropy(self.last_scale * output, target, reduction='none'), self.gamma)
