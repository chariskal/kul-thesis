import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
import numpy as np

import network.resnet38d


class Net(network.resnet38d.Net):
    def __init__(self, k_cluster, from_round_nb):
        super().__init__()

        self.k = k_cluster
        self.from_round_nb = from_round_nb
        print('k_cluster: {}'.format(self.k))
        print('Round: {}'.format(self.from_round_nb))

        self.dropout7 = torch.nn.Dropout2d(0.5)

        # class 20
        if self.from_round_nb == 0:
            self.fc8 = nn.Conv2d(4096, 8, 1, bias=False)

            torch.nn.init.xavier_uniform_(self.fc8.weight)

            self.not_training = [self.conv1a, self.b2, self.b2_1, self.b2_2]
            self.from_scratch_layers = [self.fc8]


        # class 20 + class 200
        else:
            self.fc8_8 = nn.Conv2d(4096, 8, 1, bias=False)
            self.fc8_80 = nn.Conv2d(4096, self.k*8, 1, bias=False)

            torch.nn.init.xavier_uniform_(self.fc8_8.weight)
            torch.nn.init.xavier_uniform_(self.fc8_80.weight)

            self.not_training = [self.conv1a, self.b2, self.b2_1, self.b2_2]
            self.from_scratch_layers = [self.fc8_8, self.fc8_80]



    def forward(self, x, from_round_nb):
        x = super().forward(x)
        x = self.dropout7(x)
        x = F.avg_pool2d(x, kernel_size=(x.size(2), x.size(3)), padding=0)

        feature = x
        feature = feature.view(feature.size(0), -1)

        # class 20
        if from_round_nb == 0:
            x = self.fc8(x)
            x = x.view(x.size(0), -1)
            y = torch.sigmoid(x)
            return x, feature, y

        # class 20 + class 200
        else:
            x_8 = self.fc8_8(x)
            x_8 = x_8.view(x_8.size(0), -1)
            y_8 = torch.sigmoid(x_8)


            x_80 = self.fc8_80(x)
            x_80 = x_80.view(x_80.size(0), -1)
            y_80 = torch.sigmoid(x_80)

            return x_8, feature, y_8, x_80, y_80



    def multi_label(self, x):
        x = torch.sigmoid(x)
        tmp = x.cpu()
        tmp = tmp.data.numpy()
        _, cls = np.where(tmp>0.5)

        return cls, tmp


    def forward_cam(self, x):
        x = super().forward(x)
        x = F.conv2d(x, self.fc8.weight)
        x = F.relu(x)

        return x


    def forward_two_cam(self, x):
        x_ = super().forward(x)

        x_8 = F.conv2d(x_, self.fc8_8.weight)
        cam_8 = F.relu(x_8)

        x_80 = F.conv2d(x_, self.fc8_80.weight)
        cam_80 = F.relu(x_80)

        return cam_8, cam_80

    def get_parameter_groups(self):
        groups = ([], [], [], [])

        for m in self.modules():

            if isinstance(m, nn.Conv2d):

                if m.weight.requires_grad:
                    if m in self.from_scratch_layers:
                        groups[2].append(m.weight)
                    else:
                        groups[0].append(m.weight)

                if m.bias is not None and m.bias.requires_grad:

                    if m in self.from_scratch_layers:
                        groups[3].append(m.bias)
                    else:
                        groups[1].append(m.bias)

        return groups
