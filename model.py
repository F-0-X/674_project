import torch
import ocnn
from torch.nn import Sequential
from torch.nn import ModuleList
import random
from torch.nn import Sequential as Seq, Linear as Lin, LeakyReLU, GroupNorm

def conv_block(input_channel, depth, channel_in):
    conv_bn_re = []
    conv_bn_re.append(ocnn.OctreeConvBnRelu(depth, input_channel, 2 ** (9 - depth)))
    pool = []
    pool.append(ocnn.OctreeMaxPool(depth))

    for i in range(depth - 1, 2, -1):
        conv_bn_re.append(ocnn.OctreeConvBnRelu(i, 2 ** (9 - i - 1), 2 ** (9 - i)))
        pool.append(ocnn.OctreeMaxPool(i))

    return ModuleList(conv_bn_re), ModuleList(pool)

def MLP(channels, enable_group_norm=True):
    if enable_group_norm:
        num_groups = [0]
        for i in range(1, len(channels)):
            if channels[i] >= 32:
                num_groups.append(channels[i] // 32)
            else:
                num_groups.append(1)
        return Seq(*[
            Seq(torch.nn.utils.weight_norm(Lin(channels[i - 1], channels[i])), LeakyReLU(negative_slope=0.2), GroupNorm(num_groups[i], channels[i]))
            for i in range(1, len(channels))])
    else:
        return Seq(*[Seq(torch.nn.utils.weight_norm(Lin(channels[i - 1]), channels[i]), LeakyReLU(negative_slope=0.2))
                     for i in range(1, len(channels))])

class ourNet(torch.nn.Module):
    def __init__(self, depth, channel_in, nout):
        super(ourNet, self).__init__()
        self.depth = depth
        self.channel_in = channel_in

        channels = [2 ** max(9 - i, 2) for i in range(depth + 1)]
        channels.append(channel_in)
        self.conv_block, self.pool_block = conv_block(channel_in, depth, channel_in)
        self.octree2voxel = ocnn.FullOctree2Voxel(2)

        self.drop = torch.nn.Dropout(p=0.4)
        self.drop2 = torch.nn.Dropout(p=0.6)

        self.flat = torch.nn.Flatten(start_dim=1)
        self.fc = torch.nn.utils.weight_norm(torch.nn.Linear(channels[3] * 64, channels[2], bias=False))
        self.bn = torch.nn.BatchNorm1d(channels[2])
        self.relu = torch.nn.ReLU(inplace=True)

        self.fcRd = torch.nn.utils.weight_norm(torch.nn.Linear(channels[2], channels[2], bias=False))
        self.fc2 = torch.nn.utils.weight_norm(torch.nn.Linear(channels[2], nout))

        self.mlp1 = MLP([channels[2], 32, 64, 128])
        self.mlp2 = MLP([128, 128])
        #  concatenate
        self.mlp3 = MLP([256, 128, channels[2]])

    def forward(self, octree):
        x = ocnn.octree_property(octree, 'feature', self.depth)
        assert x.size(1) == self.channel_in
        for i in range(len(self.conv_block)):
            x = self.conv_block[i](x, octree)
            x = self.pool_block[i](x, octree)
        x = self.octree2voxel(x)
        # data = self.header(data)
        x = self.drop(x)
        x = self.flat(x)

        x = self.fc(x)
        xskip = x
        x = self.bn(x)
        x = self.relu(x)

        r = random.randint(0, 1)
        if r == 1:

            x = self.mlp1(x)
            xm = x
            x = self.mlp2(x)
            glob_feat, indice = torch.max(xm, 0)
            glob_feat = glob_feat.repeat(x.size()[0], 1)
            concat = torch.cat((xskip, glob_feat), 1)
            x = self.mlp3(concat)

        x = self.drop2(x)
        x = self.fc2(x)
        return x