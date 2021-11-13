import torch
import torch.nn as nn
from torchinfo import summary


def select_net(network_name):
    ''' Network selector '''
    if network_name == "Test":
        net = Test()
    elif network_name == "FpDnn":
        net = FpDnn()

    return net


class Test(nn.Module):
    def __init__(self):
        super(Test, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(6, 3, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, image):
        x = self.conv1(image)
        x = self.relu(x)
        x = self.conv2(x)
        return x




class UNet(nn.Module):
    # https://www.youtube.com/watch?v=u1loyDCoGbE
    def __init__(self):
        super(UNet, self).__init__()

        self.max_pool_2x2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down_conv_1 = self.double_conv(1, 64)
        self.down_conv_2 = self.double_conv(64, 128)
        self.down_conv_3 = self.double_conv(128, 256)
        self.down_conv_4 = self.double_conv(256, 512)
        self.down_conv_5 = self.double_conv(512, 1024)

        self.up_trans_1 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=2, stride=2)
        self.up_conv_1 = self.double_conv(1024, 512)

        self.up_trans_2 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2)
        self.up_conv_2 = self.double_conv(512, 256)

        self.up_trans_3 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2)
        self.up_conv_3 = self.double_conv(256, 128)

        self.up_trans_4 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2)
        self.up_conv_4 = self.double_conv(128, 64)

        self.out = nn.Conv2d(in_channels=64, out_channels=2, kernel_size=1)


    def double_conv(self, in_c, out_c):
        conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3),
            nn.ReLU(inplace=True)
        )
        return conv

    def crop_img(self, tensor, target_tensor):
        target_size = target_tensor.size()[2]
        tensor_size = tensor.size()[2]
        delta = tensor_size - target_size
        delta = delta //2
        return tensor[:, :, delta:tensor_size-delta, delta:tensor_size-delta]


    def forward(self, image):
        # bs, c, h, w
        # encoder
        x1 = self.down_conv_1(image)  # skip connection
        print(x1.size())
        x2 = self.max_pool_2x2(x1)
        x3 = self.down_conv_2(x2)  # skip connection
        x4 = self.max_pool_2x2(x3)
        x5 = self.down_conv_3(x4)  # skip connection
        x6 = self.max_pool_2x2(x5)
        x7 = self.down_conv_4(x6)  # skip connection
        x8 = self.max_pool_2x2(x7)
        x9 = self.down_conv_5(x8)
        print(x9.size())

        # decoder
        x = self.up_trans_1(x9)
        y = self.crop_img(x7, x)
        x = self.up_conv_1(torch.cat([x, y], 1))

        x = self.up_trans_2(x)
        y = self.crop_img(x5, x)
        x = self.up_conv_2(torch.cat([x, y], 1))

        x = self.up_trans_3(x)
        y = self.crop_img(x3, x)
        x = self.up_conv_3(torch.cat([x, y], 1))

        x = self.up_trans_4(x)
        y = self.crop_img(x1, x)
        x = self.up_conv_4(torch.cat([x, y], 1))

        x = self.out(x)

        print(x.size())
        # print(x7.size())
        # print(y.size())

        return x



# =====================================================================

class Encode(nn.Module):
    def __init__(self, in_c, out_c, k_size=4, pad=1, bn=True):
        super(Encode, self).__init__()
        self.activation = nn.LeakyReLU(0.2, inplace=True)
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=k_size, stride=2, padding=pad)
        self.bn = nn.BatchNorm2d(out_c)
        self.bn_sw = bn

    def forward(self, x):
        x = self.activation(x)
        x = self.conv(x)
        if self.bn_sw:
            x = self.bn(x)

        return x


class Decode(nn.Module):
    def __init__(self, in_c, out_c, k_size=4, st=1, pad=1, bn=True):
        super(Decode, self).__init__()
        self.activation = nn.ReLU(inplace=True)
        self.convt =  nn.ConvTranspose2d(in_c, out_c, kernel_size=k_size, stride=st, padding=pad)
        self.bn = nn.BatchNorm2d(out_c)
        self.bn_sw = bn

    def forward(self, x):
        x = self.activation(x)
        x = self.convt(x)
        if self.bn_sw:
            x = self.bn(x)
        return x



class FpDnn(nn.Module):
    ''' Flow prediction with Deep neural network '''
    def __init__(self):
        super(FpDnn, self).__init__()
        ch = 64

        self.layer1 = nn.Conv2d(3, ch, kernel_size=4, stride=2, padding=1)

        self.layer2 = Encode(ch,   ch*2)
        self.layer3 = Encode(ch*2, ch*2)
        self.layer4 = Encode(ch*2, ch*4)
        self.layer5 = Encode(ch*4, ch*8)
        self.layer6 = Encode(ch*8, ch*8, k_size=2, pad=0)
        self.layer7 = Encode(ch*8, ch*8, k_size=2, pad=0, bn=False)

        self.dlayer7 = Decode(ch*8,  ch*8, k_size=2, st=2, pad=0)
        self.dlayer6 = Decode(ch*16, ch*8, k_size=2, st=2, pad=0)
        self.dlayer5 = Decode(ch*16, ch*4, k_size=4, st=2, pad=1)
        self.dlayer4 = Decode(ch*8,  ch*2, k_size=4, st=2, pad=1)
        self.dlayer3 = Decode(ch*4,  ch*2, k_size=4, st=2, pad=1)
        self.dlayer2 = Decode(ch*4,  ch  , k_size=4, st=2, pad=1)
        self.dlayer1 = Decode(ch*2,  3   , k_size=4, st=2, pad=1, bn=False)



    def forward(self, x):
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        out5 = self.layer5(out4)
        out6 = self.layer6(out5)
        out7 = self.layer7(out6)

        dout7 = self.dlayer7(out7)
        dout7_out6 = torch.cat([dout7, out6], dim=1)

        dout6 = self.dlayer6(dout7_out6)
        dout6_out5 = torch.cat([dout6, out5], dim=1)

        dout5 = self.dlayer5(dout6_out5)
        dout5_out4 = torch.cat([dout5, out4], dim=1)

        dout4 = self.dlayer4(dout5_out4)
        dout4_out3 = torch.cat([dout4, out3], dim=1)

        dout3 = self.dlayer3(dout4_out3)
        dout3_out2 = torch.cat([dout3, out2], dim=1)

        dout2 = self.dlayer2(dout3_out2)
        dout2_out1 = torch.cat([dout2, out1], dim=1)

        dout1 = self.dlayer1(dout2_out1)

        return dout1





if __name__ == "__main__":
    image = torch.rand((2, 3, 128, 128))  # batch_size, channels, height, width
    # net = Test()
    # print( net(image).shape )


    net = FpDnn()
    print(net(image).shape)

    # print(net)

    summary(net, input_size=(2, 3, 128, 128))
