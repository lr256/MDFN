import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class BasicConvolution2D(nn.Module):
    def __init__(self, inChannels, outChannels, kernelSize, stride, padding = 0):
        super().__init__()
        self.Convolution = nn.Conv2d(inChannels, outChannels, kernelSize, stride, padding, bias = False)
        self.BatchNormalization = nn.BatchNorm2d(outChannels, eps = 0.001, momentum = 0.1, affine = True)
        self.ReLU = nn.ReLU(inplace = False)

    def forward(self, x):
        x = self.Convolution(x)
        x = self.BatchNormalization(x)
        x = self.ReLU(x)
        return x

class Mixed5B(nn.Module):
    def __init__(self):
        super().__init__()
        self.Branch0 = BasicConvolution2D(192, 96, kernelSize = 1, stride = 1)
        self.Branch1 = nn.Sequential(
            BasicConvolution2D(192, 48, kernelSize = 1, stride = 1),
            BasicConvolution2D(48, 64, kernelSize = 5, stride = 1, padding = 2))
        self.Branch2 = nn.Sequential(
            BasicConvolution2D(192, 64, kernelSize = 1, stride = 1),
            BasicConvolution2D(64, 96, kernelSize = 3, stride = 1, padding = 1),
            BasicConvolution2D(96, 96, kernelSize = 3, stride = 1, padding = 1))
        self.Branch3 = nn.Sequential(
            nn.AvgPool2d(3, stride = 1, padding = 1, count_include_pad = False),
            BasicConvolution2D(192, 64, kernelSize = 1, stride =1))

    def forward(self, x):
        x0 = self.Branch0(x)
        x1 = self.Branch1(x)
        x2 = self.Branch2(x)
        x3 = self.Branch3(x)
        x = torch.cat((x0, x1, x2, x3), 1)
        return x

class Block35(nn.Module):
    def __init__(self, scale = 1.0):
        super().__init__()
        self.Scale = scale
        self.Branch0 = BasicConvolution2D(320, 32, kernelSize = 1, stride = 1)
        self.Branch1 = nn.Sequential(
            BasicConvolution2D(320, 32, kernelSize = 1, stride = 1),
            BasicConvolution2D(32, 32, kernelSize = 3, stride = 1, padding = 1))
        self.Branch2 = nn.Sequential(
            BasicConvolution2D(320, 32, kernelSize = 1, stride = 1),
            BasicConvolution2D(32, 48, kernelSize = 3, stride = 1, padding = 1),
            BasicConvolution2D(48, 64, kernelSize = 3, stride = 1, padding = 1))
        self.Convolution = nn.Conv2d(128, 320, kernel_size = 1, stride = 1)
        self.ReLU = nn.ReLU(inplace = False)

    def forward(self, x):
        x0 = self.Branch0(x)
        x1 = self.Branch1(x)
        x2 = self.Branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        out = self.Convolution(out)
        out = out * self.Scale + x
        out = self.ReLU(out)
        return out

class Mixed6A(nn.Module):
    def __init__(self):
        super().__init__()
        self.Branch0 = BasicConvolution2D(320, 384, kernelSize = 3, stride = 2)
        self.Branch1 = nn.Sequential(
            BasicConvolution2D(320, 256, kernelSize = 1, stride = 1),
            BasicConvolution2D(256, 256, kernelSize = 3, stride = 1, padding = 1),
            BasicConvolution2D(256, 384, kernelSize = 3, stride = 2))
        self.Branch2 = nn.MaxPool2d(3, stride = 2)

    def forward(self, x):
        x0 = self.Branch0(x)
        x1 = self.Branch1(x)
        x2 = self.Branch2(x)
        x = torch.cat((x0, x1, x2), 1)
        return x

class Block17(nn.Module):
    def __init__(self, scale = 1.0):
        super().__init__()
        self.Scale = scale
        self.Branch0 = BasicConvolution2D(1088, 192, kernelSize = 1, stride = 1)
        self.Branch1 = nn.Sequential(
            BasicConvolution2D(1088, 128, kernelSize = 1, stride = 1),
            BasicConvolution2D(128, 160, kernelSize = (1, 7), stride = 1, padding = (0, 3)),
            BasicConvolution2D(160, 192, kernelSize = (7, 1), stride = 1, padding = (3, 0)))
        self.Convolution = nn.Conv2d(384, 1088, kernel_size = 1, stride = 1)
        self.ReLU = nn.ReLU(inplace = False)

    def forward(self, x):
        x0 = self.Branch0(x)
        x1 = self.Branch1(x)
        out = torch.cat((x0, x1), 1)
        out = self.Convolution(out)
        out = out * self.Scale + x
        out = self.ReLU(out)
        return out

class Mixed7A(nn.Module):
    def __init__(self):
        super().__init__()
        self.Branch0 = nn.Sequential(
            BasicConvolution2D(1088, 256, kernelSize = 1, stride = 1),
            BasicConvolution2D(256, 384, kernelSize = 3, stride = 2))
        self.Branch1 = nn.Sequential(
            BasicConvolution2D(1088, 256, kernelSize = 1, stride = 1),
            BasicConvolution2D(256, 288, kernelSize = 3, stride = 2))
        self.Branch2 = nn.Sequential(
            BasicConvolution2D(1088, 256, kernelSize = 1, stride = 1),
            BasicConvolution2D(256, 288, kernelSize = 3, stride = 1, padding = 1),
            BasicConvolution2D(288, 320, kernelSize = 3, stride = 2))
        self.Branch3 = nn.MaxPool2d(3, stride = 2)

    def forward(self, x):
        x0 = self.Branch0(x)
        x1 = self.Branch1(x)
        x2 = self.Branch2(x)
        x3 = self.Branch3(x)
        x = torch.cat((x0, x1, x2, x3), 1)
        return x

class Block8(nn.Module):
    def __init__(self, scale = 1.0, needReLU = True):
        super().__init__()
        self.Scale = scale
        self.NeedReLU = needReLU
        self.Branch0 = BasicConvolution2D(2080, 192, kernelSize = 1, stride = 1)
        self.Branch1 = nn.Sequential(
            BasicConvolution2D(2080, 192, kernelSize = 1, stride = 1),
            BasicConvolution2D(192, 224, kernelSize = (1, 3), stride = 1, padding = (0, 1)),
            BasicConvolution2D(224, 256, kernelSize = (3, 1), stride = 1, padding = (1, 0)))
        self.Convolution = nn.Conv2d(448, 2080, kernel_size = 1, stride = 1)
        if self.NeedReLU:
            self.ReLU = nn.ReLU(inplace = False)

    def forward(self, x):
        x0 = self.Branch0(x)
        x1 = self.Branch1(x)
        out = torch.cat((x0, x1), 1)
        out = self.Convolution(out)
        out = out * self.Scale + x
        if self.NeedReLU:
            out = self.ReLU(out)
        return out

class MultiScaleBlock(nn.Module):
    def __init__(self, inChannel, outChannel, stride, padding = 0):
        super(MultiScaleBlock, self).__init__()
        self.Branch0 = BasicConvolution2D(inChannel, outChannel // 4, kernelSize = 3, stride = stride, padding = 0 + padding)
        self.Branch1 = BasicConvolution2D(inChannel, outChannel // 4, kernelSize = 5, stride = stride, padding = 1 + padding)
        self.Branch2 = BasicConvolution2D(inChannel, outChannel // 4, kernelSize = 7, stride = stride, padding = 2 + padding)
        self.Branch3 = BasicConvolution2D(inChannel, outChannel // 4, kernelSize = 9, stride = stride, padding = 3 + padding)

    def forward(self, x):
        x0 = self.Branch0(x)
        x1 = self.Branch1(x)
        x2 = self.Branch2(x)
        x3 = self.Branch3(x)
        return torch.cat((x0, x1, x2, x3), 1)

class TFN(nn.Module):

    def __init__(self,numOfClasses):
       
        super(TFN, self).__init__()
        self.US_subnet = SubNet(1024,32, 0.2)
        self.CDFI_subnet = SubNet(1024,32, 0.2)
        self.marker_subnet = SubNet(10, 32, 0.2)
        self.post_fusion_dropout = nn.Dropout(p=0.2)

        self.post_fusion_layer_last_CDFI = nn.Sequential(nn.Linear(33*33*33, 33),
                                                         nn.ReLU(),
                                                         nn.Linear(33, 5))

    def forward(self, US_x, CDFI_x,marker_x):
        
        US_h = self.US_subnet(US_x)
        CDFI_h = self.CDFI_subnet(CDFI_x)
        marker_x = self.marker_subnet(marker_x)
        batch_size = US_x.data.shape[0]
        if US_h.is_cuda:
            DTYPE = torch.cuda.FloatTensor
        else:
            DTYPE = torch.FloatTensor

        _US_h = torch.cat((Variable(torch.ones(batch_size, 1).type(DTYPE), requires_grad=False), US_h), dim=1)
        _CDFI_h = torch.cat((Variable(torch.ones(batch_size, 1).type(DTYPE), requires_grad=False), CDFI_h), dim=1)
        _marker_x = torch.cat((Variable(torch.ones(batch_size, 1).type(DTYPE), requires_grad=False), marker_x), dim=1)
        # 计算张量积
        x1 = _US_h.unsqueeze(1).unsqueeze(2)  # shape: (batch_size, 1, 1, output_dim)
        x2 = _CDFI_h.unsqueeze(1).unsqueeze(3)  # shape: (batch_size, 1, output_dim, 1)
        x3 = _marker_x.unsqueeze(2).unsqueeze(3)  # shape: (batch_size, output_dim, 1, 1)
        # 计算张量积
        fused = x1 * x2 * x3  # shape: (batch_size, output_dim, output_dim, output_dim)
        # 展平张量
        fused = fused.view(fused.size(0), -1) 
        post_fusion_dropped = self.post_fusion_dropout(fused)
       
        post_fusion_dropped = post_fusion_dropped.squeeze() 
        
        output = F.sigmoid(self.post_fusion_layer_last_CDFI(post_fusion_dropped))

        return output


class InceptionResNetV2(nn.Module):
    def __init__(self, numOfClasses):
        super().__init__()
        self.Convolution1A = BasicConvolution2D(1, 32, kernelSize = 3, stride = 2)#1,32
        self.Convolution2A = BasicConvolution2D(32, 32, kernelSize = 3, stride = 1)
        self.Convolution2B = BasicConvolution2D(32, 64, kernelSize = 3, stride = 1, padding = 1)
        self.MaxPooling3A = nn.MaxPool2d(3, stride = 2)
        self.Convolution3B = BasicConvolution2D(64, 80, kernelSize = 1, stride = 1)
        self.Convolution4A = BasicConvolution2D(80, 192, kernelSize = 1, stride = 1)
        self.MaxPooling5A = nn.MaxPool2d(3, stride = 2)
        self.Mixed5B = Mixed5B()
        self.Repeat0 = nn.Sequential(
            Block35(scale = 0.17),
            Block35(scale = 0.17),
            Block35(scale = 0.17),
            Block35(scale = 0.17),
            Block35(scale = 0.17),
            Block35(scale = 0.17),
            Block35(scale = 0.17),
            Block35(scale = 0.17),
            Block35(scale = 0.17),
            Block35(scale = 0.17))
        self.Mixed6A = Mixed6A()
        self.Repeat1 = nn.Sequential(
            Block17(scale = 0.10),
            Block17(scale = 0.10),
            Block17(scale = 0.10),
            Block17(scale = 0.10),
            Block17(scale = 0.10),
            Block17(scale = 0.10),
            Block17(scale = 0.10),
            Block17(scale = 0.10),
            Block17(scale = 0.10),
            Block17(scale = 0.10),
            Block17(scale = 0.10),
            Block17(scale = 0.10),
            Block17(scale = 0.10),
            Block17(scale = 0.10),
            Block17(scale = 0.10),
            Block17(scale = 0.10),
            Block17(scale = 0.10),
            Block17(scale = 0.10),
            Block17(scale = 0.10),
            Block17(scale = 0.10))
        self.Mixed7A = Mixed7A()
        self.Repeat2 = nn.Sequential(
            Block8(scale = 0.20),
            Block8(scale = 0.20),
            Block8(scale = 0.20),
            Block8(scale = 0.20),
            Block8(scale = 0.20),
            Block8(scale = 0.20),
            Block8(scale = 0.20),
            Block8(scale = 0.20),
            Block8(scale = 0.20),
            Block8(scale = 0.20))
        self.Block8 = Block8(needReLU = False)
        self.Convolution7B = BasicConvolution2D(2080, 1536, kernelSize = 1, stride = 1)
        self.AveragePooling1A = nn.AvgPool2d(5, count_include_pad=False)#224 224

        self.LastLinear1 = nn.Sequential(
            nn.Linear(1536, 512),  
            nn.ReLU(),
            nn.Linear(512, numOfClasses))  # nn.Linear(1537, numOfClasses)#!!!1536
        self.LastLinear_relu = nn.Sequential(
            nn.Linear(1536, 1536),  
            nn.ReLU(),
            nn.Linear(1536, 1024))
        self.LastLinear_1 = nn.Sequential(
            nn.Linear(1536+10, 512),
            nn.ReLU(),
            nn.Linear(512, numOfClasses),
        )

    def forward(self, x,x2,ratio):#x2为肿瘤标记物
        x_f_1 = self.Convolution1A(x)
        x_f_2= self.Convolution2A(x_f_1)
        x_f_3 = self.Convolution2B(x_f_2)
        x_f_4 = self.MaxPooling3A(x_f_3)
        x_f_5 = self.Convolution3B(x_f_4)
        x_f_6 = self.Convolution4A(x_f_5)
        x = self.MaxPooling5A(x_f_6)
        x = self.Mixed5B(x)
        x= self.Repeat0(x)
        x_f_7 = self.Mixed6A(x)
        x = self.Repeat1(x_f_7)
        x_f_8 = self.Mixed7A(x)
        x = self.Repeat2(x_f_8)
        x = self.Block8(x)
        x_f_9 = self.Convolution7B(x)
        x = self.AveragePooling1A(x_f_9)
        x = x.view(x.size(0), -1)
        if ratio ==0:
            x = self.LastLinear1(x)
            return x
        elif ratio==1:
            x2 = x2.view(x2.size(0), -1)
            x = torch.cat((x, x2), 1)
            x = self.LastLinear_1(x)
            return x
        elif ratio==2:
            x1 = x
            x = self.LastLinear1(x)
            return x,x1
        else:
            x = x
            x = self.LastLinear_relu(x)
            return x

class ChannelAttention(nn.Module):
    def __init__(self):
        super(ChannelAttention, self).__init__()
        self.GlobalPooling = nn.AdaptiveAvgPool2d(1)
        self.RGB_attention = nn.Sequential(
            nn.Linear(3, 10),
            nn.ReLU(),
            nn.Linear(10, 3),
            nn.Softmax(-1)
        )

    def forward(self, x):
        channel_weights = self.RGB_attention(self.GlobalPooling(x).view(-1, 3))
        x = torch.mul(x, channel_weights.view(-1, 3, 1, 1))
        return x

class MDFN(nn.Module):
        def __init__(self, numOfClasses):
            super().__init__()
            self.CDFI_color = ChannelAttention()  
            self.model1 = InceptionResNetV2(2) 
            self.model1.Convolution1A = BasicConvolution2D(1, 32, kernelSize=3, stride=2)  
            self.model2 = InceptionResNetV2(2)
            self.model2.Convolution1A = BasicConvolution2D(3, 32, kernelSize=3, stride=2)  
            self.LastLinear_all = nn.Sequential(
                nn.Linear(1024*2, 512),  
                nn.ReLU(),
                nn.Linear(512, numOfClasses))
            self.TFN = TFN(numOfClasses)
            self.fc_US = nn.Sequential(
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Linear(512, numOfClasses))
            
        def forward(self, x, x2,ratio,flag_feature):  # x2为纵横比
            
            x_us = x[:,0,:,:].unsqueeze(1)
            x_cdfi = x[:, 1:, :, :]
            x_cdfi = self.CDFI_color(x_cdfi)
            f_us = self.model1(x_us,0,3)   
            f_cdfi = self.model2(x_cdfi,0,3) 
            x2 = x2.view(x2.size(0), -1)  #
  
            if ratio ==1:#加入肿瘤标记物
                f_us_2 = torch.cat((f_us, x2), dim=1)
                f_cdfi_2 = torch.cat((f_cdfi, x2),dim=1)
                x = self.TFN(f_us,f_cdfi,x2)#TFN融合
                x_us_2 = self.fc_US(f_us)
                x_cdfi_2 = self.fc_US(f_cdfi)
            else:
                f_map = torch.cat([f_us, f_cdfi], dim=1)
                ddd = f_map
                d_new = ddd
                x = self.LastLinear_all(d_new)
                x_us_2 = self.fc_US(f_us)
                x_cdfi_2 = self.fc_US(f_cdfi)
            return x, x_us_2, x_cdfi_2
           
