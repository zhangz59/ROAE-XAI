import torch
from torch import nn
import torch.nn.functional as F

# vgg后面都是经过三个全连接层以及soft-max函数
# 采用前面参数区分，后面统一构建

class VGG(nn.Module):
    def __init__(self, features, class_num=100, init_weight=False):
        super(VGG, self).__init__()
        self.features = features
        # 全连接
        self.classifier = nn.Sequential(
            nn.Dropout(),  # 减小过拟合，50%失活神经元
            nn.Linear(18432, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 2048),
            nn.ReLU(True),
            nn.Linear(2048, 1024),
            nn.Linear(1024,512),
            nn.Linear(512,class_num),
            #nn.Linear(class_num,1)
        )
        if init_weight:
            self._initialize_weights()

    def forward(self, x):
        # 前面模型搭建
        x = self.features(x)
        # 展平操作
        # N * 512 * 7 * 7
        x = torch.flatten(x, start_dim=1)
        # 全连接
        x = self.classifier(x)
        #softmax
        x = F.log_softmax(x)
        #x = torch.Tensor([[torch.argmax(x[0])],[torch.argmax(x[1])]])
        results = []
        for row in x:
            # Apply torch.argmax to each row and append the result to the list
            result_row = torch.argmax(row)
            results.append(result_row)
        x = results

        # Convert the list of results into a new tensor
        x = torch.Tensor(x).unsqueeze(1)




        x = x.to(torch.device('cuda:0'))
        return x

    # 初始化权重函数
    def _initialize_weights(self):
        for m in self.modules:
            # 若是卷积层，则利用xavier进行初始化
            if isinstance(m, nn.Conv3d):
                nn.init.xavier_uniform_(m.weight)
                # 若使用偏置
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0) # 将偏置置为0
            # 若是全连接层
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)


# 以列表形式记录vgg各个模型的参数
cfgs = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}

# 构建前半部分模型
def make_feature(cfgs):
    layers = []
    in_channels = 1 # 最初输入cannel为3
    for v in cfgs:
        # 最大池化
        if v == 'M':
            layers += [nn.MaxPool3d(kernel_size=(3,3,3), stride=(2,2,2),padding=1)]
        else:
            conv3d = nn.Conv3d(in_channels=in_channels, out_channels=v, kernel_size=(3,3,3),padding=1)
            layers += [conv3d, nn.ReLU(inplace=True)] # 采用ReLu激活函数
            in_channels = v  # 卷积后，卷积层输入channel变成上一层的channel
    # torch.nn.Sequential(* args) 按顺序添加到容器
    return nn.Sequential(*layers)

# 实例化vgg
def vgg(model_name="vgg19", **kwargs):
    try:
        cfg = cfgs[model_name]
    except:
        print("Warning: Model {} not in cfs dict!".format(model_name))
        exit(-1)

    model = VGG(make_feature(cfg), **kwargs)
    return model


# if __name__ == '__main__':
#     # 默认是vgg16，可修改model名字
#     vgg_model = vgg(model_name="vgg13")
#     print(vgg_model)