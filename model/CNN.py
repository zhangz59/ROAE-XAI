import torch
import torch.nn as nn
import torch.nn.functional as F




# Create CNN Model
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()

        self.conv_layer1 = self._conv_layer_set(1, 2)
        self.conv_layer2 = self._conv_layer_set(2, 4)
        self.conv_layer3 = self._conv_layer_set(4, 8)
        self.conv_layer4 = self._conv_layer_set(8,16)

        self.conv_layer5 = self._conv_layer_set_1_conv(16,16)

        self.conv_layer6 = self._conv_1(16,16)




        self.fc1 = nn.Linear(576, 162)
        self.fc2= nn.Linear(162,81)
        self.fc3 = nn.Linear(81,1)

        self.relu = nn.LeakyReLU()

        self.drop = nn.Dropout(p=0.15)

        self.avg = nn.AvgPool3d(1,1,1)

        self.classifier = nn.Linear(576, 81)




    def _conv_layer_set(self, in_c, out_c):
        conv_layer = nn.Sequential(
            nn.Conv3d(in_c, out_c, kernel_size=(3, 3, 3), padding=0),
            nn.BatchNorm3d(out_c),
            nn.MaxPool3d((2, 2, 2)),
            nn.LeakyReLU()

        )
        return conv_layer

    def _conv_layer_set_1_conv(self,in_c, out_c):
        conv_layer = nn.Sequential(
            nn.Conv3d(in_c, out_c, kernel_size=(1, 1, 1), padding=0),
            nn.BatchNorm3d(out_c),
            nn.LeakyReLU()

        )
        return conv_layer

    def _conv_1(self,in_c,out_c):
        conv_layer=nn.Sequential(
            nn.Conv3d(in_c,out_c,kernel_size=(1,1,1),padding=0)
        )
        return conv_layer


    def forward(self, x):
        # Set 1

        out = self.conv_layer1(x)
        out = self.conv_layer2(out)
        out = self.conv_layer3(out)
        out = self.conv_layer4(out)
        out = self.conv_layer5(out)

        out = self.drop(out)
        out = self.conv_layer6(out)
        out = out.view(out.size(0), -1)

        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)



        # softmax的回归方法，要改fc4的输出值为想要的分类数量
        # out = self.classifier(out)
        # out = F.log_softmax(out)
        # out = torch.argmax(out,1)
        # out = out.view(100,1)
        # out = out.to(torch.device('cuda:0'))





        return out


# Create CNN
#model = CNNModel()
#model.cuda()

#print(model)

# Cross Entropy Loss
#error = nn.CrossEntropyLoss()


