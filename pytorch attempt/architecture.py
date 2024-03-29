import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch
import torchvision.transforms as transforms
import numpy as np

def double_conv(in_c, out_c):
    conv = nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=3), 
        nn.ReLU(inplace=True),
        nn.Conv2d(out_c, out_c, kernel_size=3), 
        nn.ReLU(inplace=True)
    )
    return conv

def crop_img(tensor, target_tensor):
    target_size = target_tensor.size()[2]
    tensor_size = tensor.size()[2]
    delta = tensor_size - target_size
    if delta % 2 == 0:
        delta = delta // 2 
    else: 
        delta = delta // 2
        return tensor[:,:,delta:tensor_size-delta-1,delta:tensor_size-delta-1]
    
    return tensor[:,:,delta:tensor_size-delta,delta:tensor_size-delta]

def padding(tensor, target_tensor):
    tensor_size = tensor.size()[2] 
    target_size = target_tensor.size()[2] 
    delta = target_size - tensor_size
    if delta % 2 == 0:
        delta = delta // 2 
        return F.pad(tensor, (delta, delta, delta, delta))
    else: 
        delta = delta // 2
        return F.pad(tensor, (delta + 1, delta, delta + 1, delta))


class UNet(nn.Module):

    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super(UNet, self).__init__()

        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        for feature in features:
            self.downs.append(double_conv(in_channels, feature))
            in_channels = feature

        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2)
            )
            self.ups.append(double_conv(feature*2, feature))

        self.bottleneck = double_conv(features[-1], features[-1]*2)

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):

        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                # x = padding(x, skip_connection)
                skip_connection = crop_img(skip_connection, x)

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)

        return self.final_conv(x)


if __name__ == "__main__":
    image = torch.rand((1,3,1024,1024))
    model = UNet()
    output = model(image)
    print(output.shape)
    print(padding(output, image).shape)





