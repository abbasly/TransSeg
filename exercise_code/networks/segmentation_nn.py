"""SegmentationNN"""
import torch
import torch.nn as nn
from torchvision import models

class ConvLayer(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.activation = nn.ReLU()
        self.norm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x
    
class UpConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1):
        super(UpConvLayer, self).__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding)
        self.activation = nn.ReLU()
        self.norm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.upconv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x



class SegmentationNN(nn.Module):
    def __init__(self, num_classes=23, hp=None):
        super().__init__()
        self.hp = hp

        # Load pre-trained AlexNet
        alexnet = models.alexnet(pretrained=True)
        
        # Extract features from the pre-trained model (you can modify this based on your needs)
        self.features = alexnet.features

        # Define additional layers for segmentation
        # self.conv1 = nn.Conv2d(256, 128, kernel_size=3, padding=1)

        self.conv1 = ConvLayer(256, 128)
        self.conv2 = ConvLayer(128, 23)
        self.conv3 = ConvLayer(64, 64)
        self.conv4 = ConvLayer(64, num_classes)

        self.upconv1 = UpConvLayer(128, 23)

        self.upconv2 = UpConvLayer(128, 23)

        # Upsample to get back to the original size
        self.upsample1 = nn.Upsample(size=(30, 30), mode='bilinear', align_corners=False)
        self.upsample2 = nn.Upsample(size=(60, 60), mode='bilinear', align_corners=False)
        self.upsample3 = nn.Upsample(size=(120, 120), mode='bilinear', align_corners=False)
        self.upsample4 = nn.Upsample(size=(240, 240), mode='bilinear', align_corners=False)

    def forward(self, x):
        # Encoder
        x = self.features(x)
        

        # Additional layers for segmentation
        
        x = self.conv1(x)
        # x = self.relu1(x)

        # x = self.conv2(x)
        # x = self.relu2(x)

        # x = self.conv3(x)

        # Upsample to get back to the original size
        x = self.upsample1(x)
        # x = self.conv2(x)
        x = self.upsample2(x)
        # x = self.conv3(x)
        x = self.upsample3(x)
        # x = self.upconv1(x)
        # x = self.conv4(x)
        # x = self.upconv1(x)
        x = self.upsample4(x)
        x = self.conv2(x)
        # print(x.shape)
        

        return x

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)

        
class DummySegmentationModel(nn.Module):

    def __init__(self, target_image):
        super().__init__()
        def _to_one_hot(y, num_classes):
            scatter_dim = len(y.size())
            y_tensor = y.view(*y.size(), -1)
            zeros = torch.zeros(*y.size(), num_classes, dtype=y.dtype)

            return zeros.scatter(scatter_dim, y_tensor, 1)

        target_image[target_image == -1] = 1

        self.prediction = _to_one_hot(target_image, 23).permute(2, 0, 1).unsqueeze(0)

    def forward(self, x):
        return self.prediction.float()

if __name__ == "__main__":
    from torchinfo import summary
    summary(SegmentationNN(), (1, 3, 240, 240), device="cpu")