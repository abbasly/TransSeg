"""SegmentationNN"""
import torch
import torch.nn as nn
from torchvision import models

import torch.nn.functional as F


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
        self.resnet18 = models.resnet18(pretrained=True)

        # Encoder (downsampling path)
        self.encoder = nn.Sequential(
            self.resnet18.conv1,
            self.resnet18.bn1,
            self.resnet18.relu,
            self.resnet18.layer1,
            self.resnet18.layer2,
            self.resnet18.layer3,
            self.resnet18.layer4
        )

        # Define additional layers for segmentation
        # self.conv1 = nn.Conv2d(256, 128, kernel_size=3, padding=1)

        self.conv1 = ConvLayer(512, 23)
        self.conv2 = ConvLayer(128, 128)
        self.conv3 = ConvLayer(128, 128)
        self.conv4 = ConvLayer(128, num_classes)

        self.conv = ConvLayer(128, 23)

        self.upconv1 = UpConvLayer(128, 23)

        self.upconv2 = UpConvLayer(128, 23)

        # Upsample to get back to the original size
        self.upsample1 = nn.Upsample(size=(30, 30), mode='bilinear', align_corners=False)
        self.upsample2 = nn.Upsample(size=(60, 60), mode='bilinear', align_corners=False)
        self.upsample3 = nn.Upsample(size=(120, 120), mode='bilinear', align_corners=False)
        self.upsample4 = nn.Upsample(size=(240, 240), mode='bilinear', align_corners=False)

    def forward(self, x):
        # Encoder
        # x = self.features(x)

        x = self.encoder(x)
        # print(x.shape)
        

        # Additional layers for segmentation
        
        x = self.conv1(x)
        # x = self.relu1(x)

        # x = self.conv2(x)
        # x = self.relu2(x)

        # x = self.conv3(x)

        # Upsample to get back to the original size
        x = self.upsample1(x)
        # x = self.conv2(x)
        # x = self.conv2(x)
        x = self.upsample2(x)
        # x = self.conv3(x)
        x = self.upsample3(x)
        # x = self.conv3(x)
        # x = self.upconv1(x)
        # x = self.conv4(x)
        # x = self.upconv1(x)
        x = self.upsample4(x)
        # x = self.conv4(x)
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

# class SegmentationNN(nn.Module):
#     def __init__(self, num_classes=23, hp=None):
#         super().__init__()

#         # Encoder (contracting path)
#         self.encoder1 = self.conv_block(3, 64)
#         self.encoder2 = self.conv_block(64, 128)
#         self.encoder3 = self.conv_block(128, 256)
#         self.encoder4 = self.conv_block(256, 512)

#         # Bottleneck
#         self.bottleneck = self.conv_block(512, 1024)

#         # Decoder (expansive path)
#         self.decoder4 = self.conv_block(1024, 512)
#         self.decoder3 = self.conv_block(512, 256)
#         self.decoder2 = self.conv_block(256, 128)
#         self.decoder1 = self.conv_block(128, 64)

#         # Upsampling blocks
#         self.upsample4 = self.upsample_block(512, 256)
#         self.upsample3 = self.upsample_block(256, 128)
#         self.upsample2 = self.upsample_block(128, 64)
#         self.upsample1 = self.upsample_block(64, 32)

#         # Final output layer
#         self.out_conv = nn.Conv2d(32, 23, kernel_size=1)

#     def conv_block(self, in_channels, out_channels):
#         return nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2)
#         )

#     def upsample_block(self, in_channels, out_channels):
#         return nn.Sequential(
#             nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
#             nn.ReLU(inplace=True)
#         )

#     def forward(self, x):
#     # Encoder
#       enc1 = self.encoder1(x)
#       enc2 = self.encoder2(enc1)
#       enc3 = self.encoder3(enc2)
#       enc4 = self.encoder4(enc3)

#       # Bottleneck
#       bottleneck = self.bottleneck(enc4)

#       # Decoder
#       dec4 = self.upsample4(bottleneck)
#       dec4 = self.decoder4(torch.cat((dec4, enc4), dim=1))

#       dec3 = self.upsample3(dec4)
#       dec3 = self.decoder3(torch.cat((dec3, enc3), dim=1))

#       dec2 = self.upsample2(dec3)
#       dec2 = self.decoder2(torch.cat((dec2, enc2), dim=1))

#       dec1 = self.upsample1(dec2)
#       dec1 = self.decoder1(torch.cat((dec1, enc1), dim=1))

#       # Final output
#       out = self.out_conv(dec1)

#       return out

# class SegmentationNN(nn.Module):
#     def __init__(self, num_classes=23, hp=None):
#         super().__init__()

#         # Load pre-trained AlexNet
#         alexnet = models.alexnet(pretrained=True)

#         self.upsample1 = nn.Upsample(size=(14, 14), mode='bilinear', align_corners=False)
#         self.upsample2 = nn.Upsample(size=(30, 30), mode='bilinear', align_corners=False)
#         self.upsample3 = nn.Upsample(size=(60, 60), mode='bilinear', align_corners=False)
#         self.upsample4 = nn.Upsample(size=(120, 120), mode='bilinear', align_corners=False)
#         self.upsample5 = nn.Upsample(size=(240, 240), mode='bilinear', align_corners=False)


#         # Encoder (features) part of AlexNet
#         self.encoder1 = alexnet.features[0:3]  # conv1
#         self.encoder2 = alexnet.features[3:6]  # relu + maxpool
#         self.encoder3 = alexnet.features[6:8]  # conv2
#         self.encoder4 = alexnet.features[8:11]  # relu + maxpool
#         self.encoder5 = alexnet.features[11:13]  # conv3

#         # Decoder part
#         self.upconv4 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
#         self.decoder4 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        
#         self.upconv3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
#         self.decoder3 = nn.Conv2d(128, 64, kernel_size=3, padding=1)

#         self.upconv2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
#         self.decoder2 = nn.Conv2d(64, 32, kernel_size=3, padding=1)

#         self.upconv1 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
#         self.decoder1 = nn.Conv2d(32, 16, kernel_size=3, padding=1)

#         # Final output layer
#         self.out_conv = nn.Conv2d(16, num_classes, kernel_size=1)

#     def forward(self, x):
#         # Encoder
#         enc1 = self.encoder1(x)
#         enc2 = self.encoder2(enc1)
#         enc3 = self.encoder3(enc2)
#         enc4 = self.encoder4(enc3)
#         enc5 = self.encoder5(enc4)
#         # print(enc4.shape)
#         print(enc4.shape)
#         # print(enc5.shape)

#         # Decoder with skip connections
#         dec4 = self.upconv4(enc5)
#         dec4 = self.upsample1(dec4)
#         print(dec4.shape)
#         dec4 = F.interpolate(dec4, size=(enc4.size(2), enc4.size(3)), mode='bilinear', align_corners=False)
#         dec4 = torch.cat((dec4, enc4), dim=1)
#         dec4 = self.decoder4(dec4)
        
#         dec3 = self.upconv3(dec4)
#         dec3 = torch.cat((dec3, enc3), dim=1)
#         dec3 = self.decoder3(dec3)

#         dec2 = self.upconv2(dec3)
#         dec2 = torch.cat((dec2, enc2), dim=1)
#         dec2 = self.decoder2(dec2)

#         dec1 = self.upconv1(dec2)
#         dec1 = torch.cat((dec1, enc1), dim=1)
#         dec1 = self.decoder1(dec1)

#         # Final output
#         out = self.out_conv(dec1)

#         return out

#     def save(self, path):
#         """
#         Save model with its parameters to the given path. Conventionally the
#         path should end with "*.model".

#         Inputs:
#         - path: path string
#         """
#         print('Saving model... %s' % path)
#         torch.save(self, path)
        
# class SegmentationNN(nn.Module):
#     def __init__(self, num_classes=23, hp=None):
#         super().__init__()
#         self.hp = hp

#         # Load pre-trained AlexNet
#         alexnet = models.alexnet(pretrained=True)
        
#         # Extract features from the pre-trained model (you can modify this based on your needs)
#         self.features = alexnet.features

#         # Define additional layers for segmentation
#         # self.conv1 = nn.Conv2d(256, 128, kernel_size=3, padding=1)

#         self.conv1 = ConvLayer(256, 128)
#         self.conv2 = ConvLayer(128, 23)
#         self.conv3 = ConvLayer(64, 64)
#         self.conv4 = ConvLayer(64, num_classes)

#         self.upconv1 = UpConvLayer(128, 23)

#         self.upconv2 = UpConvLayer(128, 23)

#         # Upsample to get back to the original size
#         self.upsample1 = nn.Upsample(size=(30, 30), mode='bilinear', align_corners=False)
#         self.upsample2 = nn.Upsample(size=(60, 60), mode='bilinear', align_corners=False)
#         self.upsample3 = nn.Upsample(size=(120, 120), mode='bilinear', align_corners=False)
#         self.upsample4 = nn.Upsample(size=(240, 240), mode='bilinear', align_corners=False)

#     def forward(self, x):
#         # Encoder
#         x = self.features(x)
        

#         # Additional layers for segmentation
        
#         x = self.conv1(x)
#         # x = self.relu1(x)

#         # x = self.conv2(x)
#         # x = self.relu2(x)

#         # x = self.conv3(x)

#         # Upsample to get back to the original size
#         x = self.upsample1(x)
#         # x = self.conv2(x)
#         x = self.upsample2(x)
#         # x = self.conv3(x)
#         x = self.upsample3(x)
#         # x = self.upconv1(x)
#         # x = self.conv4(x)
#         # x = self.upconv1(x)
#         x = self.upsample4(x)
#         x = self.conv2(x)
#         # print(x.shape)
        

#         return x

#     def save(self, path):
#         """
#         Save model with its parameters to the given path. Conventionally the
#         path should end with "*.model".

#         Inputs:
#         - path: path string
#         """
#         print('Saving model... %s' % path)
#         torch.save(self, path)

        

        
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