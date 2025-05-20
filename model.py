import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import argparse
from torchvision import models
# Depthwise Separable Convolution for efficiency with optional dilation


from torchsummary import summary

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()
        self.atrous_block1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.atrous_block6 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=6, dilation=6, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.atrous_block12 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=12, dilation=12, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.atrous_block18 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=18, dilation=18, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.global_pooling = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        size = x.shape[2:]
        out1 = self.atrous_block1(x)
        out2 = self.atrous_block6(x)
        out3 = self.atrous_block12(x)
        out4 = self.atrous_block18(x)
        out5 = self.global_pooling(x)
        out5 = F.interpolate(out5, size=size, mode="bilinear", align_corners=False)
        x = torch.cat([out1, out2, out3, out4, out5], dim=1)
        x = self.conv1(x)
        return x

# Improved Lightweight Model with Pretrained Backbone
class LightweightResNet(nn.Module):
    def __init__(self, num_classes=21):
        super(LightweightResNet, self).__init__()
        # Use MobileNetV2 as the backbone
        self.backbone = models.mobilenet_v2(pretrained=False).features

        # Extract layers for skip connections
        self.low_level_features = self.backbone[:4]  # Output stride 4
        self.mid_level_features = self.backbone[4:7]  # Output stride 8
        self.high_level_features = self.backbone[7:]  # Output stride 16

        # ASPP module
        self.aspp = ASPP(1280, 256)

        # Adjusted Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(256 + 32, 256, kernel_size=3, padding=1, bias=False),  # Changed 96 to 32
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )

        self.classifier = nn.Conv2d(256, num_classes, kernel_size=1)

        
        self.align_layers = nn.ModuleList()
        student_feature_channels = [24, 32, 1280]  # Idea from Mobilenetv2
        teacher_feature_channels = [256, 512, 1024, 2048]  # From ResNet50

        for sf_channels, tf_channels in zip(student_feature_channels, teacher_feature_channels):
            if sf_channels != tf_channels:
                self.align_layers.append(nn.Conv2d(sf_channels, tf_channels, kernel_size=1))
            else:
                self.align_layers.append(nn.Identity())

    def forward(self, x):
        # My encoder with skip connections.
        x0 = x
        x1 = self.low_level_features(x0)  # Output stride 4
        x2 = self.mid_level_features(x1)  # Output stride 8
        x3 = self.high_level_features(x2)  # Output stride 16

        # ASPP
        x_aspp = self.aspp(x3)  # Output stride 16

        # Upsample and concatenate with mid-level features
        x_aspp = F.interpolate(x_aspp, size=x2.shape[2:], mode='bilinear', align_corners=False)
        x_cat = torch.cat([x_aspp, x2], dim=1)

        # Decoder
        x_dec = self.decoder(x_cat)

        # Final upsampling
        x_dec = F.interpolate(x_dec, size=x0.shape[2:], mode='bilinear', align_corners=False)

        # Classifier
        out = self.classifier(x_dec)

        features = [x1, x2, x3]

        return out, features





class TeacherModelWithFeatures(nn.Module):
    def __init__(self, teacher_model):
        super(TeacherModelWithFeatures, self).__init__()
        self.teacher_model = teacher_model
        self.features = []
        self._register_hooks()

    def forward(self, x):
        self.features = []
        output = self.teacher_model(x)
        return output['out'], self.features  # Return both logits and features


    def _register_hooks(self):
        def hook(module, input, output):
            self.features.append(output)

        # Register hooks on backbone layers of FCN-ResNet50
        self.teacher_model.backbone.layer1.register_forward_hook(hook)
        self.teacher_model.backbone.layer2.register_forward_hook(hook)
        self.teacher_model.backbone.layer3.register_forward_hook(hook)
        self.teacher_model.backbone.layer4.register_forward_hook(hook)
        # Optionally, register on classifier layers if needed
        # self.teacher_model.classifier.register_forward_hook(hook)


def get_model_by_name(model_name, num_classes=21):
    if model_name.lower() == "best":
        return models.segmentation.fcn_resnet50(weights=None, num_classes=num_classes)
    elif model_name.lower()=="light":
        return LightweightResNet()
    else:
        raise ValueError(f"Unknown model: {model_name}. Choose 'best'")


def parse_arguments():
    parser = argparse.ArgumentParser(description="Model Parameter Counter")
    parser.add_argument("--model_name", "-m", type=str, required=True, help="Name of the model to load")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    # Get the model
    try:
        model = get_model_by_name(args.model_name)
    except ValueError as e:
        print(e)
        exit(1)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model = LightweightResNet(num_classes=21).to("cuda" if torch.cuda.is_available() else "cpu")
    summary(model, input_size=(3, 256, 256))

    print(f"Total parameters in {args.model_name}: {total_params}")
    print(f"Trainable parameters in {args.model_name}: {trainable_params}")
    # Existing code for parameter counting...

    