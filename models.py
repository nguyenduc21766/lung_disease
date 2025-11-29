import torch.nn as nn
import torchvision.models as models
from args import get_args


class MyModel(nn.Module):
    def __init__(self, backbone='resnet18', num_classes=2, pretrained=True):
        super(MyModel, self).__init__()

        # Load proper args if created inside other scripts
        try:
            args = get_args()
            num_classes = args.num_classes
        except:
            pass

        # Choose backbone
        if backbone == 'resnet18':
            try:
                model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
            except AttributeError:
                model = models.resnet18(pretrained=pretrained)

        elif backbone == 'resnet34':
            try:
                model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT if pretrained else None)
            except AttributeError:
                model = models.resnet34(pretrained=pretrained)

        else:  # resnet50
            try:
                model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
            except AttributeError:
                model = models.resnet50(pretrained=pretrained)

        # Modify final layer to match number of classes (2)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

        self.model = model

    def forward(self, x):
        return self.model(x)
