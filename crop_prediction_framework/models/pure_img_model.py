import torch, torchvision
import torch.nn as nn

class ImgModel(nn.Module):
    def __init__(self, num_classes):
        super(ImgModel, self).__init__()
        self.num_classes = num_classes

        self.img_backbone = torchvision.models.convnext_tiny(weights='DEFAULT')

        try:
            final_layer_size = self.img_backbone.classifier[2].in_features
            self.img_backbone.classifier[2] = nn.Identity()
        except:
            final_layer_size = self.img_backbone.fc.in_features
            self.img_backbone.fc = nn.Identity()     

        self.classifier = nn.Sequential(
            nn.Linear(final_layer_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=0.85),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=0.75),
            nn.Linear(256, num_classes),
        )

    def forward(self, img, tmp=None):
        
        img_features = self.img_backbone(img)
        out = self.classifier(img_features)

        return out

def get_pure_img_model(num_classes):
    model = ImgModel(num_classes)
    return model
