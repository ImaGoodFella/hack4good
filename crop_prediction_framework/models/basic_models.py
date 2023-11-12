import torch, torchvision
import torch.nn as nn

class BasicModel(nn.Module):
    def __init__(self, num_classes, num_ts_features):
        super(BasicModel, self).__init__()
        self.num_classes = num_classes
        self.num_ts_features = num_ts_features

        self.img_backbone = torchvision.models.convnext_tiny(weights='DEFAULT')
        final_layer_size = self.img_backbone.classifier[2].in_features
        self.img_backbone.classifier[2] = nn.Identity()

        self.classifier = nn.Sequential(
            nn.Linear(final_layer_size + num_ts_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=0.85),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=0.75),
            nn.Linear(256, num_classes),
        )

    
    def forward(self, img, ts_features):
        
        img_features = self.img_backbone(img)
        features = torch.cat([img_features, ts_features], dim=1)
        out = self.classifier(features)

        return out

def get_basic_model(num_classes, num_ts_features, device, use_multi_gpu=True):

    model = BasicModel(num_classes, num_ts_features)

    if use_multi_gpu:
        model = torch.nn.DataParallel(model)

    return model.to(device)
