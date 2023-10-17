import torch, torchvision
import torch.nn as nn

class BasicModel(nn.Module):
    def __init__(self, num_classes, num_ts_features):
        super(BasicModel, self).__init__()
        self.num_classes = num_classes
        self.num_ts_features = num_ts_features

        self.img_backbone = torchvision.models.convnext_tiny(weights='DEFAULT')
        self.img_backbone.classifier[2] = nn.Identity()

        self.classifier = nn.Sequential(
            nn.Linear(768 + num_ts_features, 512),
            nn.ELU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, 512),
            nn.ELU(), 
            nn.Dropout(p=0.5),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Dropout(p=0.25),
            nn.Linear(256, num_classes),
        )

    
    def forward(self, img, ts_features):

        img_features = self.img_backbone(img)

        if self.num_ts_features == 0:
            out = self.classifier(img_features)
        else:
            
            ts_features = ts_features.unsqueeze(1) if len(ts_features.shape) == 1 else ts_features

            features = torch.cat([img_features, ts_features], dim=1)
            out = self.classifier(features)

        return out

def get_basic_img_model(num_classes, device, num_ts_features=0, use_multi_gpu=True):

    model = BasicModel(num_classes, num_ts_features)

    if use_multi_gpu:
        model = torch.nn.DataParallel(model)

    return model.to(device)
