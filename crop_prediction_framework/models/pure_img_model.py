import torch, torchvision
import torch.nn as nn

class ImgModel(nn.Module):
    def __init__(self, num_classes):
        super(ImgModel, self).__init__()
        self.num_classes = num_classes

        self.img_backbone = torchvision.models.convnext_tiny(weights='DEFAULT')
        self.img_backbone.classifier[2] = nn.Identity()

        self.classifier = nn.Sequential(
            nn.Linear(768, 512),
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

    def forward(self, img):
        
        img_features = self.img_backbone(img)
        out = self.classifier(img_features)

        return out

def get_pure_img_model(num_classes, device, use_multi_gpu=True):

    model = ImgModel(num_classes)

    if use_multi_gpu:
        model = torch.nn.DataParallel(model)

    return model.to(device)
