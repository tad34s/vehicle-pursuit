import torch
import torchvision
from projector import Projector


class DepthNetwork(torch.nn.Module):
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-7

    def __init__(self, image_size, device):
        super().__init__()

        self.alex_net_transorms = torchvision.models.AlexNet_Weights.IMAGENET1K_V1.transforms()
        self.features = torchvision.models.alexnet(
            weights=torchvision.models.AlexNet_Weights.IMAGENET1K_V1
        ).features
        self.extra = torch.nn.MaxPool2d((2, 2))
        self.predict = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(256 * 3 * 3, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 3),
        )

        self.optim = torch.optim.Adam(
            self.parameters(),
            lr=self.LEARNING_RATE,
            weight_decay=self.WEIGHT_DECAY,
        )

        self.device = device
        self.input_shape = (3, *image_size)

        self.projector = Projector("src/depth_net/utils/Prometheus.obj", image_size, self.device)

    def forward(self, img):
        img = img.view(-1, *self.input_shape)
        img = self.alex_net_transorms(img)
        features = self.features(img)
        features = self.extra(features)  # [B, 256, 3, 3]
        features = features.view(-1, 256 * 3 * 3)
        preds = self.predict(features)
        return preds
