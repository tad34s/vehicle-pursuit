import torch
import torchvision
from projector import Projector

from variables import IMAGE_SIZE


class DepthNetwork(torch.nn.Module):
    LEARNING_RATE = 1e-4

    def __init__(self):
        super().__init__()

        self.features = torchvision.models.alexnet(weights=True).features
        self.extra = torch.nn.MaxPool2d((2, 2))
        self.predict = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(256, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 3),
        )

        self.optim = torch.optim.Adam(
            self.parameters(),
            lr=self.LEARNING_RATE,
            weight_decay=1e-7,
        )

        self.device = next(self.parameters()).device
        self.input_shape = (3, *IMAGE_SIZE)

        self.projector = Projector("src/depth_net/utils/Prometheus.obj", self.device)

    def forward(self, img):
        img = img.view(-1, *self.input_shape)
        features = self.features(img)
        features = self.extra(features)
        print(features.shape)
        features = features.view(-1, 256)
        preds = self.predict(features)
        return preds
