import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from projector import Projector


class DepthNetwork(torch.nn.Module):
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-7

    def __init__(self, image_size, device):
        super().__init__()

        self.alex_net_transorms = torchvision.models.AlexNet_Weights.IMAGENET1K_V1.transforms()

        # self.features = nn.Sequential(
        #     nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=3, stride=2),
        #     nn.Conv2d(64, 192, kernel_size=5, padding=2),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(192, 384, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(384, 256, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(256, 256, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True),
        # )

        self.features = torchvision.models.alexnet(
            weights=torchvision.models.AlexNet_Weights.IMAGENET1K_V1
        ).features

        self.predict = torch.nn.Sequential(
            nn.Flatten(),
            nn.BatchNorm1d(256 * 6 * 6),  # Added batch normalization
            nn.Linear(256 * 6 * 6, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 4),
        )

        self.optim = torch.optim.Adam(
            self.parameters(),
            lr=self.LEARNING_RATE,
            weight_decay=self.WEIGHT_DECAY,
        )

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optim, mode="min", factor=0.1, patience=3, cooldown=1, threshold=0.01
        )

        self.device = device
        self.input_shape = (3, *image_size)

        self.projector = Projector("src/depth_net/utils/Prometheus.obj", image_size, self.device)
        self.focal_length = self.projector.camera.focal_length[0][1].item()
        self.image_width = self.projector.image_size[0]
        self.cx = self.image_width / 2  # Principal point (image center)

    @property
    def gradient_norm(self):
        total_norm = 0
        for p in self.parameters():
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm**0.5
        return total_norm

    def activation_fn(self, preds):
        x_raw = preds[:, 0]
        y_raw = preds[:, 1]
        sin_raw = preds[:, 2]
        cos_raw = preds[:, 3]
        theta = torch.atan2(torch.tanh(sin_raw), torch.tanh(cos_raw))

        x_ratio = torch.tanh(x_raw)  # [-1, 1]
        y_dist = F.softplus(y_raw)  # ensure distance is positive

        # Calculate x coordinate using the physical constraint
        max_x = y_dist * ((self.image_width) / (2 * self.focal_length))
        x_coord = x_ratio * max_x

        return torch.stack([x_coord, y_dist, theta], dim=1)

    def forward(self, img):
        img = img.view(-1, *self.input_shape)
        img = self.alex_net_transorms(img)
        features = self.features(img)
        features = features.view(-1, 256 * 6 * 6)
        preds = self.predict(features)
        preds = self.activation_fn(preds)

        return preds
