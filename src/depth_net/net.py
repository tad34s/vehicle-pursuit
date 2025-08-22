import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from projector import Projector


class DepthNetwork(torch.nn.Module):
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-7
    DROPOUT_RATE = 0.3

    def __init__(self, image_size, device):
        super().__init__()

        self.transforms = torchvision.models.EfficientNet_V2_S_Weights.IMAGENET1K_V1.transforms()
        self.features = torchvision.models.efficientnet_v2_s(
            weights=torchvision.models.EfficientNet_V2_S_Weights.IMAGENET1K_V1
        ).features

        self.features.requires_grad_(False)

        self.features_len = 1536 * 10 * 10

        self.predict = torch.nn.Sequential(
            nn.Flatten(),
            nn.BatchNorm1d(self.features_len),  # Added batch normalization
            nn.Dropout(self.DROPOUT_RATE),
            nn.Linear(self.features_len, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
        )

        self.optim = torch.optim.Adam(
            self.parameters(),
            lr=self.LEARNING_RATE,
            weight_decay=self.WEIGHT_DECAY,
        )

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optim, mode="min", factor=0.1, patience=3, cooldown=1, threshold=0.001
        )

        self.device = device
        self.input_shape = (3, *image_size)

        self.projector = Projector("src/depth_net/utils/Prometheus.obj", image_size, self.device)
        self.focal_length = self.projector.camera.focal_length[0][
            1
        ].item()  # focal length in image coordinates
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
        theta_raw = preds[:, 2]

        x_ratio = torch.tanh(x_raw)  # [-1, 1]
        y_dist = F.softplus(y_raw)  # ensure distance is positive
        theta = torch.tanh(theta_raw) * 180

        max_x = (y_dist - self.projector.CAMERA_POS[2]) * (
            (self.projector.SENSOR_WIDTH_MM) / (2 * self.projector.FOCAL_LENGTH_MM)
        )

        x_coord = x_ratio * max_x

        return torch.stack([x_coord, y_dist, theta], dim=1)

    def forward(self, img):
        img = img.view(-1, *self.input_shape)
        img = self.transforms(img)
        features = self.features(img)
        print(features.shape)
        return
        features = features.view(-1, self.features_len)
        preds = self.predict(features)
        preds = self.activation_fn(preds)

        return preds


if __name__ == "__main__":
    net = DepthNetwork((512, 512), torch.device("cpu"))
    position_raw = torch.tensor(
        [[10000, 30.0, 0, 1]],
        dtype=torch.float32,
    )
    position = net.activation_fn(position_raw)
    print(position)
    x = position[0][0].item()
    y = position[0][1].item()
    theta = position[0][2].item()
    net.projector.render_mask(x, y, theta)
