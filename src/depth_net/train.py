import torch
from net import DepthNetwork
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import MaskDataset


def pretrain(net, dataset, epochs=1) -> None:
    batch_size = 64
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    y = torch.tensor([[0.0, 10.0, 0.0]] * batch_size, device=net.device)
    loss_fn = torch.nn.MSELoss()

    for epoch in range(epochs):
        epoch_loss = 0.0  # Reset each epoch
        total_samples = 0
        print(f"Epoch {epoch}")
        for batch in tqdm(dataloader):
            x, _ = batch
            y_hat = net.forward(x)
            loss = loss_fn(y_hat, y)

            net.optim.zero_grad()
            loss.backward()
            net.optim.step()

            epoch_loss += loss.item() * batch_size
            total_samples += batch_size

        avg_epoch_loss = epoch_loss / total_samples
        print(f"Average loss: {avg_epoch_loss}")


def fit(net, dataset, epochs=1) -> None:
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    for epoch in range(epochs):
        epoch_loss = 0.0  # Reset each epoch
        total_samples = 0
        print(f"Epoch {epoch}")
        for batch in tqdm(dataloader):
            x, ref_img = batch
            batch_size = x.size(0)
            y_hat = net.forward(x)
            loss = net.projector.loss(y_hat, ref_img)

            net.optim.zero_grad()
            loss.backward()
            net.optim.step()

            epoch_loss += loss.item() * batch_size
            total_samples += batch_size

        avg_epoch_loss = epoch_loss / total_samples
        print(f"Average loss: {avg_epoch_loss}")


def main():
    image_size = (128, 128)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = MaskDataset("dataset/images", "dataset/masks", device, image_size)
    net = DepthNetwork(image_size, device)
    net.to(device)
    pretrain(net, dataset, epochs=20)
    fit(net, dataset, epochs=100)


if __name__ == "__main__":
    main()
