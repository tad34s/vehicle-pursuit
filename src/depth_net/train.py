from net import DepthNetwork
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import MaskDataset


def fit(net, dataset, epochs=1) -> None:
    # train qnet
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    loss_sum = 0

    for epoch in range(epochs):
        print(f"Epoch {epoch}")
        for batch in tqdm(dataloader):
            x, ref_img = batch

            y_hat = net.forward(x)
            print(y_hat.shape)
            loss = net.projector.loss(y_hat, ref_img)
            # Backprop
            net.optim.zero_grad()
            loss.backward()
            net.optim.step()
            loss_sum += loss.item()
        print(f"Average loss: {loss_sum / len(dataset)}")

    return


def main():
    dataset = MaskDataset("dataset/images", "dataset/masks")
    net = DepthNetwork()
    fit(net, dataset, epochs=100)


if __name__ == "__main__":
    main()
