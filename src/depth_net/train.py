from copy import deepcopy
from pathlib import Path

import torch
from net import DepthNetwork
from torch.utils.data import DataLoader, random_split

from dataset import MaskDataset, TestDataset


def pretrain(net, dataset, epochs=1) -> None:
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)
    loss_fn = torch.nn.MSELoss()

    for epoch in range(epochs):
        epoch_loss = 0.0  # Reset each epoch
        total_samples = 0
        for batch in dataloader:
            x, _ = batch
            x = x.to(net.device)
            batch_size = x.shape[0]
            y = torch.tensor([[0.0, 10.0, 0.0]] * batch_size, device=net.device)
            y_hat = net.forward(x)
            loss = loss_fn(y_hat, y)

            net.optim.zero_grad()
            loss.backward()
            net.optim.step()

            epoch_loss += loss.item() * batch_size
            total_samples += batch_size

        avg_epoch_loss = epoch_loss / total_samples
        print(f"Epoch {epoch}, average loss: {avg_epoch_loss}")


def train_step(net, training_loader):
    running_cum_loss = 0.0
    for data in training_loader:
        x, ref_image = data
        x = x.to(net.device)  # Move batch to GPU
        ref_image = ref_image.to(net.device)

        y_hat = net(x)
        loss = net.projector.loss(y_hat, ref_image)

        net.optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1)
        net.optim.step()

        last_mean_loss = loss.item()
        running_cum_loss += last_mean_loss * x.shape[0]

    return running_cum_loss


def validate_net(net, val_loader):
    running_cum_loss = 0.0
    for data in val_loader:
        x, ref_image = data
        x = x.to(net.device)  # Move batch to GPU
        ref_image = ref_image.to(net.device)
        with torch.no_grad():
            y_hat = net(x)
            loss = net.projector.loss(y_hat, ref_image)

        last_mean_loss = loss.item()
        running_cum_loss += last_mean_loss * x.shape[0]

    return running_cum_loss


def test_net(net, test_dataset):
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True, num_workers=4)
    errors = []

    for data in test_loader:
        x, t_ref = data
        x = x.to(net.device)
        t_ref = t_ref.to(net.device)
        with torch.no_grad():
            y_hat = net(x)
        error = y_hat - t_ref
        errors.append(error.cpu())

    all_errors = torch.cat(errors, dim=0)

    mean = all_errors.mean(dim=0)  # Mean error [3]
    std = all_errors.std(dim=0)  # Standard deviation [3]
    quantiles = torch.quantile(all_errors, torch.tensor([0.1, 0.9]), dim=0)  # Shape [2, 3]
    q1 = quantiles[0]  # 25% quantile (1/4) [3]
    q2 = quantiles[1]  # 75% quantile (3/4) [3]

    # Print results (or return/store as needed)
    print(f"Mean error: {mean}")
    print(f"Std error: {std}")
    print(f"10% quantile: {q1}")
    print(f"90% quantile: {q2}")

    # Return statistics if needed
    return


def fit(net, train_dataset, val_dataset, epochs=1) -> DepthNetwork:
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=True, num_workers=4)
    train_loss = []
    val_loss = []
    best_net = deepcopy(net)

    best_val_loss = 1000000.0
    epochs_from_best = 0
    early_stopping = 10

    for epoch in range(epochs):
        net.train(True)
        avg_loss = train_step(net, train_dataloader) / len(train_dataset)
        train_loss.append(avg_loss)
        net.train(False)
        avg_val_loss = validate_net(net, val_dataloader) / len(val_dataset)
        val_loss.append(avg_val_loss)
        print(f"Epoch {epoch}, mean training loss: {avg_loss}, mean validation loss {avg_val_loss}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_net = deepcopy(net)
            epochs_from_best = 0
        else:
            epochs_from_best += 1

        # EARLY STOPPING
        if epochs_from_best > early_stopping:
            print("Early stopping now")
            return best_net

    return best_net


def main():
    image_size = (128, 128)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_images_path = "dataset/images"
    available_ids = [int(x.name[:-4]) for x in Path(input_images_path).glob("*.png")]
    generator = torch.Generator().manual_seed(42)
    train_dataset_ids, val_dataset_ids = random_split(
        available_ids, [0.7, 0.3], generator=generator
    )

    train_dataset = MaskDataset(
        input_images_path, "dataset/masks", train_dataset_ids, device, image_size
    )
    val_dataset = MaskDataset(
        input_images_path, "dataset/masks", val_dataset_ids, device, image_size
    )

    net = DepthNetwork(image_size, device)
    net.to(device)

    print("Pretraining...")
    pretrain(net, train_dataset, epochs=3)
    print("Fitting...")
    best_net = fit(net, train_dataset, val_dataset, epochs=500)
    test_dataset = TestDataset(
        "dataset/images", "dataset/t_ref", val_dataset_ids, device, image_size
    )
    print("Testing against ground truth...")
    test_net(best_net, test_dataset)


if __name__ == "__main__":
    main()
