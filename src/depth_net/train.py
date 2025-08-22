import datetime
from copy import deepcopy
from pathlib import Path

import torch
from net import DepthNetwork
from tensorboard import program
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard.writer import SummaryWriter

from dataset import MaskDataset, TestDataset


def launch_tensor_board(logs_location: Path) -> None:
    tb = program.TensorBoard()
    tb.configure(argv=[None, "--bind_all", "--logdir", str(logs_location)])
    url = tb.launch()
    print(f"Tensorflow listening on {url}")


def pretrain(net, dataset, writer, epochs=1) -> None:
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

        writer.add_scalar("Pretraining loss", avg_epoch_loss, epoch)


def train_step(net, training_loader, writer, epoch_number):
    running_cum_loss = 0.0
    for i, data in enumerate(training_loader):
        x, ref_image = data
        x = x.to(net.device)  # Move batch to GPU
        ref_image = ref_image.to(net.device)

        y_hat = net(x)
        loss = net.projector.loss(y_hat, ref_image)

        net.optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=20.0)
        grad_norm = net.gradient_norm
        net.optim.step()
        writer.add_scalar("Gradient norm", grad_norm, epoch_number * len(training_loader) + i)

        last_mean_loss = loss.item()
        running_cum_loss += last_mean_loss * x.shape[0]

    return running_cum_loss


def validate_net(net: DepthNetwork, val_loader):
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


def test_net(net: DepthNetwork, test_dataset, writer):
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True, num_workers=4)
    errors = []
    images_dir = "projections"

    Path(images_dir).mkdir(exist_ok=True, parents=True)
    for i, data in enumerate(test_loader):
        x, t_ref = data
        x = x.to(net.device)
        t_ref = t_ref.to(net.device)
        with torch.no_grad():
            y_hat = net(x)
        position = y_hat[0]
        x = position[0].item()
        y = position[1].item()
        theta = position[2].item()

        # net.projector.render_mask(x, y, theta, file_name=f"projections/{i}.png")
        error = y_hat - t_ref
        errors.append(error.cpu())

    all_errors = torch.cat(errors, dim=0)

    mean = all_errors.mean(dim=0)  # Mean error [3]
    std = all_errors.std(dim=0)  # Standard deviation [3]
    quantiles = torch.quantile(all_errors, torch.tensor([0.1, 0.9]), dim=0)  # Shape [2, 3]
    q1 = quantiles[0]  # 25% quantile (1/4) [3]
    q2 = quantiles[1]  # 75% quantile (3/4) [3]

    # Print results (or return/store as needed)
    writer.add_text("Mean error", str(mean))
    writer.add_text("Std error", str(std))
    writer.add_text("10% quantile", str(q1))
    writer.add_text("90% quantile", str(q2))

    return


def visualize_predictions(best_net: DepthNetwork, val_dataset: MaskDataset, writer: SummaryWriter):
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True, num_workers=4)
    i = 0
    for data in val_loader:
        x, ref_image = data
        x = x.to(best_net.device)  # Move batch to GPU
        ref_image = ref_image.to(best_net.device)
        with torch.no_grad():
            y_hat = best_net(x)
            img = best_net.projector.visualize_prediction(y_hat, ref_image)
        writer.add_image(f"Prediction {i}", img)
        i += 1
        if i >= 10:
            break


def fit(net: DepthNetwork, train_dataset, val_dataset, writer, epochs=1) -> DepthNetwork:
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=True, num_workers=4)
    best_net = deepcopy(net)

    best_val_loss = 1000000.0
    epochs_from_best = 0
    early_stopping = 40

    for epoch in range(epochs):
        net.train(True)
        avg_loss = train_step(net, train_dataloader, writer, epoch) / len(train_dataset)
        writer.add_scalar("Training loss", avg_loss, epoch)
        net.train(False)
        avg_val_loss = validate_net(net, val_dataloader) / len(val_dataset)
        writer.add_scalar("Validation loss", avg_val_loss, epoch)

        net.scheduler.step(avg_val_loss)
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
    log_location = Path(__file__).parent / "runs"
    writer = SummaryWriter(log_location / datetime.datetime.now().strftime("%y-%m-%d %H%M%S"))
    launch_tensor_board(log_location)

    image_size = (256, 256)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_images_path = "dataset/images"
    available_ids = [int(x.name[:-4]) for x in Path(input_images_path).glob("*.png")]
    generator = torch.Generator().manual_seed(42)
    train_dataset_ids, val_dataset_ids = random_split(
        available_ids, [0.7, 0.3], generator=generator
    )

    train_dataset = MaskDataset(
        input_images_path, "dataset/masks", train_dataset_ids, device, image_size, flip=True
    )
    val_dataset = MaskDataset(
        input_images_path, "dataset/masks", val_dataset_ids, device, image_size
    )

    net = DepthNetwork(image_size, device)
    net.to(device)

    print("Pretraining...")
    pretrain(net, train_dataset, writer, epochs=6)
    print("Fitting...")
    best_net = fit(net, train_dataset, val_dataset, writer, epochs=500)

    test_dataset = TestDataset(
        "dataset/images", "dataset/t_ref", val_dataset_ids, device, image_size
    )
    print("Testing against ground truth...")
    test_net(best_net, test_dataset, writer)
    visualize_predictions(best_net, val_dataset, writer)


if __name__ == "__main__":
    main()
