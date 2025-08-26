import datetime
import math
from copy import deepcopy
from pathlib import Path

import torch
from net import DepthNetwork
from tensorboard import program
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard.writer import SummaryWriter

from dataset import ActiveLearningDataset, MaskDataset, TestDataset


def launch_tensor_board(logs_location: Path) -> None:
    tb = program.TensorBoard()
    tb.configure(argv=[None, "--bind_all", "--logdir", str(logs_location)])
    url = tb.launch()
    print(f"Tensorflow listening on {url}")


def pretrain(net, dataset, writer, epochs=1) -> None:
    dataloader = DataLoader(dataset, batch_size=64, num_workers=4, shuffle=True)
    loss_fn = torch.nn.MSELoss()

    for epoch in range(epochs):
        epoch_loss = 0.0  # Reset each epoch
        total_samples = 0
        for batch in dataloader:
            x, _, _ = batch
            x = x.to(net.device)
            batch_size = x.shape[0]
            y = torch.tensor([[0.0, 10.0, 0.0]] * batch_size, device=net.device)
            y[:, 2] = torch.rand(batch_size, device=net.device) * 360 - 180
            y_hat = net.forward(x)
            loss = loss_fn(y_hat, y)

            net.optim.zero_grad()
            loss.backward()
            net.optim.step()

            epoch_loss += loss.item() * batch_size
            total_samples += batch_size

        avg_epoch_loss = epoch_loss / total_samples

        writer.add_scalar("Pretraining loss", avg_epoch_loss, epoch)


def train_step(net: DepthNetwork, training_loader, writer, epoch_number):
    running_cum_loss = 0.0
    losses = {}
    write_hist = False
    if epoch_number % 10 == 0:
        write_hist = True

    if write_hist:
        y_hat_agg = [[] for _ in range(3)]

    for i, data in enumerate(training_loader):
        x, ref_image, ids = data
        x = x.to(net.device)
        ref_image = ref_image.to(net.device)

        y_hat = net(x)
        loss = net.projector.loss(y_hat, ref_image)
        loss_mean = loss.mean()
        for j, loss_val in enumerate(loss):
            running_cum_loss += loss_val.item()
            losses[ids[j]] = loss_val.item()

        net.optim.zero_grad()
        loss_mean.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=20.0)
        grad_norm = net.gradient_norm
        if math.isnan(grad_norm):
            continue
        net.optim.step()
        writer.add_scalar("Gradient norm", grad_norm, epoch_number * len(training_loader) + i)

        if write_hist:
            y_hat_cpu = y_hat.detach().cpu()
            for c in range(3):  # Iterate over each channel
                y_hat_agg[c].append(y_hat_cpu[:, c])  # Collect values for channel c

    # After processing all batches, create and write histograms if write_hist is True
    if write_hist:
        for c in range(3):
            channel_values = torch.cat(y_hat_agg[c], dim=0)
            writer.add_histogram(f"hist_{epoch_number}/channel_{c}", channel_values, epoch_number)

    return losses


def validate_net(net: DepthNetwork, val_loader):
    running_cum_loss = 0.0
    for data in val_loader:
        x, ref_image, _ = data
        x = x.to(net.device)  # Move batch to GPU
        ref_image = ref_image.to(net.device)
        with torch.no_grad():
            y_hat = net(x)
            loss = net.projector.loss(y_hat, ref_image).sum()

        loss_sum = loss.item()
        running_cum_loss += loss_sum

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
    val_loader = DataLoader(val_dataset, batch_size=1, num_workers=4)
    i = 0
    for data in val_loader:
        x, ref_image, _ = data
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
    val_dataloader = DataLoader(val_dataset, batch_size=64, num_workers=4)
    train_dataloader = DataLoader(train_dataset, batch_size=64, num_workers=4)
    best_net = deepcopy(net)

    best_val_loss = float("inf")
    epochs_from_best = 0
    early_stopping = 40
    losses = None

    for epoch in range(epochs):
        net.train(True)
        losses = train_step(net, train_dataloader, writer, epoch)
        net.train(False)

        avg_loss = sum(x for x in losses.values()) / len(losses)
        writer.add_scalar("Training loss", avg_loss, epoch)

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

        writer.flush()

    return best_net


def get_unsure_examples(net: DepthNetwork, train_dataset, threshold=0.3):
    output = []
    train_load = DataLoader(train_dataset, batch_size=64, num_workers=4)
    for batch in train_load:
        x, y, ids = batch
        x = x.to(net.device)

        net.eval()
        with torch.no_grad():
            y_hat = net(x)
        loss = net.projector.loss(y_hat, y)
        for j, loss_val in enumerate(loss):
            if loss_val > threshold:
                output.append(ids[j].item())
    return output


def generate_dataset_dict(net, unsure_examples, train_dataset, writer):
    dataset = {}
    losses = []
    for id in unsure_examples:
        x, y = train_dataset.get_by_id(id)
        x = x.to(net.device)
        net.eval()
        with torch.no_grad():
            y_hat = net(x.unsqueeze(0))

        estimate_gt, loss = net.projector.optimize(y_hat, y.unsqueeze(0))
        dataset[id] = estimate_gt.detach().cpu()
        losses.append(loss.item())

    losses_tensor = torch.tensor(losses)
    writer.add_histogram("Fianl optimized loss", losses_tensor, 0)

    return dataset


def active_train_step(net: DepthNetwork, training_loader, writer, epoch_number):
    loss_fn = torch.nn.MSELoss()
    epoch_loss = 0.0
    for batch in training_loader:
        x, y = batch
        x = x.to(net.device)
        y = y.to(net.device)
        batch_size = x.shape[0]
        y_hat = net.forward(x)
        loss = loss_fn(y_hat, y)

        net.optim.zero_grad()
        loss.backward()
        net.optim.step()

        epoch_loss += loss.item() * batch_size

    return epoch_loss


def active_learn(net: DepthNetwork, train_dataset, val_dataset, writer, epochs=1) -> DepthNetwork:
    unsure_examples = get_unsure_examples(net, train_dataset)
    print(f"Generationg dataset for {len(unsure_examples)} entries")
    dataset_dict = generate_dataset_dict(net, unsure_examples, train_dataset, writer)
    new_dataset = ActiveLearningDataset(train_dataset, dataset_dict)

    train_dataloader = DataLoader(new_dataset, batch_size=64, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=64, num_workers=4)
    best_net = deepcopy(net)

    best_val_loss = float("inf")
    epochs_from_best = 0
    early_stopping = 40
    losses = None

    for epoch in range(epochs):
        net.train(True)
        loss = active_train_step(net, train_dataloader, writer, epoch)
        net.train(False)

        avg_loss = loss / len(unsure_examples)
        writer.add_scalar("Active Training loss", avg_loss, epoch)

        avg_val_loss = validate_net(net, val_dataloader) / len(val_dataset)
        writer.add_scalar("Active Training Validation loss", avg_val_loss, epoch)

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

        writer.flush()

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
    pretrain(net, train_dataset, writer, epochs=3)
    print("Fitting...")
    best_net = fit(net, train_dataset, val_dataset, writer, epochs=500)
    test_dataset = TestDataset(
        "dataset/images", "dataset/t_ref", val_dataset_ids, device, image_size
    )
    print("Active learning...")
    active_learn(net, train_dataset, val_dataset, writer, epochs=100)
    print("Testing against ground truth...")
    test_net(best_net, test_dataset, writer)
    visualize_predictions(best_net, val_dataset, writer)
    writer.flush()


if __name__ == "__main__":
    main()
