import time
import random
from PIL import Image
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from eval import evaluate
from detect import detect
from model import SSD300, MultiBoxLoss
from datasets import PascalVOCDataset
from clearml import Task
from utils import *

# Data parameters
data_folder = "./"  # folder with data files
keep_difficult = True  # use objects considered difficult to detect?

# Model parameters
# Not too many here since the SSD300 has a very specific structure
n_classes = len(label_map)  # number of different types of objects
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Learning parameters
checkpoint = None  # path to model checkpoint, None if none
batch_size = 10  # batch size
iterations = 1000  # number of iterations to train
workers = 2  # number of workers for loading data in the DataLoader
print_freq = 10  # print training status every __ batches
eval_freq = 1  # conduct eval every __ epochs
lr = 1e-3  # learning rate
decay_lr_at = [
    int(iterations * (80000 / 120000)),
    int(iterations * (100000 / 120000)),
]  # decay learning rate after these many iterations
decay_lr_to = 0.1  # decay learning rate to this fraction of the existing learning rate
momentum = 0.9  # momentum
weight_decay = 5e-4  # weight decay
grad_clip = None  # clip if gradients are exploding, which may happen at larger batch sizes (sometimes at 32) - you will recognize it by a sorting error in the MuliBox loss calculation
subsample_fraction = (
    0.1  # fraction of the dataset to use (default: 1.0, use full dataset)
)
cudnn.benchmark = True


def main():
    """
    Training.
    """
    global start_epoch, label_map, epoch, checkpoint, decay_lr_at

    task = Task.init(project_name="SSD300", task_name="test")
    logger = task.get_logger()
    # Initialize model or load checkpoint
    if checkpoint is None:
        start_epoch = 0
        model = SSD300(n_classes=n_classes)
        # Initialize the optimizer, with twice the default learning rate for biases, as in the original Caffe repo

        # Freeze specific prediction layers
        # Freeze specific auxiliary layers
        # model.freeze_aux_layers(["conv8_1", "conv8_2"])
        # model.freeze_pred_layers(["loc_conv4_3", "conf_conv4_3"])

        model.print_trainable_parameters()
        biases = []
        not_biases = []
        for param_name, param in model.named_parameters():
            if param.requires_grad:
                if param_name.endswith(".bias"):
                    biases.append(param)
                else:
                    not_biases.append(param)
        optimizer = torch.optim.SGD(
            params=[{"params": biases, "lr": 2 * lr}, {"params": not_biases}],
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
        )

    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint["epoch"] + 1
        print("\nLoaded checkpoint from epoch %d.\n" % start_epoch)
        model = checkpoint["model"]
        optimizer = checkpoint["optimizer"]

    # Move to default device
    model = model.to(device)
    criterion = MultiBoxLoss(priors_cxcy=model.priors_cxcy).to(device)

    # Custom dataloaders
    train_dataset = PascalVOCDataset(
        data_folder,
        split="train",
        keep_difficult=keep_difficult,
        subsample_fraction=subsample_fraction,
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=train_dataset.collate_fn,
        num_workers=workers,
        pin_memory=True,
    )  # note that we're passing the collate function here

    with open(os.path.join(data_folder, f"Fiftyone_TEST_images.json"), "r") as j:
        test_images = json.load(j)

    debug_img_paths = random.sample(test_images, 5)

    # Custom dataloaders
    val_dataset = PascalVOCDataset(
        data_folder,
        split="test",
        keep_difficult=keep_difficult,
        subsample_fraction=subsample_fraction,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=train_dataset.collate_fn,
        num_workers=workers,
        pin_memory=True,
    )  # note that we're passing the collate function here

    # Calculate total number of epochs to train and the epochs to decay learning rate at (i.e. convert iterations to epochs)
    # To convert iterations to epochs, divide iterations by the number of iterations per epoch
    # The paper trains for 120,000 iterations with a batch size of 32, decays after 80,000 and 100,000 iterations
    epochs = iterations // (len(train_dataset) // batch_size)
    decay_lr_at = [it // (len(train_dataset) // batch_size) for it in decay_lr_at]

    print(epochs)
    print(decay_lr_at)

    # Epochs
    for epoch in range(start_epoch, epochs):

        # Decay learning rate at particular epochs
        if epoch in decay_lr_at:
            adjust_learning_rate(optimizer, decay_lr_to)

        # One epoch's training
        train(
            train_loader=train_loader,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            epoch=epoch,
            logger=logger,
        )

        if epoch % eval_freq == 0:
            APs, mAP = evaluate(test_loader=val_loader, model=model)
            for label in labels:
                logger.report_scalar(
                    title=f"val/bbox_AP_{label}",
                    series=f"val/bbox_AP_{label}",
                    iteration=epoch,
                    value=APs[label],
                )

            logger.report_scalar(
                title="val/mAP", series="val/mAP", iteration=epoch, value=mAP
            )

            for img_path in debug_img_paths:
                original_image = Image.open(img_path, mode="r")
                original_image = original_image.convert("RGB")
                annotated_img = detect(
                    original_image,
                    model,
                    device,
                    min_score=0.2,
                    max_overlap=0.5,
                    top_k=10,
                )
                logger.report_image(
                    title="debug image",
                    series="debug image",
                    iteration=epoch,
                    image=annotated_img,
                )

        # Save checkpoint
        save_checkpoint(epoch, model, optimizer)


def train(train_loader, model, criterion, optimizer, epoch, logger):
    """
    One epoch's training.

    :param train_loader: DataLoader for training data
    :param model: model
    :param criterion: MultiBox loss
    :param optimizer: optimizer
    :param epoch: epoch number
    """
    model.train()  # training mode enables dropout

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss

    start = time.time()

    # Batches
    for i, (images, boxes, labels, _) in enumerate(train_loader):
        data_time.update(time.time() - start)

        # Move to default device
        images = images.to(device)  # (batch_size (N), 3, 300, 300)
        boxes = [b.to(device) for b in boxes]
        labels = [l.to(device) for l in labels]

        # Forward prop.
        predicted_locs, predicted_scores = model(
            images
        )  # (N, 8732, 4), (N, 8732, n_classes)

        # Loss
        loss = criterion(predicted_locs, predicted_scores, boxes, labels)  # scalar

        # Backward prop.
        optimizer.zero_grad()
        loss.backward()

        # Clip gradients, if necessary
        if grad_clip is not None:
            clip_gradient(optimizer, grad_clip)

        # Update model
        optimizer.step()

        losses.update(loss.item(), images.size(0))
        batch_time.update(time.time() - start)

        start = time.time()

        # Print status
        if i % print_freq == 0:
            print(
                "Epoch: [{0}][{1}/{2}]\t"
                "Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                "Data Time {data_time.val:.3f} ({data_time.avg:.3f})\t"
                "Loss {loss.val:.4f} ({loss.avg:.4f})\t".format(
                    epoch,
                    i,
                    len(train_loader),
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=losses,
                )
            )
            logger.report_scalar(
                title="Loss", series="loss", iteration=i, value=losses.val
            )
            logger.report_scalar(
                title="Loss", series="loss_avg", iteration=i, value=losses.avg
            )
            logger.report_scalar(
                title="Time", series="batch time", iteration=i, value=batch_time.val
            )
            logger.report_scalar(
                title="Time", series="data time", iteration=i, value=data_time.val
            )
            logger.report_scalar(
                title="Learning Rate",
                series="learning_rate",
                iteration=i,
                value=optimizer.param_groups[1]["lr"],
            )
    del (
        predicted_locs,
        predicted_scores,
        images,
        boxes,
        labels,
    )  # free some memory since their histories may be stored


if __name__ == "__main__":
    main()
