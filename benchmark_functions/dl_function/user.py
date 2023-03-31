import logging
logging.getLogger("PIL.TiffImagePlugin").setLevel(logging.CRITICAL + 1)

from aiaccel.util import aiaccel
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models

import random
import numpy as np
import warnings


def torch_fix_seed(seed=42):
    # Python random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

    g = torch.Generator()
    g.manual_seed(seed)

    return g


# Train
def train_func(model, train_loader, optimizer, device, criterion):
    train_correct, train_loss, sum_of_train_data = 0, 0, 0
    model.train()

    for (inputs, labels) in train_loader:
        optimizer.zero_grad()
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs.data, 1)
        train_loss += loss.item() * inputs.size(0)
        train_correct += (predicted == labels).sum().item()
        sum_of_train_data += inputs.size(0)

    train_loss /= float(sum_of_train_data)
    train_acc = 100. * train_correct / float(sum_of_train_data)

    return train_loss, train_acc


# Validation and Test
def val_test_func(model, val_loader, device, criterion):
    val_correct, val_loss = 0, 0
    model.eval()

    with torch.no_grad():
        for (inputs, labels) in val_loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(inputs)
            val_loss += criterion(outputs, labels).item() * inputs.size(0)

            _, predicted = torch.max(outputs.data, 1)
            val_correct += (predicted == labels).sum().item()

    val_loss /= float(len(val_loader.dataset))
    val_acc = 100. * val_correct / float(len(val_loader.dataset))

    return val_loss, val_acc


def main(p):
    # Setup data augmentation
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    val_test_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    warnings.simplefilter('ignore')
    g = torch_fix_seed()

    # Setup dataset
    train_dataset = datasets.DTD(
        root='/opt/dataset', split='train', transform=train_transform)
    val_dataset = datasets.DTD(
        root='/opt/dataset', split='val', transform=val_test_transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=p["batch_size"],
        shuffle=True,
        num_workers=10,
        pin_memory=True,
        generator=g,
        drop_last=True,
        persistent_workers=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=p["batch_size"],
        shuffle=False,
        num_workers=10,
        pin_memory=True,
        generator=g,
        drop_last=False,
        persistent_workers=True,
    )

    # (train=False) Test dataset 10000
    test_dataset = datasets.DTD(
        root='/opt/dataset', split='test', transform=val_test_transform)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=p["batch_size"], shuffle=False, num_workers=10,
        pin_memory=True, generator=g,)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Setup model
    model = models.mobilenet_v2(pretrained=False)

    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, 47)

    model = model.to(device)

    model = nn.DataParallel(model)

    # Setup loss function
    criterion = nn.CrossEntropyLoss()

    # Setup optimizer
    epochs = 20
    optimizer = optim.SGD(model.parameters(), lr=p["lr"],
                          momentum=p["momentum"], weight_decay=p["lr"] * p["weight_lr_ratios"], nesterov=True)

    logger = logging.getLogger(__name__)

    for epoch in range(1, epochs + 1):

        # Train
        train_loss, train_acc = train_func(
            model, train_loader, optimizer, device, criterion)

        # Validation
        val_loss, val_acc = val_test_func(model, val_loader, device, criterion)
        logger.info(f"epoch[{epoch}/{epochs}] "
                    f"lr = {optimizer.param_groups[0]['lr']:.4f}, "
                    f"train acc: {train_acc:.4f} "
                    f"train loss: {train_loss:.4f}, val acc: {val_acc:.4f} "
                    f"val loss: {val_loss:.4f}")

    # Test
    test_loss, test_acc = val_test_func(model, test_loader, device, criterion)
    logger.info(f"test acc: {test_acc:.4f}, test loss: {test_loss:.4f}")

    # Return val error rate
    return 100. - val_acc


if __name__ == "__main__":
    run = aiaccel.Run()
    run.execute_and_report(main)
