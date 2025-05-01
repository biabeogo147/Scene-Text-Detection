import os
import time
import numpy as np
import torch
import torch.nn as nn
import torchvision
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
from recognition.build_vocab import build_vocab, encode, decode
from recognition.build_dataset import build_dataset, STRDataset
from recognition.network import CRNN


def plot_batch(images, labels):
    plt.figure(figsize=(10, 20))
    labels = decode(labels, idx2char)
    grid = torchvision.utils.make_grid(images, nrow=4, normalize=True)
    plt.imshow(np.transpose(grid, (1, 2, 0)))
    plt.tight_layout()
    plt.axis('off')
    plt.show()
    print(labels)


def evaluate(model, dataloader, criterion, device):
    model.eval()
    losses = []
    with torch.no_grad():
        for idx, (inputs, labels, labels_len) in enumerate(dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            labels_len = labels_len.to(device)

            outputs = model(inputs)
            logits_lens = torch.full(size=(outputs.size(1),), fill_value=outputs.size(0), dtype=torch.long).to(device)

            loss = criterion(outputs, labels, logits_lens, labels_len)
            losses.append(loss.item())

    loss = sum(losses) / len(losses)

    return loss

def fit(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    device,
    epochs,
    max_grad_norm=5,
):
    train_losses, val_losses = [], []

    for epoch in range(epochs):
        start = time.time()
        batch_train_losses = []

        model.train()
        for idx, (inputs, labels, labels_len) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            labels_len = labels_len.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            logits_lens = torch.full(size=(outputs.size(1),), fill_value=outputs.size(0), dtype=torch.long).to(device)

            loss = criterion(outputs, labels.cpu(), logits_lens.cpu(), labels_len.cpu())
            loss.backward()

            # Debug
            # print(f"Epoch {epoch}, Batch {idx}: Outputs = {outputs}")
            # print(f"Epoch {epoch}, Batch {idx}: Loss = {loss.item()}")

            # Gradient clipping with a configurable max norm to prevent gradient exploding
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            optimizer.step()

            batch_train_losses.append(loss.item())

        train_loss = sum(batch_train_losses) / len(batch_train_losses)
        train_losses.append(train_loss)

        val_loss = evaluate(model, val_loader, criterion, device)
        val_losses.append(val_loss)

        print(
            f"EPOCH {epoch + 1}:\tTrain loss: {train_loss:.4f}\tVal loss: {val_loss:.4f}\t\t Time: {time.time() - start:.2f} seconds"
        )

        scheduler.step()

    return train_losses, val_losses


def predict(model, im):
    model.eval()
    with torch.no_grad():
        outputs = model(im)
        print(outputs)


def decode_labels(encoded_sequences, idx2char, blank_char='-'):
    decoded_sequences = []
    for seq in encoded_sequences:
        decoded_label = []
        for idx, token in enumerate(seq):
            if token != 0: # Ignore padding (token=0)
                char = idx2char[token.item()]
                if char != blank_char:
                    decoded_label.append(char)
        decoded_sequences.append(''.join(decoded_label))
    return decoded_sequences


def test():
    sample_result = []

    for i in range(50):
        idx = np.random.randint(len(val_dataset))
        img, label, label_len = train_dataset[idx]
        img = img.to(device)
        label = label.to(device)
        label = decode_labels([label], idx2char)[0]
        logits = model(img.unsqueeze(0))

        pred_text = decode(logits.permute(1, 0, 2).argmax(2), idx2char)[0]

        sample_result.append((img, label, pred_text))

    fig = plt.figure(figsize=(17, 20))
    for i in range(50):
        ax = fig.add_subplot(10, 5, i + 1, xticks=[], yticks=[])

        img, label, pred_text = sample_result[i]
        img = img.cpu()
        title = f"Truth: {label} | Pred: {pred_text}"

        ax.imshow(img.permute(1, 2, 0), cmap="gray")
        ax.set_title(title)

    plt.show()


if __name__ == "__main__":
    im_paths, im_labels = [], []
    save_dir = './datasets/ocr_dataset'
    with open(os.path.join(save_dir, 'labels.txt'), 'r') as f:
        for label in f:
            im_labels.append(label.strip().split('\t')[1])
            im_paths.append(label.strip().split('\t')[0])

    data_transforms = {
        "train": transforms.Compose(
            [
                transforms.Resize((100, 420)),
                transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
                transforms.Grayscale(num_output_channels=1),
                transforms.GaussianBlur(3),
                transforms.RandomAffine(degrees=1, shear=1),
                transforms.RandomPerspective(distortion_scale=0.3, p=0.5, interpolation=3),
                transforms.RandomRotation(degrees=2),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        ),
        "val": transforms.Compose(
            [
                transforms.Resize((100, 420)),
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        ),
    }

    X_train, y_train, X_val, y_val, X_test, y_test = build_dataset(im_paths, im_labels)
    char2idx, idx2char, max_label_len, vocab_size, blank_char = build_vocab(im_paths, im_labels)

    train_dataset = STRDataset(
        X_train, y_train,
        char2idx=char2idx,
        max_label_len=max_label_len,
        label_encoder=encode,
        transform=data_transforms["train"],
    )

    val_dataset = STRDataset(
        X_val, y_val,
        char2idx=char2idx,
        max_label_len=max_label_len,
        label_encoder=encode,
        transform=data_transforms["val"],
    )

    test_dataset = STRDataset(
        X_test, y_test,
        char2idx=char2idx,
        max_label_len=max_label_len,
        label_encoder=encode,
        transform=data_transforms["val"],
    )

    train_batch_size = 64
    test_batch_size = train_batch_size * 2

    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=test_batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

    for images, labels, label_len in train_loader:
        plot_batch(images, labels)
        break

    hidden_size = 256
    n_layers = 3
    dropout = 0.2
    unfreeze_layers = 3
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = CRNN(vocab_size, hidden_size, n_layers, dropout, unfreeze_layers).to(device)

    epochs = 100
    lr = 1e-3
    weight_decay = 1e-5
    scheduler_step_size = epochs * 0.5

    criterion = nn.CTCLoss(blank=char2idx[blank_char], zero_infinity=True, reduction="mean")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step_size, gamma=0.1)

    train_losses, val_losses = fit(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        device,
        epochs,
    )

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))

    ax[0].plot(train_losses)
    ax[0].set_title('Train loss')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Loss')

    ax[1].plot(val_losses, color='green')
    ax[1].set_title('Val loss')
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Loss')

    plt.tight_layout()
    plt.show()

    val_loss = evaluate(model, val_loader, criterion, device)
    test_loss = evaluate(model, test_loader, criterion, device)

    print("Evaluation on val/test dataset")
    print("Val loss: ", val_loss)
    print("Test loss: ", test_loss)

    save_model_path = '../models/ocr_crnn.pt'
    torch.save(model.state_dict(), save_model_path)

    test()