from PIL import Image
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


class STRDataset(Dataset):
    def __init__(self, X, y, char2idx, max_label_len, label_encoder=None, transform=None):
        self.transform = transform
        self.im_paths = X
        self.im_labels = y
        self.char2idx = char2idx
        self.max_label_len = max_label_len
        self.label_encoder = label_encoder

    def __len__(self):
        return len(self.im_paths)

    def __getitem__(self, idx):
        im_path = self.im_paths[idx]
        im_label = self.im_labels[idx]

        im = Image.open(im_path).convert('RGB')
        if self.transform:
            im = self.transform(im)

        if self.label_encoder:
            encoded_labels, label_len = self.label_encoder(im_label, self.char2idx, self.max_label_len)

        return im, encoded_labels, label_len


def build_dataset(im_paths, im_labels):
    seed = 0
    val_size = 0.1
    test_size = 0.1

    X_train, X_val, y_train, y_val = train_test_split(
        im_paths,
        im_labels,
        test_size=val_size,
        random_state=seed,
        shuffle=True,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X_train,
        y_train,
        test_size=test_size,
        random_state=seed,
        shuffle=True,
    )

    return X_train, y_train, X_val, y_val, X_test, y_test
