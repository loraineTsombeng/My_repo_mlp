import numpy as np
import pathlib
import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision import transforms
import cv2
import os

from torchvision.datasets import MNIST
from torchvision import transforms

# EMNIST(
#     root="data2",
#     split="balanced",
#     train=True,
#     download=True
# )


def load_mnist_cnn():
    # load MNIST data from local npz file
    path = pathlib.Path(__file__).parent.absolute() / "data" / "mnist.npz"
    with np.load(path) as f:
        images, labels = f["x_train"], f["y_train"]

    # normalize to 0-1
    images = images.astype("float32") / 255.0

    # CNN-Format: (batch, channels, height, width)
    images = images.reshape(-1, 1, 28, 28)

    # labels to int64
    labels = labels.astype(np.int64)

    return images, labels


def load_emnist_cnn(train=True):
# load EMNIST balanced split data from torchvision.datasets
    transform = transforms.Compose([
        transforms.ToTensor(),                                                  # -> [0,1], shape (1,28,28)
        transforms.Lambda(lambda x: torch.rot90(x, 3, [1,2]).contiguous()),     # rotate 3x90 degrees = -90 degrees
        transforms.Lambda(lambda x: torch.flip(x, [2]))                         # flip horizontal
    ])

    dataset = MNIST(
        root="data",
        split="balanced",
        train=train,
        download=True,
        transform=transform
    )                           # dataset of PIL images and labels

    mapping = load_emnist_mapping()
    images = []
    labels = []
    for img, label in dataset:
        if label in mapping:       # Label existiert im Mapping
            char = mapping[label]
            if char.isdigit():     # nur Zahlen behalten
                images.append(img.numpy())
                labels.append(int(char))

    images = np.stack(images).astype("float32")   # (N,1,28,28)
    labels = np.array(labels, dtype=np.int64)

    return images, labels

def load_emnist_mapping(path = r"data/EMNIST/raw/emnist-balanced-mapping.txt"):
    mapping = {}

    with open(path) as f:
        for line in f:
            key, val = line.split()
            mapping[int(key)] = chr(int(val))

    return mapping

def batch_generator_augmented(images, labels, batch_size=64, shuffle=True, augment=True):
# generate batches with optional data augmentation

     # NumPy → Torch Tensor
    images = torch.from_numpy(images).float()   # (N,1,28,28)
    labels = torch.from_numpy(labels).long()    # (N,)
    
    indices = np.arange(len(images))
    if shuffle:
        np.random.shuffle(indices)

    # Augmentation definieren
    transform = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomAffine(0, translate=(0.1, 0.1))
    ])

    for start in range(0, len(images), batch_size):
        end = start + batch_size
        batch_idx = indices[start:end]
        batch_imgs = images[batch_idx]
        batch_lbls = labels[batch_idx]

        # augmentation if enabled
        if augment:
            batch_imgs = torch.stack([transform(img) for img in batch_imgs])

        yield batch_imgs, batch_lbls


def show_image(img):
    # img size (1, 28, 28) oder (28, 28)
    if img.ndim == 3:
        img = img[0]  # delete channel dimension

    plt.imshow(img, cmap="gray")
    plt.axis("off")
    plt.show()

def show_feature_maps(feature_maps):
#view all feature maps in a grid
    # feature_maps: (1, C, H, W)
    maps = feature_maps[0]          # delete batch dimension
    maps = maps.unsqueeze(1)        # (C, 1, H, W) for make_grid

    grid = torchvision.utils.make_grid(maps, nrow=8, padding=1) # make grid (C, 1, H, W) -> (1, H_grid, W_grid)
    grid = grid.permute(1, 2, 0).numpy()  # (H, W, 3)
    
    plt.imshow(grid)
    plt.show()

def prep_image(image_path):
    # load image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"{image_path} nicht gefunden")
    # normalize to 0-1
    img = img.astype("float32") / 255.0

    img = 1 - img  # invert colors

    # CNN-Format: (batch, channels, height, width)
    img = img.reshape(-1, 1, 28, 28)

    # uint8 -> float32 für Berechnungen
    img = img.astype(np.float32)
    
    return img

def get_activation(activations, name):
    def hook(model, input, output):
        activations[name] = output.detach()
    return hook

def normalize_per_channel(x):
    # x: (C, H, W)
    c, h, w = x.shape
    x = x.view(c, -1)

    min_vals = x.min(dim=1, keepdim=True)[0]
    max_vals = x.max(dim=1, keepdim=True)[0]

    x = (x - min_vals) / (max_vals - min_vals + 1e-6)
    return x.view(c, h, w)


def visualize_activations(x, name="layer"):
# Visualize feature maps or activations of a layer
    x = x.cpu()

    # CONV FEATURE MAPS
    if x.dim() == 4:  # (B, C, H, W)
        maps = x[0]
        maps = normalize_per_channel(maps)
        maps = maps.unsqueeze(1) #(num_maps, 1, H, W)

        grid = torchvision.utils.make_grid(
            maps,
            nrow=8,
            padding=1,
            pad_value=0.0
        )

        grid = grid.permute(1, 2, 0).numpy()

        h, w, _ = grid.shape
        plt.figure(figsize=(w / 120, h / 120), dpi=120)
        plt.imshow(grid, interpolation="nearest")
        plt.title(name)
        plt.axis("off")
        plt.show()

    # FC / VIEW LAYERS
    elif x.dim() == 2:  # (B, N)
        vec = x[0].numpy()
        vec = (vec - vec.min()) / (vec.max() - vec.min() + 1e-6)

        plt.figure(figsize=(10, 2))
        plt.imshow(vec[np.newaxis, :], aspect="auto", cmap="viridis")
        plt.colorbar(label="activation")
        plt.title(name)
        plt.yticks([])
        plt.xlabel("Neuron index")
        plt.show()

    else:
        print(f"{name}: unsupported shape {x.shape}")

def save_activations(x, name="layer", out_dir="outputs"):

    os.makedirs(out_dir, exist_ok=True)
    # layer_dir = os.path.join(out_dir, name)
    # os.makedirs(layer_dir, exist_ok=True)

    x = x.cpu()

    # CONV FEATURE MAPS
    if x.dim() == 4:  # (B, C, H, W)
        maps = x[0]
        maps = normalize_per_channel(maps) 
        maps = maps.unsqueeze(1) #(num_maps, 1, H, W)

        grid = torchvision.utils.make_grid(
            maps,
            nrow=8,
            padding=1,
            pad_value=0.0
        )

        grid = grid.permute(1, 2, 0).numpy()

        plt.imsave(os.path.join(out_dir, f"{name}.png"), grid)
        plt.close()

    # FC / VIEW LAYERS
    elif x.dim() == 2:  # (B, N)
        vec = x[0].numpy()
        vec = (vec - vec.min()) / (vec.max() - vec.min() + 1e-6)
        vec_img = np.tile(vec, (20, 1))
        
        plt.imsave(os.path.join(out_dir, f"{name}.png"), vec_img, cmap="viridis")
        plt.close()

    else:
        print(f"{name}: unsupported shape {x.shape}")

def save_feature_maps(feature_map, layer_name, out_dir="outputs"):
    os.makedirs(out_dir, exist_ok=True)
    layer_dir = os.path.join(out_dir, layer_name)
    os.makedirs(layer_dir, exist_ok=True)

    fmap = feature_map[0]  # Batch 0 → (C, H, W)

    for i in range(fmap.shape[0]):
        img = fmap[i].cpu().numpy()
        img = (img - img.min()) / (img.max() - img.min() + 1e-6)

        plt.imsave(
            os.path.join(layer_dir, f"channel_{i:03d}.png"),
            img,
            cmap="gray"
        )

