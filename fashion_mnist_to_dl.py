import torchvision
import matplotlib.pyplot as plt
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

COLOUR_CHANNEL_COUNT = 1
IMAGE_WIDTH = 28
IMAGE_HEIGHT = 28
MINI_BATCH_SIZE = 32 # constant for now

train_data = datasets.FashionMNIST(
    root="data", # where to download data to?
    train=True, # get training data (and not testing data)
    download=True, # download data if it doesn't exist on disk
    transform=ToTensor(), # images come as PIL format, we want to turn into Torch tensors
    target_transform=None # you can transform labels as well
)
test_data = datasets.FashionMNIST(root="data", train=False, download=True, transform=ToTensor())
CLASSIFICATION_CATEGORIES_COUNT = len(train_data.classes)
print(f"train_data.classes = {train_data.classes}\nHas {len(train_data.classes)} elements.")

# Turn datasets into iterables (batches), shuffeling train data every epoch (test data not)
train_dl = DataLoader(train_data, batch_size=MINI_BATCH_SIZE, shuffle=True)
test_dl = DataLoader(test_data, batch_size=MINI_BATCH_SIZE, shuffle=False)

print(f"train_dl_f_mnist.batch_size = {train_dl.batch_size}") 
print(f"len(next(iter(train_dl_f_mnist))) = {len(next(iter(train_dl)))}") 
print(f"next(iter(train_dl_f_mnist))[0].shape = {next(iter(train_dl))[0].shape}") 
print(f"[{MINI_BATCH_SIZE}, {COLOUR_CHANNEL_COUNT}, {IMAGE_WIDTH}, {IMAGE_HEIGHT}] = [MINI_BATCH_SIZE, COLOUR_CHANNEL_COUNT, IMAGE_WIDTH, IMAGE_HEIGHT]")
print(f"len(train_dl_f_mnist) = {len(train_dl)}, len(test_dl_f_mnist) = {len(test_dl)}")

image_index = 2 # index in the batch, 0 .. 31
train_features_batch, train_labels_batch = next(iter(train_dl))
print(f"Image shape: {train_features_batch[image_index].shape}")
plt.imshow(train_features_batch[image_index].squeeze(), cmap="gray") # image shape is [1, 28, 28] (colour channels, height, width)
plt.title(str(train_labels_batch[image_index].item())+" i.e. "+train_data.classes[train_labels_batch[image_index].item()]);