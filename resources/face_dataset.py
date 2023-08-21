# Create Face Dataset
from torch.utils.data import Dataset, DataLoader
import os
import cv2
from torchvision.transforms import transforms

# Make a database of faces using DataLoader
class FaceDataset(Dataset):
    def __init__(self, root_dir, transform= None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = os.listdir(root_dir)
        self.labels = [os.path.splitext(filename)[0] for filename in os.listdir(root_dir)]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = cv2.imread(os.path.join(self.root_dir, self.images[idx]))
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

    def get_label(self):
        return self.labels[0]

    def get_image(self, idx):
        return self.images[idx]


# Initialize the database
def initialize_database(image_path: str,
                        transformers: transforms.Compose):
    """
    Initialize the database of faces.
    :param image_path: The path to the folder of all the faces in the database
    :param transformers: torch.transformers.Compose: The transformers to apply to the images
    """
    face_dataset = FaceDataset(image_path, transform=transformers)
    dataloader = DataLoader(face_dataset, batch_size=len(face_dataset), shuffle=False)
    database, labels = next(iter(dataloader))
    return database, labels
