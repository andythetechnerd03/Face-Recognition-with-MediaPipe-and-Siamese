import torch
from torch import nn
import onnxruntime as ort
import numpy as np


class SiameseNetwork(nn.Module):
    """
    The Siamese Network model, which is a CNN with 3 convolutional layers and 3 fully connected layers.
    The input will be two images, the anchor image and the reference image, both with the shape of (1,256,256).
    Yes, they are greyscale!
    The output will be a tuple of the outputs of the two branches of the model, each of size (1, 16).
    The model has been trained on AT&T Faces Database.
    """

    def __init__(self):
        super(SiameseNetwork, self).__init__()

        # Setting up the Sequential of CNN Layers
        self.cnn1 = nn.Sequential(
            nn.Conv2d(1, 96, kernel_size=11, stride=4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),

            nn.Conv2d(96, 256, kernel_size=5, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(256, 384, kernel_size=3, stride=1),
            nn.ReLU(inplace=True)
        )

        # Setting up the Fully Connected Layers
        self.fc1 = nn.Sequential(
            nn.Linear(384, 1024),
            nn.ReLU(inplace=True),

            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),

            nn.Linear(256, 16)
        )

    def forward_once(self, x):
        """
        Forward pass of the single branch of the model
        :param x: Image of size (1, 256, 256)
        :return: Output of size (1, 2)
        """
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output

    def forward(self, input1, input2):
        """
        Forward pass of the Siamese Network, receiving anchor and reference images
        :param input1: Anchor image of size (1, 256, 256), the image from the frame
        :param input2: Reference image of size (1, 256, 256), the image from the database
        :return: A tuple of the outputs of the two branches of the model
        """
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)

        return output1, output2


# Load the model


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert a PyTorch tensor to a NumPy array
    :param tensor:
    :return: numpy array
    """
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def load_model(format: str,
               model_path: str):
    """
    Load the model, either in PyTorch or ONNX format
    :param model_path:  str, the path to the model, must be either .pth or .onnx and align with the specified format
    :param format: str, either 'torch' or 'onnx'
    :return: Their respective format, either torch.nn.Module or onnxruntime.InferenceSession
    """
    if format == 'torch':
        return torch.load(model_path, map_location=torch.device('cpu'))
    elif format == 'onnx':
        return ort.InferenceSession(model_path)
