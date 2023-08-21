# Define the parameters and constants
from torchvision.transforms import transforms

torch_model = "siamese_16.pth"
onnx_model = "Siamese_16.onnx"
image_path = "images"

# Transform the images to grayscale, resize to 256x256, and normalize
transformers = transforms.Compose([
                            transforms.ToPILImage(),
                            transforms.Grayscale(num_output_channels=1),
                            transforms.Resize((256, 256)),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5,), (0.5,))
                            ])
