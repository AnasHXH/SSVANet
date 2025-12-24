# test.py

import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
from models.SSVANet import SSVA_Net
import yaml

def test():
    # Load configuration
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    image_size = config['image_size']  # Read image size from config
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])

    # Define the test dataset
    class TestDataset(torch.utils.data.Dataset):
        def __init__(self, lq_dir, transform=None):
            self.lq_dir = lq_dir
            self.transform = transform
            self.lq_images = sorted(os.listdir(lq_dir))

        def __len__(self):
            return len(self.lq_images)

        def __getitem__(self, idx):
            lq_img_name = self.lq_images[idx]
            lq_img_path = os.path.join(self.lq_dir, lq_img_name)
            lq_image = Image.open(lq_img_path).convert("RGB")

            if self.transform:
                lq_image = self.transform(lq_image)

            return lq_image, lq_img_name  # Return the image tensor and its filename

    # Paths to your test images and the directory to save output images
    test_lq_dir = config['test_lq_dir']
    output_dir = os.path.join(config['output_dir'], 'test_results')
    os.makedirs(output_dir, exist_ok=True)

    # Create the test dataset and dataloader
    test_dataset = TestDataset(test_lq_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Initialize model
    model = SSVA_Net(**config['model'])
    model.load_state_dict(torch.load(config['weights_path'], map_location=device))
    model.to(device)
    model.eval()

    with torch.no_grad():
        for input_image, image_name in test_loader:
            input_image = input_image.to(device)
            output_image = model(input_image)
            output_image = torch.clamp(output_image, 0, 1)  # Ensure the pixel values are in [0, 1]
            output_image = output_image.cpu()

            # Save the output image
            save_image(output_image, os.path.join(output_dir, image_name[0]))

            print(f"Processed and saved: {image_name[0]}")

    print("All images have been processed and saved.")

if __name__ == "__main__":
    test()
