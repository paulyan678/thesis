import os
import nibabel as nib
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import torch


class BraTSDataset(Dataset):
    def __init__(self, root_dir, transform=None, slice_indices=None, rotation_angle=45):
        """
        Args:
            root_dir (str): Root directory containing the BraTS20 dataset folders.
            transform (callable, optional): Transformation to apply to each 2D slice.
            slice_indices (list of int, optional): List of slice indices to extract from each volume.
            rotation_angle (float): The angle by which to rotate the image for contrastive learning.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.slice_indices = slice_indices if slice_indices else [77]  # Default to slice 77
        self.rotation_angle = rotation_angle

        # List all patient folders
        self.patient_dirs = [
            os.path.join(root_dir, d) for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))
        ]

    def __len__(self):
        # Total number of slices = number of patients * number of slices per volume
        return len(self.patient_dirs) * len(self.slice_indices)

    def __getitem__(self, idx):
        # Determine which patient and slice to use
        patient_idx = idx // len(self.slice_indices)  # Which patient folder
        slice_idx = self.slice_indices[idx % len(self.slice_indices)]  # Which slice index

        # Get the path to the current patient's folder
        patient_dir = self.patient_dirs[patient_idx]

        # Load the FLAIR image (modify if you'd like to use other modalities)
        flair_path = os.path.join(patient_dir, os.path.basename(patient_dir) + "_flair.nii.gz")
        flair_image = nib.load(flair_path).get_fdata()

        # Extract the specified slice
        slice_data = flair_image[:, :, slice_idx]

        # Normalize the slice to [0, 1]
        slice_data = (slice_data - np.min(slice_data)) / (np.max(slice_data) - np.min(slice_data))
        slice_data = (slice_data * 255).astype(np.uint8)

        # Convert to PIL Image for further transformations
        slice_image = Image.fromarray(slice_data)

        # Create the rotated version
        rotated_image = slice_image.rotate(self.rotation_angle, resample=Image.BILINEAR)

        # Apply transformations (if provided)
        if self.transform:
            original_image = self.transform(slice_image)  # First view
            rotated_image = self.transform(rotated_image)  # Second view
        else:
            original_image = transforms.ToTensor()(slice_image)
            rotated_image = transforms.ToTensor()(rotated_image)

        # Return a tuple of the two views for contrastive learning
        return (original_image, rotated_image)




if __name__ == "__main__":

    from torch.utils.data import DataLoader
    from torchvision import transforms

    # Define transformations
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),  # Convert grayscale to 3-channel RGB
        transforms.Resize((224, 224)),  # Resize to 224x224
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize
    ])

    # Instantiate the dataset
    train_dataset = BraTSDataset(
        root_dir="datasets/MICCAI_BraTS2020_TrainingData",
        transform=transform,
        slice_indices=[77],  # Specify slice indices
        rotation_angle=30,   # Rotate by 30 degrees
    )

    # Create the DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,  # Specify batch size
        shuffle=True,
        num_workers=4,
    )

    # Iterate over the DataLoader
    for i, (orginal, rotated) in enumerate(train_loader):
        print(f"Image 1 shape (im_q): {images[0].shape}")  # Should be [batch_size, 3, 224, 224]
        print(f"Image 2 shape (im_k): {images[1].shape}")  # Should be [batch_size, 3, 224, 224]
        break


