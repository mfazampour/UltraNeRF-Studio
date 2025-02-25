import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm  # Import tqdm for progress bar
from skimage.measure import marching_cubes
import trimesh
import torch.nn.functional as F

# 1. Load the occupancy data from .npy file
class OccupancyDataset(Dataset):
    def __init__(self, occupancy_file, grid_size=(100, 100, 100), batch_size=2048):
        super().__init__()
        self.grid_size = grid_size
        self.occupancy_data = self.load_occupancy_data(occupancy_file).reshape(-1, 1)
        print(self.occupancy_data.shape)
        self.grid_points = self.create_grid()
        self.batch_size = batch_size

    def load_occupancy_data(self, file_path):
        """Load occupancy data from a file."""
        return np.load(file_path).astype(np.float32)  # Assume .npy file containing occupancy values

    def create_grid(self):
        """Create a 3D grid of coordinates."""
        x = np.linspace(0, 1, self.grid_size[0])
        y = np.linspace(0, 1, self.grid_size[1])
        z = np.linspace(0, 1, self.grid_size[2])
        grid = np.array(np.meshgrid(x, y, z, indexing='ij'))
        return grid.reshape(-1, 3)  # Shape: (N, 3)

    def __len__(self):
        return len(self.grid_points)

    def __getitem__(self, idx):
        point = self.grid_points[idx]  # (x, y, z) coordinates
        occupancy_value = self.occupancy_data[idx]  # Corresponding occupancy value
        return torch.tensor(point, dtype=torch.float32), torch.tensor(occupancy_value, dtype=torch.float32)


# 2. Define the 8-layer MLP network
class OccupancyNetwork(nn.Module):
    def __init__(self, D=8, W=128, input_ch=3, output_ch=6, skips=[4]):
        """ """
        super(OccupancyNetwork, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.skips = skips

        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)]
            + [
                nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W)
                for i in range(D - 1)
            ]
        )
        for l in self.pts_linears:
            nn.init.uniform_(l.weight, a=-0.05, b=0.05)
            nn.init.uniform_(l.bias, a=-0.05, b=0.05)

        self.output_linear = nn.Linear(W, 1)
        nn.init.uniform_(self.output_linear.weight, a=-0.05, b=0.05)
        nn.init.uniform_(self.output_linear.bias, a=-0.05, b=0.05)

    def forward(self, x):
        input = x
        h = input
        for i, l in enumerate(self.pts_linears):
            h = l(h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input, h], -1)

        outputs = F.sigmoid(self.output_linear(h))

        return outputs


# 3. Training setup with progress bar
def train(model, dataloader, optimizer, criterion, num_epochs=100):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        with tqdm(dataloader, desc=f'Epoch [{epoch + 1}/{num_epochs}]', unit='batch') as pbar:
            for data in pbar:
                pts = data[0]
                optimizer.zero_grad()
                # Forward pass
                output = model(pts)
                target = data[1]
                # Calculate loss (binary cross entropy for occupancy grid)
                loss = criterion(output, target)
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                pbar.set_postfix(loss=running_loss / (pbar.n + 1))  # average loss

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss:.16f}")
    torch.save(model.state_dict(), 'occupancy_network_mlp.pth')
# 4. Load data, initialize model, loss function, and optimizer
occupancy_file = 'occupancy_13.npy'  # Replace with your .npy file path
dataset = OccupancyDataset(occupancy_file)
dataloader = DataLoader(dataset, batch_size=100*100*100, shuffle=True)
def visualize_occupancy_with_marching_cubes(occupancy_output, grid_shape=None):
    """
    Visualize the occupancy network output using marching cubes for surface extraction.

    Args:
        occupancy_output (torch.Tensor): Output tensor from the occupancy network (flattened).
        grid_shape (tuple): Shape of the original 3D occupancy grid (D, H, W).
    """
    # Reshape the network output back to a 3D occupancy grid
    occupancy_grid = occupancy_output
    # Apply marching cubes to extract the surface mesh (isosurface extraction)
    verts, faces, _, _ = marching_cubes(occupancy_grid, level=0.5)  # 0.5 is the occupancy threshold
    # Create the 3D mesh using trimesh
    mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    # Show the mesh in 3D
    mesh.show()
visualize_occupancy_with_marching_cubes(dataset.occupancy_data.reshape((100, 100, 100)))
# Get the shape of the input (flattened 3D grid)
input_size = 3
# Initialize the model, optimizer, and loss function
model = OccupancyNetwork()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.BCELoss()  # Binary Cross-Entropy loss for occupancy data

# 5. Train the model
train(model, dataloader, optimizer, criterion, num_epochs=1000)

# 6. Save the model

