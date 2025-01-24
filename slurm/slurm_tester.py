import torch
import torch.nn as nn
import torch.optim as optim
import time
import os

# Define a simple sanity check model
class SanityCheckModel(nn.Module):
    def __init__(self):
        super(SanityCheckModel, self).__init__()
        self.fc1 = nn.Linear(100, 50)
        self.fc2 = nn.Linear(50, 10)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def check_slurm_env():
    """Logs SLURM environment variables for debugging."""
    slurm_vars = ["SLURM_JOB_ID", "SLURM_JOB_NAME", "SLURM_CPUS_PER_TASK",
                  "SLURM_NTASKS", "SLURM_MEM_PER_NODE", "SLURM_GPUS", "SLURM_JOB_NODELIST"]
    print("\nSLURM Environment Variables:")
    for var in slurm_vars:
        value = os.getenv(var, "Not Set")
        print(f"{var}: {value}")

def sanity_check():
    """Performs a simple forward and backward pass on a test model."""
    # Check for GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create a dummy dataset
    x = torch.randn(128, 100).to(device)  # Batch size 128, input size 100
    y = torch.randint(0, 10, (128,)).to(device)

    # Initialize model, loss, and optimizer
    model = SanityCheckModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop for sanity check
    model.train()
    start_time = time.time()
    for _ in range(5):  # Run a few iterations to test performance
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
    end_time = time.time()

    print(f"Sanity check completed. Final loss: {loss.item():.4f}")
    print(f"Training took {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    print("Starting SLURM Sanity Check Script...")
    check_slurm_env()
    sanity_check()
    print("Sanity check completed successfully!")
