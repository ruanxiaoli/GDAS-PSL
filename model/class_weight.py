import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class WeightedBinaryCrossEntropy(nn.Module):
    def __init__(self, class_weights):
        super(WeightedBinaryCrossEntropy, self).__init__()
        self.class_weights = class_weights

    def forward(self, y_pred, y_true):
        # Calculate binary cross entropy
        loss = F.binary_cross_entropy_with_logits(y_pred, y_true.float(), reduction='none')

        # Apply weighting to the loss
        weighted_loss = loss * self.class_weights

        # Return the weighted loss
        return torch.sum(weighted_loss)

# Example usage
num_classes = 10  # Number of classes
batch_size = 32  # Batch size

# Create model
model = nn.Sequential(
    nn.Linear(num_classes, 1)
)

# Generate example data
x_train = torch.randn((100, num_classes))
y_train = torch.randint(0, 2, (100, num_classes)).float()

# Set some labels to -1 to simulate label masking
y_train = torch.where(torch.rand((100, num_classes)) > 0.5, y_train, -1)

# Calculate class weights (computed on entire dataset)
class_counts = torch.sum(y_train != -1, dim=0)
class_weights = torch.max(class_counts) / class_counts.float()

# Pass weights to loss function
criterion = WeightedBinaryCrossEntropy(class_weights)

# Define optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train model
epochs = 5
for epoch in range(epochs):
    optimizer.zero_grad()
    y_pred = model(x_train)
    loss = criterion(y_pred, y_train)
    loss.backward()
    optimizer.step()
    print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}')