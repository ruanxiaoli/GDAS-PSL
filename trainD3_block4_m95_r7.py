import random
from torchvision import transforms
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from dataset.dataset import Dataset
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from metrics import *
#from model.MSTLoc import MSTLoc
from model.MSTLoc_attention_95 import MSTLoc_attention
import ast

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

# Define the data augmentation
data_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
])
# seed
setup_seed(3407)

lr = 0.008
epoch = 200
batch_size = 32
theta = 0.1
epsilon = 1e-8

class_dict = {
    "Nuclear membrane": 0,
    "Cytoplasm": 1,
    "Vesicles": 2,
    "Mitochondria": 3,
    "Golgi Apparatus": 4,
    "Nucleoli": 0,
    "Plasma Membrane": 1,
    "Nucleoplasm": 0,
    "Endoplasmic Reticulum": 5
}




# Count label distribution in enhanced files

def label_distribution(csv_file):
    with open(csv_file, 'r+') as f:
        lines = f.readlines()[1:]
    gene = []
    for line in lines:
        gene.append(line.replace("\n", '').split(",")[1])
    gene_key = list(set(gene))

    # Initialize dictionary using list elements with empty list as values
    gene_dict = {key: [] for key in gene_key}
    for line in lines:
        gene = line.replace("\n", '').split(",")[1]
        categories = line.replace("\n", '').split(",")[-1].split(';')
        for category in categories:
            gene_dict[gene].append(category)
    label_distributions = [0, 0, 0, 0, 0, 0]
    count = 0
    for key in gene_key:
        gene_dict[key] = list(set(gene_dict[key]))
        count+=1
        for class_name in gene_dict[key]:
            label_distributions[class_dict[class_name]] += 1
    return label_distributions,count


def class_weight(csv_file):
    data_distribution = label_distribution(csv_file)[0]
    max_count = max(data_distribution)
    data_count = label_distribution(csv_file)[1]
    weight = []
    for i in data_distribution:
        weight.append(data_count / i)
    return torch.tensor(weight)

# Calculate proportions
def class_rate(csv_file):
    data_distribution = label_distribution(csv_file)[0]
    data_count = label_distribution(csv_file)[1]
    # max_count = max(data_distribution)
    rate = []
    for i in data_distribution:
        rate.append( i / data_count)
    rate = torch.tensor(rate)
#    print(rate)
    return rate


class WeightedBinaryCrossEntropy(nn.Module):
    def __init__(self, class_weights, alpha=0.7):
        super(WeightedBinaryCrossEntropy, self).__init__()
        self.class_weights = class_weights.to(device)  # Ensure class_weights are on the specified device
        self.alpha = alpha

    def forward(self, y_pred, y_true):
        # Calculate uncertainty weights for each class
        batch_size, num_classes = y_pred.size()
        
        # Transpose matrix so each column contains all samples of one class
        y_pred_transposed = y_pred.transpose(0, 1)
        
        # Calculate entropy for each class
        entropy = torch.zeros(num_classes).to(device)  # Ensure entropy is on the specified device
        for class_index in range(num_classes):
            probs = y_pred_transposed[class_index]  # Get prediction probabilities for all samples of class class_index
            entropy[class_index] = -torch.sum(probs * torch.log(probs + epsilon)) / batch_size
        
        # Calculate uncertainty weights (can use entropy directly or normalize further)
        uncertainty_weights = entropy / torch.mean(entropy)
        
        # Ensure class_weights and uncertainty_weights are on the same device
        combined_weights = self.alpha * self.class_weights + (1 - self.alpha) * uncertainty_weights
        
        # Initialize loss
        loss_value = torch.zeros(size=(y_pred.shape[1],)).to(device)
        
        # Calculate weighted binary cross entropy loss
        for index in range(y_pred.shape[0]):
            for class_index in range(y_pred.shape[1]):
                loss_value[class_index] += -(
                    y_true[index][class_index] * torch.log(epsilon + y_pred[index][class_index]) +
                    (1 - y_true[index][class_index]) * torch.log(1 - y_pred[index][class_index] + epsilon)) * \
                    combined_weights[class_index]
        
        loss_value = loss_value / y_pred.shape[0]
        return torch.sum(loss_value)

if __name__ == '__main__':
    log_file = open("./train_final_noFESM_95.log", "a")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = MSTLoc_attention()
    net.load_state_dict(torch.load("./model_save/D3_diffusion_fv3.pth"))
#    weight = class_weight("./utils/train_weak_add_1.csv")
    weight = class_weight("./utils/train_D3_diffusion_fv3.csv")

#    weight = class_weight("./utils/train_restore_3.csv")
    criterion = WeightedBinaryCrossEntropy(weight).to(device)
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.005)
    # Load the dataset with data augmentation
#    train_set = Dataset("./utils/train_weak_add_1.npy", transform=None)
    train_set = Dataset("./utils/train_D3_diffusion_fv3.npy", transform=None)

#    train_set = Dataset("./utils/train_restore_3.npy", transform=None)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
#    test_set = Dataset("./utils/test_weak_add_1.npy")
    test_set = Dataset("./utils/test_D3_diffusion_fv3.npy")

#    test_set = Dataset("./utils/test_restore_3.npy")

    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=4)

    # Define the learning rate scheduler
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoch, eta_min=0)
    net.to(device)
    # Best test set loss
    best_loss = 99999
    best_acc = 0
    warmup_epochs = 4
    warmup_lr = 1e-5  # Starting learning rate for warmup

    for epoch_index in range(epoch):
        if epoch_index < warmup_epochs:
            warmup_factor = (lr - warmup_lr) / warmup_epochs
            adjusted_lr = warmup_lr + warmup_factor * epoch_index*0.1
            for param_group in optimizer.param_groups:
                param_group['lr'] = adjusted_lr

        total_loss = 0
        batch_num = 0
#        net.train()
#        for img, label in train_loader:
#            img = img.to(device)
#            label = label.reshape(label.shape[0], -1).to(device)
#            # Clear gradients
#            optimizer.zero_grad()
#            # Forward propagation
#            outputs = net(img)
#            # Calculate loss
#            loss = criterion(outputs, label)
#            # Backward propagation
#            loss.backward()
#            # Update weights
#            optimizer.step()
#            total_loss += loss.item()
#            batch_num += 1
#            print(f"Training epoch {epoch_index + 1}, batch {batch_num}, loss: {loss.item()}")
#
#        scheduler.step()  # Update learning rate for the next epoch

        # Testing
        net.eval()
        test_loss = 0
        all_pred = []
        all_label = []
        for img, label in test_loader:
            img = img.to(device)
            label = label.reshape(label.shape[0], -1).to(device)
            # Forward propagation
            outputs = net(img)
            # Calculate loss
            loss = criterion(outputs, label)
            test_loss += loss.item()
            # Calculate metrics
            theta_pred = ((torch.max(outputs) - outputs) < theta).int()
            all_label.append(label)
            all_pred.append(theta_pred)

        all_label = torch.cat(all_label)
        all_pred = torch.cat(all_pred)
        metrics_values = calc_metrics(all_label, all_pred, mode="macro")
        example_acc = metrics_values['example_acc']
        print(f"Test set metrics: {metrics_values}")
        exit()
        if  example_acc > best_acc:
            # Save model
            torch.save(net.state_dict(), "./model_save/D3_diffusion_fv3.pth")
            best_acc = example_acc

        print(f"Training epoch {epoch_index + 1}, average training loss: {total_loss / batch_num}, test set metrics: {metrics_values}")
        log_file.write(
            f"Training epoch {epoch_index + 1}, average training loss: {total_loss / batch_num}, test set metrics: {metrics_values}")
        log_file.write('\n')
    log_file.close()