import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.distributed as dist
from pyspark import SparkContext
import torch.multiprocessing as mp
import matplotlib.pyplot as plt
from torch.multiprocessing import Manager

# ----------------- 1. 配置并初始化Spark -----------------
sc = SparkContext("local", "MNIST_DDP")

# ----------------- 2. 定义数据预处理函数 -----------------
def preprocess_data(record):
    if "label" in record:  # Skip header
        return None
    values = record.split(",")
    label = int(values[0])
    pixels = torch.tensor([int(pixel) for pixel in values[1:]], dtype=torch.float32).view(1, 28, 28)
    pixels = pixels / 255.0
    return label, pixels

def load_data_from_hdfs(file_path):
    rdd = sc.textFile(file_path)
    processed_rdd = rdd.map(preprocess_data)
    return processed_rdd

# ----------------- 3. 自定义数据集 -----------------
class MNISTDataset(Dataset):
    def __init__(self, rdd):
        self.data = [item for item in rdd.collect() if item is not None]
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        label, image = self.data[idx]
        return image, label

# ----------------- 4. CNN模型 -----------------
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# ----------------- 5. DDP初始化 -----------------
def init_process(rank, size, model, optimizer, data_loader, world_size, loss_list, accuracy_list):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    model.train()
    model = nn.parallel.DistributedDataParallel(model)

    # Training loop
    for epoch in range(2):
        epoch_loss = 0
        correct = 0
        total = 0
        for batch_idx, (data, target) in enumerate(data_loader):
            data, target = data.to("cpu"), target.to("cpu")
            optimizer.zero_grad()
            output = model(data)
            loss = nn.CrossEntropyLoss()(output, target)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            _, predicted = torch.max(output, 1)
            correct += (predicted == target).sum().item()
            total += target.size(0)

            if batch_idx % 10 == 0:
                print(f"Rank {rank}, Epoch {epoch}, Batch {batch_idx}, Loss {loss.item()}")

        # Record loss and accuracy at the end of the epoch in rank 0
        avg_loss = epoch_loss / len(data_loader)
        accuracy = 100 * correct / total


        loss_list.append(avg_loss)
        accuracy_list.append(accuracy)

        # Plot loss and accuracy for each rank, and save to a unique file
        if rank == 0:  # Only plot and save for rank 0
            plt.figure(figsize=(12, 5))
            plt.subplot(1, 2, 1)
            plt.plot(list(loss_list), label='Loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.title(f'Rank {rank} Training Loss')
            plt.legend()
            plt.savefig(f'loss_rank_{rank}.png')  # Save plot to file

            plt.subplot(1, 2, 2)
            plt.plot(list(accuracy_list), label='Accuracy')
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy (%)')
            plt.title(f'Rank {rank} Training Accuracy')
            plt.legend()
            plt.savefig(f'accuracy_rank_{rank}.png')  # Save plot to file
            plt.show()  # Show plot for rank 0
            plt.close()  # Close the plot to avoid memory issues

    dist.barrier()

    # Aggregate loss and accuracy across all ranks (e.g., average)
    #total_loss = torch.tensor(0.0).to("cpu")
    #total_accuracy = torch.tensor(0.0).to("cpu")
    # Aggregate loss and accuracy across all ranks (use total_loss/total_accuracy)
    total_loss = torch.tensor(epoch_loss).to("cpu")
    total_accuracy = torch.tensor(accuracy).to("cpu")

    #dist.reduce(total_loss, op=dist.ReduceOp.SUM, root=0)
    #dist.reduce(total_accuracy, op=dist.ReduceOp.SUM, root=0)

    # Use rank 0 as the destination for the reduce operation
    dist.reduce(total_loss, op=dist.ReduceOp.SUM, dst=0)  # Specify 'dst=0'
    dist.reduce(total_accuracy, op=dist.ReduceOp.SUM, dst=0)  # Specify 'dst=0'

    if rank == 0:
        # Average the total_loss and total_accuracy across all processes
        #avg_loss = total_loss.item() / world_size
        #avg_accuracy = total_accuracy.item() / world_size
        print(f"Final Training Loss: {total_loss.item() / world_size}")
        print(f"Final Training Accuracy: {total_accuracy.item() / world_size}%")
        #Average the total_loss and total_accuracy across all processes
        #avg_loss = total_loss.item() / world_size
        # avg_accuracy = total_accuracy.item() / world_size
        # print(f"Epoch {epoch} - Final Training Loss: {avg_loss:.4f}")
        # print(f"Epoch {epoch} - Final Training Accuracy: {avg_accuracy:.2f}%")

    dist.barrier()


# ----------------- 6. 计算准确率 -----------------
def calculate_accuracy(model, data_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in data_loader:
            output = model(data)
            _, predicted = torch.max(output, 1)
            correct += (predicted == target).sum().item()
            total += target.size(0)
    accuracy = 100 * correct / total
    return accuracy

# ----------------- 7. 设置模型和优化器 -----------------
def main():
    device = torch.device("cpu")
    mnist_train_rdd = load_data_from_hdfs("hdfs://localhost:9000/user/MNISTinCSV/mnist_train.csv")
    mnist_test_rdd = load_data_from_hdfs("hdfs://localhost:9000/user/MNISTinCSV/mnist_test.csv")

    train_dataset = MNISTDataset(mnist_train_rdd)
    test_dataset = MNISTDataset(mnist_test_rdd)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    model = SimpleCNN().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    world_size = 1  # Number of processes (CPU cores)

    # Using Manager to create a shared list for loss and accuracy
    manager = Manager()
    loss_list = manager.list()
    accuracy_list = manager.list()

    # Run DDP training using multiprocessing
    mp.spawn(init_process, args=(world_size, model, optimizer, train_loader, world_size, loss_list, accuracy_list), nprocs=world_size, join=True)

    # Final accuracy calculation (rank 0 will handle it)
    if 0 == 0:
        final_accuracy = calculate_accuracy(model, test_loader)
        print(f"Final Test Accuracy: {final_accuracy:.2f}%")

if __name__ == "__main__":
    main()
