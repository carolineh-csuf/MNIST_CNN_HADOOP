import os
import time
import torch
import random
import logging
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
import matplotlib.pyplot as plt
import torchvision.models as models
from pyspark import SparkContext
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.multiprocessing import Manager

NUM_EPOCH = 1
WORLD_SIZE = 6 # Number of processes (CPU cores)
BATCH_SIZE = 64

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

class OptimizedCNN(nn.Module):
    def __init__(self):
        super(OptimizedCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def print_and_check_model_params(model, rank, world_size):
    """
    在训练过程中，打印并检查模型的参数是否在所有 rank 中一致。
    """
    # 获取模型的 state_dict（模型的参数）
    model_params = model.state_dict()

    # 收集所有 rank 的参数
    all_params = []

    # for name, param in model.named_parameters():
    #     print(f"Rank {rank}: {name} - Shape: {param.data.shape}")

    # # 在每个 rank 打印当前参数
    # if rank == 0:
    #     print(f"Rank {rank} - Parameter {name}:")
    #     for i in range(world_size):
    #         print(f"    Rank {i}: {gathered_params[i]}")

    #在 rank 0 比较参数是否一致
    if rank == 0:
        for name, gathered_params in zip(model_params.keys(), all_params):
            # 检查每个参数在所有 rank 中是否一致
            for i in range(1, world_size):
                assert torch.allclose(gathered_params[0], gathered_params[i], atol=1e-4), f"Mismatch in parameter {name} between rank 0 and rank {i}"
    #         # print(f"Parameter {name} is consistent across all ranks.")


def load_checkpoint(model, optimizer, filename="model_checkpoint.pth"):
    """
    加载模型的 checkpoint。只有 rank 0 加载 checkpoint，并且广播给其他 ranks。
    """
    if torch.distributed.get_rank() == 0:  # 确保只有 rank 0 加载 checkpoint
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        print(f"Checkpoint loaded from {filename} at epoch {epoch}")
    else:
        epoch = None
        loss = None
    # 广播 epoch 和 loss 给其他 ranks
    epoch = torch.tensor(epoch).to(torch.device("cpu"))
    loss = torch.tensor(loss).to(torch.device("cpu"))
    torch.distributed.broadcast(epoch, 0)
    torch.distributed.broadcast(loss, 0)
    return epoch.item(), loss.item()

def save_checkpoint(model, optimizer, epoch, loss, filename="model_checkpoint.pth"):
    """
    保存模型的 checkpoint。只有 rank 0 保存 checkpoint。
    """
    if torch.distributed.get_rank() == 0:  # 确保只有 rank 0 保存 checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }
        torch.save(checkpoint, filename)
        print(f"Checkpoint saved to {filename} at epoch {epoch}")


# ----------------- 5. DDP初始化 -----------------
def init_process(rank, model, optimizer, train_dataset, world_size, loss_dict, accuracy_dict):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # 为每个 rank 配置日志记录
    log_filename = f"rank_{rank}_training_log.txt"
    logging.basicConfig(filename=log_filename, level=logging.INFO)

    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    model = nn.parallel.DistributedDataParallel(model)

    # # 在训练开始之前加载 checkpoint
    # if rank == 0:
    #     # 仅 rank 0 加载 checkpoint
    #     start_epoch, loss = load_checkpoint(model, optimizer, filename="model_checkpoint.pth")
    #     print(f"Checkpoint loaded from rank {rank}, starting from epoch {start_epoch}, loss {loss}")

    # Using DistributedSampler for data parallelism
    #train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler)

    # 记录每个 worker 的开始时间
    start_time = time.time()  # 记录开始时间

    # Record loss for each rank
    rank_loss = []  # This will store the loss for each batch in this rank
    rank_accuracy = []  # List to store accuracy for each batch in this rank

    # Training loop
    for epoch in range(NUM_EPOCH):
        train_sampler.set_epoch(epoch)  # Ensure proper shuffling
        epoch_loss = 0
        correct = 0
        total = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to("cpu"), target.to("cpu")
            optimizer.zero_grad()
            output = model(data)
            loss = nn.CrossEntropyLoss()(output, target)

            loss.backward()
            optimizer.step()

            # 将每个 batch 的信息写入日志
            logging.info(f"Rank {rank}, Epoch {epoch}, Batch {batch_idx}, Loss {loss.item()}")

            # 在同步后打印并检查模型参数
            print_and_check_model_params(model, rank, world_size)

            epoch_loss += loss.item()
            _, predicted = torch.max(output, 1)
            correct += (predicted == target).sum().item()
            total += target.size(0)

            # 记录每个batch的损失
            #loss_list.append(loss.item())  # 将当前 batch 的损失添加到 loss_list
            # Record the loss for this batch
            rank_loss.append(loss.item())  # Append loss of this batch to rank's list
            accuracy = 100 * correct / total
            rank_accuracy.append(accuracy)

            if batch_idx % 10 == 0:
                print(f"Rank {rank}, PID {os.getpid()} with {len(data)} samples, Epoch {epoch}, Batch {batch_idx}, Loss {loss.item()} ")

        # 在每个 epoch 结束后同步
        dist.barrier()

        # Store the loss for this rank into the shared dictionary
        loss_dict[rank] = rank_loss  # Direct assignment without get_lock()
        accuracy_dict[rank] = rank_accuracy  # Store accuracy for each rank

    # 记录结束时间
    end_time = time.time()  # 记录结束时间
    training_time = end_time - start_time  # 计算总的训练时间

    # # 在训练完成后，只有 rank 0 保存 checkpoint
    # if rank == 0:
    #     torch.save(model.state_dict(), "model_checkpoint.pth")

    dist.barrier()

    # Aggregate loss and accuracy across all ranks
    total_loss = torch.tensor(epoch_loss).to("cpu")
    total_accuracy = torch.tensor(accuracy).to("cpu")
    dist.reduce(total_loss, op=dist.ReduceOp.SUM, dst=0)
    dist.reduce(total_accuracy, op=dist.ReduceOp.SUM, dst=0)

    if rank == 0:
        print(f"Final Training Loss: {total_loss.item() / world_size}")
        print(f"Final Training Accuracy: {total_accuracy.item() / world_size}%")
        print(f"Total Training Time for Rank {rank}: {training_time:.2f} seconds")
        logging.info(f"Final Training Loss: {total_loss.item() / world_size}")
        logging.info(f"Final Training Accuracy: {total_accuracy.item() / world_size}%")
        logging.info(f"Total Training Time for Rank {rank}: {training_time:.2f} seconds")

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

# ----------------- 8. 显示样本图像 -----------------
def show_sample_images(dataset, num_images=10):
    """
    Randomly selects and displays a few images from the dataset along with their labels.
    """
    # Randomly sample indices from the dataset
    sample_indices = random.sample(range(len(dataset)), num_images)
    images, labels = zip(*[dataset[i] for i in sample_indices])

    # Plot the sampled images
    plt.figure(figsize=(15, 5))
    for i, (image, label) in enumerate(zip(images, labels)):
        plt.subplot(1, num_images, i + 1)
        plt.imshow(image.squeeze(0), cmap="gray")  # Remove channel dimension and show grayscale
        plt.title(f"Label: {label}")
        plt.axis("off")
    plt.show()


def plot_loss_curve(loss_dict):
    """
    Plot the loss curve for each rank.
    """
    plt.figure(figsize=(12, 6))

    # Loop through each rank and plot its loss
    for rank, rank_loss in loss_dict.items():
        plt.plot(rank_loss, label=f"Rank {rank}")

    plt.xlabel("Batch")
    plt.ylabel("Loss")
    plt.title("Loss Curve for Each Rank")
    plt.legend()
    plt.show()

def plot_accuracy_curve(accuracy_dict):
    """
    Plot the accuracy curve for each rank.
    """
    plt.figure(figsize=(10, 6))

    # Loop through each rank and plot its accuracy
    for rank, accuracy in accuracy_dict.items():
        plt.plot(accuracy, label=f'Rank {rank}')

    plt.title("Training Accuracy for Each Rank")
    plt.xlabel("Batch Index")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True)
    plt.show()


def print_rdd_samples(rdd, num_samples=5):
    """
    Print the first few samples of the RDD.
    """
    samples = rdd.take(num_samples)
    for idx, sample in enumerate(samples):
        print(f"Sample {idx + 1}: {sample}")



# ----------------- 6. 设置模型和优化器 -----------------
def main():
    device = torch.device("cpu")
    mnist_train_rdd = load_data_from_hdfs("hdfs://localhost:9000/user/MNISTinCSV/mnist_train.csv")
    mnist_test_rdd = load_data_from_hdfs("hdfs://localhost:9000/user/MNISTinCSV/mnist_test.csv")

    # # Print first 5 samples from the training RDD
    # print("First 5 samples from the training dataset:")
    # print_rdd_samples(mnist_train_rdd)
    #
    # # Print first 5 samples from the test RDD
    # print("\nFirst 5 samples from the test dataset:")
    # print_rdd_samples(mnist_test_rdd)

    train_dataset = MNISTDataset(mnist_train_rdd)
    test_dataset = MNISTDataset(mnist_test_rdd)

    # 打印训练数据集的长度
    print(f"Number of training MNISTDataset: {len(train_dataset)}")

    # 打印测试数据集的长度
    print(f"Number of test MNISTDataset: {len(test_dataset)}")


    # Display some sample images from the training dataset
    show_sample_images(train_dataset, num_images=10)

    #train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    #model = SimpleCNN().to(device)
    #model = OptimizedCNN().to(device)

    # 选择 ResNet-18 作为模型
    model = models.resnet18(pretrained=True).to(device)
    # Modify the first convolutional layer to accept 1 input channel instead of 3
    model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)


    # 冻结ResNet的前几层（例如冻结前面的18层）
    for param in model.parameters():
        param.requires_grad = False

    # 替换最后一层以适应您的数据集类别数
    num_classes = 10
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    #optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    optimizer = optim.SGD(model.fc.parameters(), lr=0.001, momentum=0.9)  # 只优化最后一层


# # 尝试加载 checkpoint
    # start_epoch = 0
    # loss = None
    # if os.path.exists("model_checkpoint.pth"):
    #     start_epoch, loss = load_checkpoint(model, optimizer, filename="model_checkpoint.pth")
    # else:
    #     print("No checkpoint found, starting from scratch.")

    world_size = WORLD_SIZE  # Number of processes (CPU cores)

    # Using Manager to create a shared list for loss and accuracy
    manager = Manager()
    #loss_list = manager.list()
    loss_dict = manager.dict()
    #accuracy_list = manager.list()
    accuracy_dict = manager.dict()

    # Run DDP training using multiprocessing
    mp.spawn(init_process, args=(model, optimizer, train_dataset, world_size, loss_dict, accuracy_dict), nprocs=world_size, join=True)

    # Plot loss curve for each rank (only rank 0 should plot)
    if 0 == 0:
        plot_loss_curve(loss_dict)

    # Plot accuracy curve for each rank (only rank 0 should plot)
    if 0 == 0:
        plot_accuracy_curve(accuracy_dict)

    # Final accuracy calculation (rank 0 will handle it)
    if 0 == 0:
        final_accuracy = calculate_accuracy(model, test_loader)
        print(f"Final Test Accuracy: {final_accuracy:.2f}%")

if __name__ == "__main__":
    main()
