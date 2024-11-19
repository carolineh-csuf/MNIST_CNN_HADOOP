import os
import time
import torch
import random
import logging
import subprocess
import torch.nn as nn
import seaborn as sns
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
import matplotlib.pyplot as plt
from datetime import datetime
from pyspark import SparkContext
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.multiprocessing import Manager
from sklearn.metrics import confusion_matrix



NUM_EPOCH = 1
WORLD_SIZE = 4 # Number of processes (CPU cores)
BATCH_SIZE = 64

# ----------------- 1. 配置并初始化Spark -----------------
sc = SparkContext("local", "MNIST_CNN_DDP")

# 设置日志级别为 ERROR
#sc.setLogLevel("OFF")
sc.setLogLevel("OFF")

# 本地文件路径和 HDFS 路径
local_checkpoint_path = "model_checkpoint.pth"
hdfs_checkpoint_folder = "/user/MNIST_Checkpoint"

# 为文件添加时间戳
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
file_name_with_timestamp = f"model_checkpoint_{timestamp}.pth"


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
                print(f"Parameter {name} is consistent across all ranks.")

# ----------------- 6. 保存与加载Checkpoint -----------------
def load_checkpoint(model, optimizer, filename="model_checkpoint.pth"):
    """
    Load model checkpoint if it exists. If the file does not exist, return a starting epoch and loss of 0.
    """
    epoch = 0
    loss = 0.0

    # Only rank 0 should load the checkpoint
    if torch.distributed.get_rank() == 0 and os.path.exists(filename) :
        checkpoint = torch.load(filename)
        print(f"Loaded checkpoint from {filename}")
        #epoch = checkpoint['epoch']
        epoch = checkpoint['epoch'] + 1  # 从保存的 epoch 后一个 epoch 开始训练
        loss = checkpoint.get('loss', 0.0)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Checkpoint loaded from epoch {epoch - 1}, loss {loss}")

        epoch_tensor = torch.tensor(epoch).to(torch.device("cpu"))
        loss_tensor = torch.tensor(loss).to(torch.device("cpu"))
        # Broadcast the epoch and loss to all ranks
        if torch.distributed.get_rank() == 0:
            #print("Rank 0 broadcasting epoch and loss.")
            torch.distributed.broadcast(epoch_tensor, 0)
            torch.distributed.broadcast(loss_tensor, 0)
        if torch.distributed.get_rank() == 0:
            print(f"Rank 0 broadcasting epoch {epoch_tensor.item()} and loss {loss_tensor.item()}")
        else:
            print(f"Rank {torch.distributed.get_rank()} received epoch {epoch_tensor.item()} and loss {loss_tensor.item()}")

    else:
        print("Checkpoint not found or rank != 0, starting from scratch.")
        # Skip broadcasting and return default values when no checkpoint is found
        epoch_tensor = torch.tensor(epoch).to(torch.device("cpu"))
        loss_tensor = torch.tensor(loss).to(torch.device("cpu"))

        # No broadcast needed, as all ranks will start from scratch
        # Only return the default epoch and loss without broadcasting
        return epoch_tensor.item(), loss_tensor.item()

    return epoch_tensor.item(), loss_tensor.item()

def save_checkpoint(model, optimizer, epoch, loss, filename="model_checkpoint.pth"):
    """
    Save the checkpoint with model state and optimizer state.
    Only rank 0 saves the checkpoint.
    保存模型和优化器的状态： 在训练过程中，通常会在每个 epoch 或某个间隔时保存模型的状态和优化器的状态。保存的信息通常包括：

    模型的参数 (model.state_dict())
    优化器的状态 (optimizer.state_dict())
    当前的 epoch 和 loss 等其他训练状态
    """
    if torch.distributed.get_rank() == 0:
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }
        torch.save(checkpoint, filename)
        print(f"Checkpoint saved to {filename} at epoch {epoch}")

def save_checkpoint_to_hdfs_via_subprocess(local_path, hdfs_folder):
    if not os.path.exists(local_path):
        print(f"本地文件 {local_path} 不存在，无法上传。")
        return


    try:
        # 确保 HDFS 目录存在
        subprocess.run(["hdfs", "dfs", "-mkdir", "-p", hdfs_folder], check=True)

        # 上传文件到 HDFS 并使用时间戳命名
        hdfs_full_path = os.path.join(hdfs_folder, file_name_with_timestamp)

        # 上传文件
        #subprocess.run(["hdfs", "dfs", "-put", "-f", local_path, hdfs_folder], check=True)
        subprocess.run(["hdfs", "dfs", "-put", "-f", local_path, hdfs_full_path], check=True)
        #print(f"文件已成功上传到 HDFS：{hdfs_folder}/{os.path.basename(local_path)}")
        print(f"文件已成功上传到 HDFS：{hdfs_full_path}")
        # 删除本地文件
        os.remove(local_path)
        print(f"本地文件 {local_path} 已删除。")
    except subprocess.CalledProcessError as e:
        print(f"HDFS 操作失败：{e}")
    except Exception as e:
        print(f"发生错误：{e}")

# ----------------- 5. DDP初始化 -----------------
def init_process(rank, model, optimizer, train_dataset, world_size, loss_dict, accuracy_dict):
    # 设置线程数
    #torch.set_num_threads(1)
    #os.environ["OMP_NUM_THREADS"] = "1"
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # 为每个 rank 配置日志记录
    log_filename = f"rank_{rank}_training_log.txt"
    logging.basicConfig(filename=log_filename, level=logging.INFO)

    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    model = nn.parallel.DistributedDataParallel(model)

    # 只有 rank 0 加载 checkpoint
    # To avoid multi process simulation deadlock. Here is only allow 1 worker to use checkpoint function.
    if rank == 0 :
    #if rank == 0 and WORLD_SIZE == 1 :
        start_epoch, loss = load_checkpoint(model, optimizer, filename="model_checkpoint.pth")
    else:
        start_epoch, loss = 0, 0.0  # 其他 rank 从头开始

    # Using DistributedSampler for data parallelism
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler,num_workers=1)

    dist.barrier()

    # 设置 epoch，确保数据加载器按正确顺序读取数据
    train_sampler.set_epoch(start_epoch)  # 设置正确的 epoch

    # 记录每个 worker 的开始时间
    start_time = time.time()  # 记录开始时间

    # Record loss for each rank
    rank_loss = []  # This will store the loss for each batch in this rank
    rank_accuracy = []  # List to store accuracy for each batch in this rank

    # Training loop
    for epoch in range(start_epoch, NUM_EPOCH):
        print(f"Rank {rank} - Epoch {epoch} starts.")
        # Debug: Check DataLoader
        #print(f"Rank {rank}: Number of batches in train_loader: {len(train_loader)}")

        #logging.info(f"Rank {rank} - Epoch {epoch} starts.")
        train_sampler.set_epoch(epoch)  # Ensure proper shuffling
        epoch_loss = 0
        correct = 0
        total = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to("cpu"), target.to("cpu")
            optimizer.zero_grad()

            # Debug: Check DataLoader
            #print(f"Rank {rank} - Batch ID {batch_idx} - data {data}")
            output = model(data)

            loss = nn.CrossEntropyLoss()(output, target)
            loss.backward()
            optimizer.step()

            # 将每个 batch 的信息写入日志
            #logging.info(f"Rank {rank}, Epoch {epoch}, Batch {batch_idx}, Loss {loss.item()}")

            # 在同步后打印并检查模型参数
            #print_and_check_model_params(model, rank, world_size)

            epoch_loss += loss.item()
            _, predicted = torch.max(output, 1)
            correct += (predicted == target).sum().item()
            total += target.size(0)

            # 记录每个batch的损失
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

        # Save checkpoint only for rank 0
        if rank == 0:
            save_checkpoint(model, optimizer, epoch, loss.item(), filename="model_checkpoint.pth")

    # 记录结束时间
    end_time = time.time()  # 记录结束时间
    training_time = end_time - start_time  # 计算总的训练时间


    dist.barrier()

    # # 使用指数加权平均来平滑损失
    # alpha = 0.1  # 平滑系数，越小越平滑
    # smoothed_loss = 0
    # for epoch_losses in loss_dict.values():
    #     for loss in epoch_losses:
    #         smoothed_loss = alpha * loss + (1 - alpha) * smoothed_loss
    # print(f"Smoothed Final Loss: {smoothed_loss}")
    # 在每个 rank 上计算局部损失和准确率

    # Aggregate loss and accuracy across all ranks
    total_loss = torch.tensor(epoch_loss).to("cpu")
    total_accuracy = torch.tensor(accuracy).to("cpu")
    dist.reduce(total_loss, op=dist.ReduceOp.SUM, dst=0)
    dist.reduce(total_accuracy, op=dist.ReduceOp.SUM, dst=0)



    if rank == 0:
         # 格式化输出训练结果
        print(f"\n{'='*53}")
        # 仅在 world_size > 1 时打印并显示 Parallel Training 和 World Size
        if world_size > 1:
            print(f"Training Results ：Parallel Training - World Size: {world_size}")
        else:
            print("Training Results ：Single Node Training")
        print(f"{'='*53}")
        print(f"Total Epochs: {NUM_EPOCH}")
        print(f"Batch Size: {BATCH_SIZE}")
        print(f"\nFinal Training Loss:       {total_loss.item() / world_size:.4f}")
        print(f"Final Training Accuracy:   {total_accuracy.item() / world_size:.2f}%")
        print(f"Total Training Time:       {training_time:.2f} seconds")
        print(f"{'='*53}")

        # 记录日志
        logging.info(f"\n{'='*53}")
        if world_size > 1:
            logging.info(f"Training Results ：Parallel Training - World Size: {world_size}")
        else:
            logging.info("Training Results ：Single Node Training")
        logging.info(f"Training Results (Parallel Training) - World Size: {world_size}")
        logging.info(f"{'='*53}")
        logging.info(f"Total Epochs: {NUM_EPOCH}")
        logging.info(f"Batch Size: {BATCH_SIZE}")
        logging.info(f"\nFinal Training Loss:       {total_loss.item() / world_size:.4f}")
        logging.info(f"Final Training Accuracy:   {total_accuracy.item() / world_size:.2f}%")
        logging.info(f"Total Training Time:       {training_time:.2f} seconds")
        logging.info(f"{'='*53}")

    if rank == 0:
        save_checkpoint_to_hdfs_via_subprocess(local_checkpoint_path, hdfs_checkpoint_folder)

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


from sklearn.metrics import confusion_matrix
import seaborn as sns

# ----------------- 6. 计算混淆矩阵 -----------------
def plot_confusion_matrix(model, data_loader, num_classes=10):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for data, target in data_loader:
            output = model(data)
            _, predicted = torch.max(output, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(target.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))

    # 使用 seaborn 绘制混淆矩阵
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=list(range(num_classes)), yticklabels=list(range(num_classes)))
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()

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

    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    #model = SimpleCNN().to(device)
    model = OptimizedCNN().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    world_size = WORLD_SIZE  # Number of processes (CPU cores)

    # Using Manager to create a shared list for loss and accuracy
    manager = Manager()
    loss_dict = manager.dict()
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
        # Plot confusion matrix after final accuracy
        plot_confusion_matrix(model, test_loader)

    # 在主函数最后停止 SparkContext
    sc.stop()

if __name__ == "__main__":
    main()
