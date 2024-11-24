import os
import time
import torch
import random
import logging
import subprocess
import numpy as np
import torch.nn as nn
import seaborn as sns
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.multiprocessing import Manager
from torchsummary import summary
from datetime import datetime
from pyspark import SparkContext
from sklearn.metrics import confusion_matrix
#from pyspark import SparkConf, SparkContext
#from pyspark.sql import SparkSession
#import torchvision.transforms.functional as TF

NUM_EPOCH = 5
WORLD_SIZE = 1 # Number of processes (CPU cores)
BATCH_SIZE = 64

# 本地文件路径和 HDFS 路径
local_checkpoint_path = "model_checkpoint.pth"
hdfs_checkpoint_folder = "/user/MNIST_Checkpoint"

# 为文件添加时间戳
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
file_name_with_timestamp = f"model_checkpoint_{timestamp}.pth"

# 初始化 SparkContext 和 SparkSession
sc = SparkContext.getOrCreate()
#spark = SparkSession(sc)

# HDFS 路径
train_images_path = "hdfs://localhost:9000/user/MNIST/train-images-idx3-ubyte"
train_labels_path = "hdfs://localhost:9000/user/MNIST/train-labels-idx1-ubyte"
t10k_images_path = "hdfs://localhost:9000/user/MNIST/t10k-images-idx3-ubyte"
t10k_labels_path = "hdfs://localhost:9000/user/MNIST/t10k-labels-idx1-ubyte"


# 解析 MNIST 图像数据（28x28 像素）
def parse_mnist_image(raw_bytes):
    # 解析 IDX 文件中的图像数据
    data = np.frombuffer(raw_bytes, dtype=np.uint8)
    images = data[16:].reshape(-1, 28, 28)  # 跳过前 16 字节，并重塑为图像格式
    return images

# 解析 MNIST 标签数据
def parse_mnist_labels(raw_bytes):
    # 解析 IDX 文件中的标签数据
    data = np.frombuffer(raw_bytes, dtype=np.uint8)
    labels = data[8:]  # 跳过前 8 字节，剩下的是标签
    return labels

#解析图像和标签数据
# def get_mnist_data(train_images, train_labels):
#     train_image_data = train_images.take(1)[0][1]  # 读取第一个文件的字节流
#     train_label_data = train_labels.take(1)[0][1]  # 读取第一个标签的字节流
#
#     images = parse_mnist_image(train_image_data)  # 解析图像
#     labels = parse_mnist_labels(train_label_data)  # 解析标签
#
#     return images, labels

def print_image_by_column(image):
    """按列打印图像（28x28），每行打印28个像素值，确保列宽为3个字符"""
    for row in range(image.shape[0]):  # 对于每一行
        for col in range(image.shape[1]):  # 对于每一列
            # 使用格式化字符串来控制每列宽度为3个字符
            print(f"{image[row, col]:3d}", end=" ")  # 打印每个像素值，宽度为3
        print()  # 每行之后换行
    print("\n")  # 添加额外换行，用于分隔不同的图像

# 假设 train_images 和 train_labels 是存储 MNIST 图像和标签字节数据的 RDDs
def get_mnist_data(images_rdd, labels_rdd):
    # # 使用 RDD 的 map 操作平行处理图像数据和标签数据
    # images_rdd = train_images_rdd.map(lambda x: parse_mnist_image(x[1]))
    # labels_rdd = train_labels_rdd.map(lambda x: parse_mnist_labels(x[1]))
    # 使用 flatMap 来将每个文件中的多个图像展开
    images_rdd = images_rdd.flatMap(lambda x: parse_mnist_image(x[1]))  # 解析图像数据
    labels_rdd = labels_rdd.flatMap(lambda x: parse_mnist_labels(x[1]))  # 解析标签数据

    #print(images_rdd.take(1)[0])  # 输出前 1 个元素查看格式
    print_image_by_column(images_rdd.take(1)[0])  # 按列打印图像
    print(labels_rdd.take(1)[0])  # 输出前 1 个元素查看格式


    # 将图像和标签配对
    labeled_image_rdd = images_rdd.zip(labels_rdd)

    # 使用 reduceByKey 来统计每个标签的出现次数
    label_counts = labeled_image_rdd.map(lambda x: (x[1], 1))  # (标签, 1)
    label_count_rdd = label_counts.reduceByKey(lambda a, b: a + b)

    # Collect the result to print it
    label_count_result = label_count_rdd.collect()
    for label, count in label_count_result:
        print(f"Label {label}: {count} times")

    return labeled_image_rdd  # 返回配对的数据





class MNISTDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        # 将图像转换为 Tensor 并增加通道维度 (unsqueeze(0) adds a channel dimension)
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)  # Add channel dimension (1, 28, 28)
        image = image / 255.0  # Normalize to [0, 1]

        label = torch.tensor(label, dtype=torch.long)

        return image, label

# ----------------- 4. CNN模型 -----------------
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()

        # 第一层卷积: 输入通道 1 (灰度图)，输出通道 32，卷积核大小 3x3
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)

        # 第二层卷积: 输入通道 32，输出通道 64，卷积核大小 3x3
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)

        # 全连接层：输入 64 * 7 * 7（经过池化后的图像尺寸），输出 10（分类数量）
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

        # 池化层
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        # 第一个卷积层 + 池化
        x = self.pool(F.relu(self.conv1(x)))

        # 第二个卷积层 + 池化
        x = self.pool(F.relu(self.conv2(x)))

        # 展平为全连接层输入
        x = x.view(-1, 64 * 7 * 7)

        # 第一个全连接层 + ReLU
        x = F.relu(self.fc1(x))

        # 输出层
        x = self.fc2(x)
        return x

class ImprovedCNN2(nn.Module):
    def __init__(self):
        super(ImprovedCNN2, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)  # New conv layer
        self.bn3 = nn.BatchNorm2d(128)

        # Update the input size of fc1 based on the output size of the last conv layer
        self.fc1 = nn.Linear(128 * 3 * 3, 256)  # Adjusted size
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.5)
        self.swish = nn.SiLU()

    def forward(self, x):
        x = self.swish(self.bn1(self.conv1(x)))
        x = torch.max_pool2d(x, 2)
        x = self.swish(self.bn2(self.conv2(x)))
        x = torch.max_pool2d(x, 2)
        x = self.swish(self.bn3(self.conv3(x)))  # New conv layer
        x = torch.max_pool2d(x, 2)

        # Print the shape before flattening to determine the input size for fc1
        #print(x.shape)  # Debugging line to check the shape

        x = x.view(x.size(0), -1)  # Flatten the output

        # Now x has the correct size for fc1
        x = self.swish(self.fc1(x))  # Swish activation
        x = self.dropout(x)
        x = self.swish(self.fc2(x))
        x = self.fc3(x)  # Output layer
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
def init_process(rank, model, optimizer, train_dataset, world_size, loss_dict, accuracy_dict, batch_loss_dict):
    # 设置线程数
    #torch.set_num_threads(1)
    #os.environ["OMP_NUM_THREADS"] = "1"
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # 为每个 rank 配置日志记录
    log_filename = f"rank_{rank}_training_log.txt"
    logging.basicConfig(filename=log_filename, level=logging.INFO)

    # if rank == 0 :
    #     # 初始化 SummaryWriter，指定日志目录
    #     writer = SummaryWriter(log_dir='runs/experiment_name')

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

    # Initialize variables to track total loss and total samples
    total_loss = 0.0
    total_batches = 0
    total_samples = 0

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

            total_loss += loss.item()  # Accumulate total loss
            total_batches += 1  # Count total batches

            _, predicted = torch.max(output, 1)
            correct += (predicted == target).sum().item()
            total += target.size(0)

            # 记录每个batch的损失
            rank_loss.append(loss.item())  # Append loss of this batch to rank's list
            accuracy = 100 * correct / total
            rank_accuracy.append(accuracy)

            if batch_idx % 50 == 0:
                print(f"Rank {rank}, PID {os.getpid()} with {len(data)} samples, Epoch {epoch}, Batch {batch_idx}, Loss {loss.item()} ")

        # 在每个 epoch 结束后计算准确率
        epoch_accuracy = 100 * correct / total
        rank_accuracy.append(epoch_accuracy)  # 存储当前 epoch 的准确率

        # 在每个 epoch 结束后同步
        dist.barrier()

        # # Store the loss for this rank into the shared dictionary
        #loss_dict[rank] = rank_loss  # Direct assignment without get_lock()
        batch_loss_dict[rank] = rank_loss  # Store batch-wise losses

        # accuracy_dict[rank] = rank_accuracy  # Store accuracy for each rank

        # Store the loss for this rank into the shared dictionary
        loss_dict[rank] = epoch_loss / len(train_loader)  # 记录平均损失
        accuracy_dict[rank] = rank_accuracy  # 存储准确率


        # Save checkpoint only for rank 0
        if rank == 0:
            save_checkpoint(model, optimizer, epoch, loss.item(), filename="model_checkpoint.pth")

    # 记录结束时间
    end_time = time.time()  # 记录结束时间
    training_time = end_time - start_time  # 计算总的训练时间


    dist.barrier()

    # Aggregate loss across all ranks
    total_loss = torch.tensor(total_loss).to("cpu")
    dist.reduce(total_loss, op=dist.ReduceOp.SUM, dst=0)

    # 聚合损失和准确率
    total_accuracy = torch.tensor(epoch_accuracy).to("cpu")
    dist.reduce(total_accuracy, op=dist.ReduceOp.SUM, dst=0)



    if rank == 0:
        total_accuracy /= world_size
        avg_loss = total_loss.item() / total_batches  # Average loss across all batches

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
        print(f"Final Training Loss:       {avg_loss:.4f}")  # Report average loss
        print(f"Final Training Accuracy:   {total_accuracy.item():.2f}%")
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
        #logging.info(f"\nFinal Training Loss:       {total_loss.item() / world_size:.4f}")
        logging.info(f"Final Training Loss:       {avg_loss:.4f}")
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
# def show_sample_images(dataset, num_images=10):
#
#     # Randomly sample indices from the dataset
#     sample_indices = random.sample(range(len(dataset)), num_images)
#     images, labels = zip(*[dataset[i] for i in sample_indices])
#
#     # Plot the sampled images
#     plt.figure(figsize=(15, 5))
#     for i, (image, label) in enumerate(zip(images, labels)):
#         plt.subplot(1, num_images, i + 1)
#         plt.imshow(image.squeeze(0), cmap="gray")  # Remove channel dimension and show grayscale
#         plt.title(f"Label: {label}")
#         plt.axis("off")
#     plt.show()

def show_sample_images(dataset, num_images=10):
    if len(dataset) == 0:
        print("Dataset is empty.")
        return

    # Ensure num_images is not larger than the dataset size
    num_images = min(num_images, len(dataset))

    # Randomly sample indices from the dataset
    sample_indices = random.sample(range(len(dataset)), num_images)
    images, labels = zip(*[dataset[i] for i in sample_indices])

    # Plot the sampled images
    plt.figure(figsize=(15, 5))
    for i, (image, label) in enumerate(zip(images, labels)):
        plt.subplot(1, num_images, i + 1)

        # Make sure image is 2D by removing the channel dimension (if any)
        image = image.squeeze(0)  # Remove channel dimension (1, 28, 28) -> (28, 28)

        plt.imshow(image, cmap="gray")  # Display the image in grayscale
        plt.title(f"Label: {label}")
        plt.axis("off")

    plt.show()

def plot_loss_curve(loss_dict):
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


def prepare_data(train_images_path, train_labels_path, t10k_images_path, t10k_labels_path):
    """
    Prepare the training and testing datasets from binary files.
    """
    train_images = sc.binaryFiles(train_images_path)
    train_labels = sc.binaryFiles(train_labels_path)
    test_images = sc.binaryFiles(t10k_images_path)
    test_labels = sc.binaryFiles(t10k_labels_path)

    train_data_rdd = get_mnist_data(train_images, train_labels)
    test_data_rdd = get_mnist_data(test_images, test_labels)

    train_data = train_data_rdd.collect()
    test_data = test_data_rdd.collect()

    return train_data, test_data


def build_datasets(train_data, test_data):
    """
    Build PyTorch datasets from the processed MNIST data.
    """
    train_images, train_labels = zip(*train_data)
    test_images, test_labels = zip(*test_data)

    train_dataset = MNISTDataset(train_images, train_labels)
    test_dataset = MNISTDataset(test_images, test_labels)

    return train_dataset, test_dataset

def initialize_model(device):
    """
    Initialize the model and optimizer.
    """
    model = ImprovedCNN2().to(device)
    summary(model, input_size=(1, 28, 28))
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    return model, optimizer


def train_model(train_dataset, model, optimizer, world_size):
    """
    Train the model using distributed data parallel (DDP) with multiprocessing.
    """
    manager = Manager()
    batch_loss_dict = manager.dict()
    loss_dict = manager.dict()
    accuracy_dict = manager.dict()

    mp.spawn(
        init_process,
        args=(model, optimizer, train_dataset, world_size, loss_dict, accuracy_dict, batch_loss_dict),
        nprocs=world_size,
        join=True
    )

    return batch_loss_dict, loss_dict, accuracy_dict

def evaluate_model(model, test_dataset):
    """
    Evaluate the model and display the results.
    """
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    final_accuracy = calculate_accuracy(model, test_loader)
    print(f"Final Test Accuracy: {final_accuracy:.2f}%")
    plot_confusion_matrix(model, test_loader)
# ----------------- 6. 设置模型和优化器 -----------------
def main():
    device = torch.device("cpu")

    # Data preparation
    train_data, test_data = prepare_data(train_images_path, train_labels_path, t10k_images_path, t10k_labels_path)
    train_dataset, test_dataset = build_datasets(train_data, test_data)

    # 打印训练数据集的长度
    print(f"Number of training MNISTDataset: {len(train_dataset)}")
    # 打印测试数据集的长度
    print(f"Number of test MNISTDataset: {len(test_dataset)}")

    # Display some sample images from the training dataset
    show_sample_images(train_dataset, num_images=10)

    # Model initialization
    model, optimizer = initialize_model(device)

    # Distributed training
    batch_loss_dict, loss_dict, accuracy_dict = train_model(train_dataset, model, optimizer, WORLD_SIZE)

    #     # 读取原始数据
#     train_images = sc.binaryFiles(train_images_path)
#     train_labels = sc.binaryFiles(train_labels_path)
#     test_images = sc.binaryFiles(t10k_images_path)  # 读取测试集图像
#     test_labels = sc.binaryFiles(t10k_labels_path)  # 读取测试集标签
#
#
#     # # 获取训练数据
#     # train_images, train_labels = get_mnist_data(train_images, train_labels)
#     #
#     # # 获取测试数据
#     # test_images, test_labels = get_mnist_data(test_images, test_labels)
#     #
#     #
#     #
#     # # 创建 PyTorch 数据集
#     # train_dataset = MNISTDataset(train_images, train_labels)
#     #
#     # test_dataset = MNISTDataset(test_images, test_labels)
#
#     # 将数据收集到本地
#     train_mnist_data_rdd = get_mnist_data(train_images, train_labels)
#     test_mnist_data_rdd = get_mnist_data(test_images, test_labels)
#     train_data = train_mnist_data_rdd.collect()  # 触发计算并收集结果到本地
#     # Check the length of train_data
#     #print(f"Length of train_data: {len(train_data)}")  # Should be 60,000 (one tuple for each image)
#
#     test_data = test_mnist_data_rdd.collect()  # 触发计算并收集结果到本地
#
#     # Unzip the train data into individual images and labels
#     train_images, train_labels = zip(*train_data)
#
#     # Check the lengths of train_images and train_labels
#     #print(f"Number of train images: {len(train_images)}")  # Should be 60,000
#     #print(f"Number of train labels: {len(train_labels)}")  # Should be 60,000
#
#     # Check the shape of the first image and the first label
#     print(f"First image shape: {train_images[0].shape}")  # Should be (28, 28)
#     print(f"First label: {train_labels[0]}")  # The first label (e.g., 5)
#
#     # Create the MNISTDataset
#     train_dataset = MNISTDataset(train_images, train_labels)
#
#     # Verify the dataset length
#     #print(f"Number of training MNISTDataset: {len(train_dataset)}")  # Should be 60,000
#
#
# # 将结果转换为 PyTorch Dataset
#     test_images, test_labels = zip(*test_data)  # 将解析出的图像和标签分开
#     test_dataset = MNISTDataset(test_images, test_labels)
#
#     # 打印训练数据集的长度
#     print(f"Number of training MNISTDataset: {len(train_dataset)}")
#
#     # 打印测试数据集的长度
#     print(f"Number of test MNISTDataset: {len(test_dataset)}")
#
#     # Display some sample images from the training dataset
#     show_sample_images(train_dataset, num_images=10)
#
#     test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
#
#     #model = SimpleCNN().to(device)
#     #model = OptimizedCNN().to(device)
#     #model = ImprovedCNN().to(device)
#     model = ImprovedCNN2().to(device)
#
#     # 打印模型摘要，输入尺寸为 (1, 28, 28)，因为 MNIST 图像是 28x28 的单通道图像
#     summary(model, input_size=(1, 28, 28))
#
#     optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
#
#     world_size = WORLD_SIZE  # Number of processes (CPU cores)
#
#     # Using Manager to create a shared list for loss and accuracy
#     manager = Manager()
#     batch_loss_dict = manager.dict()
#     loss_dict = manager.dict()
#     accuracy_dict = manager.dict()
#
#     # Run DDP training using multiprocessing
#     mp.spawn(init_process, args=(model, optimizer, train_dataset, world_size, loss_dict, accuracy_dict,batch_loss_dict), nprocs=world_size, join=True)

    # Plot loss curve for each rank (only rank 0 should plot)
    if 0 == 0:
        plot_loss_curve(batch_loss_dict)

    # Plot accuracy curve for each rank (only rank 0 should plot)
    if 0 == 0:
        plot_accuracy_curve(accuracy_dict)


    # Model evaluation
    if 0 == 0:
        evaluate_model(model, test_dataset)

    # 在主函数最后停止 SparkContext
    sc.stop()

if __name__ == "__main__":
    main()
