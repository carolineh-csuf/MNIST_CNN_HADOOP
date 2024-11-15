import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.distributed as dist
from pyspark import SparkContext

# ----------------- 1. 配置并初始化Spark -----------------
# 创建Spark上下文
sc = SparkContext("local", "MNIST_DDP")

# ----------------- 2. 定义数据预处理函数 -----------------
def preprocess_data(record):
    # 如果是标题行，跳过
    if "label" in record:  # 检查记录是否包含 'label' 字段
        return None  # 跳过标题行

    # 假设record是一个CSV格式的字符串
    values = record.split(",")  # 根据逗号分割

    # 提取标签（第一列）
    label = int(values[0])

    # 提取像素值并转换为Tensor，重塑为1x28x28的图像
    pixels = torch.tensor([int(pixel) for pixel in values[1:]], dtype=torch.float32).view(1, 28, 28)

    # 正常化像素值为[0, 1]之间
    pixels = pixels / 255.0

    return label, pixels

# ----------------- 3. 加载MNIST数据 -----------------
def load_data_from_hdfs(file_path):
    # 读取HDFS上的CSV文件
    rdd = sc.textFile(file_path)

    # 使用map操作对每条记录进行预处理
    processed_rdd = rdd.map(preprocess_data)

    return processed_rdd

# ----------------- 4. 定义自定义数据集 -----------------
class MNISTDataset(Dataset):
    def __init__(self, rdd):
        #self.data = rdd.collect()
        # Filter out None values from the RDD before collecting into the dataset
        self.data = [item for item in rdd.collect() if item is not None]
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        label, image = self.data[idx]
        return image, label

# ----------------- 5. 定义CNN模型 -----------------
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        #self.fc1 = nn.Linear(64 * 28 * 28, 128)
        # Update the input size for the first fully connected layer
        self.fc1 = nn.Linear(64 * 7 * 7, 128)  # Flattened 7x7x64
        self.fc2 = nn.Linear(128, 10)

    # def forward(self, x):
    #     x = torch.relu(self.conv1(x))
    #     x = torch.max_pool2d(x, 2)
    #     x = torch.relu(self.conv2(x))
    #     x = torch.max_pool2d(x, 2)
    #     x = x.view(x.size(0), -1)  # Flatten
    #     x = torch.relu(self.fc1(x))
    #     x = self.fc2(x)
    #     return x
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)  # 28x28 -> 14x14
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)  # 14x14 -> 7x7
        x = x.view(x.size(0), -1)  # Flatten to (batch_size, 3136)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# ----------------- 6. DDP初始化 -----------------
def init_process(rank, size, model, optimizer, data_loader, world_size):
    # Set environment variables for local distributed training
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'  # You can choose any free port

    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    model.train()

    # 设置DistributedDataParallel
    #model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])
    # 使用DistributedDataParallel时，不需要指定device_ids和output_device对于CPU
    model = nn.parallel.DistributedDataParallel(model)



    # 训练循环
    for epoch in range(1):  # 这里可以设置训练epoch
        for batch_idx, (data, target) in enumerate(data_loader):
            #data, target = data.to(device), target.to(device)  # Send to CPU (no .cuda() here)
            data, target = data.to("cpu"), target.to("cpu")  # 发送到CPU
            optimizer.zero_grad()
            output = model(data)
            loss = nn.CrossEntropyLoss()(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % 10 == 0:
                print(f"Rank {rank}, Epoch {epoch}, Batch {batch_idx}, Loss {loss.item()}")
    dist.barrier()

# ----------------- 7. 设置模型和优化器 -----------------
def main():
    # 设置设备为 CPU
    device = torch.device("cpu")

    # 加载训练和测试数据
    mnist_train_rdd = load_data_from_hdfs("hdfs://localhost:9000/user/MNISTinCSV/mnist_train.csv")  # 替换为实际路径
    mnist_test_rdd = load_data_from_hdfs("hdfs://localhost:9000/user/MNISTinCSV/mnist_test.csv")    # 替换为实际路径

    # 创建训练集和测试集Dataset
    train_dataset = MNISTDataset(mnist_train_rdd)
    test_dataset = MNISTDataset(mnist_test_rdd)

    # 创建DataLoader
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # 创建模型和优化器
    model = SimpleCNN().to(device)  # Ensure the model is on CPU
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    # 使用PyTorch DDP
    world_size = 1  # 单机
    init_process(0, world_size, model, optimizer, train_loader, world_size)

if __name__ == "__main__":
    main()
