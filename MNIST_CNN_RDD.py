from pyspark.sql import SparkSession
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from tensorflow.keras.utils import to_categorical
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# 创建 SparkSession
spark = SparkSession.builder \
    .appName("Read MNIST CSV from HDFS（localhost） using RDD") \
    .config("spark.default.parallelism", "24") \
    .getOrCreate()

# 从 HDFS 读取训练和测试数据为 RDD
mnist_train_rdd = spark.sparkContext.textFile("hdfs://localhost:9000/user/hadoop/MNISTinCSV/mnist_train.csv")
mnist_test_rdd = spark.sparkContext.textFile("hdfs://localhost:9000/user/hadoop/MNISTinCSV/mnist_test.csv")

# 跳过标题行
header_train = mnist_train_rdd.first()  # 获取训练数据的标题行
header_test = mnist_test_rdd.first()    # 获取测试数据的标题行
mnist_train_rdd = mnist_train_rdd.filter(lambda line: line != header_train)
mnist_test_rdd = mnist_test_rdd.filter(lambda line: line != header_test)

# 定义一个函数来处理 CSV 数据行，转换成 (label, features) 格式
def parse_line(line):
    values = line.split(",")
    label = int(values[0])  # 第一个值为标签
    features = np.array([float(x) for x in values[1:]], dtype='float32') / 255.0  # 其余值为特征，归一化
    return label, features

# 使用 map 转换数据
mnist_train_rdd = mnist_train_rdd.map(parse_line)
mnist_test_rdd = mnist_test_rdd.map(parse_line)

# 将标签和特征分开提取
train_labels = mnist_train_rdd.map(lambda x: x[0]).collect()  # 提取训练标签
test_labels = mnist_test_rdd.map(lambda x: x[0]).collect()    # 提取测试标签

train_images = np.array(mnist_train_rdd.map(lambda x: x[1]).collect())  # 提取并转换训练图像特征
test_images = np.array(mnist_test_rdd.map(lambda x: x[1]).collect())    # 提取并转换测试图像特征

# 检查标签和图像特征的形状
print(f"Train labels shape: {len(train_labels)}")
print(f"Test labels shape: {len(test_labels)}")
print(f"Train images shape: {train_images.shape}")
print(f"Test images shape: {test_images.shape}")



# 定义一个函数来解析 MNIST 数据并重塑为 28x28 图像
def parse_mnist_csv(line):
    label, image = line  # 直接从元组中获取标签和图像数据
    return label, image.reshape(28, 28)  # 将图像特征重塑为 28x28 图像

# 从 RDD 中取出一些样本数据进行可视化
sample_image_data = mnist_train_rdd.take(10)  # 取出10个样本（示例）

# 可视化前10个样本
plt.figure(figsize=(10, 5))
for i, line in enumerate(sample_image_data):
    label, image = parse_mnist_csv(line)
    plt.subplot(2, 5, i + 1)
    plt.imshow(image, cmap='gray')
    plt.title(f'Label: {label}')
    plt.axis('off')
plt.tight_layout()
plt.show()

# Convert the images to the shape (num_samples, 1, 28, 28)
train_images = train_images.reshape(-1, 1, 28, 28)
test_images = test_images.reshape(-1, 1, 28, 28)

# 标签进行 one-hot 编码
train_labels_onehot = to_categorical(train_labels, 10)
test_labels_onehot = to_categorical(test_labels, 10)

# 转换为 PyTorch 张量
train_images_tensor = torch.tensor(train_images).float()
test_images_tensor = torch.tensor(test_images).float()
train_labels_tensor = torch.tensor(train_labels_onehot).float()
test_labels_tensor = torch.tensor(test_labels_onehot).float()

# 将数据加载到 DataLoader 中
train_dataset = TensorDataset(train_images_tensor, train_labels_tensor)
test_dataset = TensorDataset(test_images_tensor, test_labels_tensor)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Define a simple CNN for MNIST
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)  # Input channels = 1, output channels = 32
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32 * 13 * 13, 128)  # Correct input size after convolution and pooling
        self.fc2 = nn.Linear(128, 10)  # Output layer for 10 classes

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # Apply conv1 and pooling
        #        print(x.shape)  # Check the tensor shape here
        x = x.view(-1, 32 * 13 * 13)  # Flatten the tensor to match the input size of fc1
        x = F.relu(self.fc1(x))  # Apply the first fully connected layer
        x = self.fc2(x)  # Apply the output layer
        return x


# Initialize the model
model = SimpleCNN()

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Print model architecture
print(model)

# 使用交叉熵作为损失函数
criterion = nn.CrossEntropyLoss()

# 使用 Adam 优化器
optimizer = optim.Adam(model.parameters(), lr=0.001)


#import matplotlib.pyplot as plt

# Lists to store loss and accuracy for plotting
losses = []
accuracies = []

epochs = 5
for epoch in range(epochs):
    model.train()  # Set the model to training mode
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in train_loader:
        # Prepare data for model input
        inputs = inputs.squeeze(3)  # Remove the single color channel
        labels = torch.argmax(labels, dim=1)  # Convert labels to index format

        optimizer.zero_grad()  # Clear gradients
        outputs = model(inputs)  # Model prediction
        loss = criterion(outputs, labels)  # Calculate loss
        loss.backward()  # Backpropagation
        optimizer.step()  # Update weights

        # Track loss and accuracy
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_loader)
    epoch_accuracy = correct / total
    losses.append(epoch_loss)  # Store loss for plotting
    accuracies.append(epoch_accuracy)  # Store accuracy for plotting

    print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")

# After training, plot the curves
plt.figure(figsize=(12, 5))

# Plot Loss Curve
plt.subplot(1, 2, 1)
plt.plot(range(1, epochs+1), losses, label='Loss', color='blue')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.grid(True)

# Plot Accuracy Curve
plt.subplot(1, 2, 2)
plt.plot(range(1, epochs+1), accuracies, label='Accuracy', color='green')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training Accuracy Curve')
plt.grid(True)

# Display the plots
plt.tight_layout()
plt.show()


# 评估模型
model.eval()  # 设置模型为评估模式
correct = 0
total = 0
with torch.no_grad():  # 关闭梯度计算
    for inputs, labels in test_loader:
        inputs = inputs.squeeze(3)
        labels = torch.argmax(labels, dim=1)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Test Accuracy: {correct/total:.4f}")