from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("Read MNIST CSV from WebHDFS") \
    .getOrCreate()

# # Read the train and test datasets from HDFS as RDDs。 Need change preprocessing if use RDDs.
# mnist_train_rdd = spark.sparkContext.textFile("hdfs://localhost:9000/user/MNISTinCSV/mnist_train.csv")
# mnist_test_rdd = spark.sparkContext.textFile("hdfs://localhost:9000/user/MNISTinCSV/mnist_test.csv")
#
#
# # Print the first few rows of the training and testing RDDs
# print("Training RDD sample data:")
# for row in mnist_train_rdd.take(5):  # Take the first 5 rows
#     print(row)
#
# print("\nTesting RDD sample data:")
# for row in mnist_test_rdd.take(5):  # Take the first 5 rows
#     print(row)


# read files from hdfs
mnist_train_df = spark.read.csv("hdfs://localhost:9000/user/MNISTinCSV/mnist_train.csv", header=True, inferSchema=True)
mnist_test_df = spark.read.csv("hdfs://localhost:9000/user/MNISTinCSV/mnist_test.csv", header=True, inferSchema=True)


# read files from localhost, for test purppose.
#mnist_train_df = spark.read.csv("/content/drive/MyDrive/MNISTinCSV/mnist_train.csv", header=True, inferSchema=True)
#mnist_test_df = spark.read.csv("/content/drive/MyDrive/MNISTinCSV/mnist_test.csv", header=True, inferSchema=True)


# 显示前 5 行
mnist_train_df.show(5)
mnist_test_df.show(5)


import pandas as pd
import numpy as np
import torch
from tensorflow.keras.utils import to_categorical
from torch.utils.data import DataLoader, TensorDataset


# Convert the 'label' column to a Pandas Series or collect it as a list
train_labels = mnist_train_df.select('label').rdd.flatMap(lambda x: x).collect()
test_labels = mnist_test_df.select('label').rdd.flatMap(lambda x: x).collect()

# Check the shape of the labels
print(f"Train labels shape: {len(train_labels)}")
print(f"Test labels shape: {len(test_labels)}")


train_images = mnist_train_df.drop('label').toPandas().values.astype('float32') / 255.0
test_images = mnist_test_df.drop('label').toPandas().values.astype('float32') / 255.0

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



import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

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


import matplotlib.pyplot as plt

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