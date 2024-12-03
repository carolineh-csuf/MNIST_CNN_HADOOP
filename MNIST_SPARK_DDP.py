import os
import time
import sys
import torch
import random
import logging
import pyspark
import subprocess
import numpy as np
import torch.nn as nn
import seaborn as sns
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.multiprocessing import Manager
from torchsummary import summary
from datetime import datetime
from pyspark import SparkContext
from sklearn.metrics import confusion_matrix
from pyspark.sql import SparkSession

# Training Configuration
EPOCH_COUNT = 1         # Number of training epochs
WORKER_COUNT = 6        # Number of processes (e.g., CPU cores) for training
BATCH_SIZE = 64         # Number of samples per batch

# Local file path for model checkpoint and HDFS directory for saving checkpoints
local_checkpoint_file_path = "model_checkpoint.pth"  # Path to save the model checkpoint locally
hdfs_checkpoint_dir = "/user/MNIST_Checkpoint"  # Directory on HDFS where model checkpoints will be stored

# Base HDFS path for the MNIST dataset
hdfs_base_path = "hdfs://localhost:9000/user/MNIST"  # The base HDFS path where the MNIST dataset is stored

# Paths to MNIST dataset files on HDFS
mnist_paths = {
    "train_images": f"{hdfs_base_path}/train-images-idx3-ubyte",  # Path to training image data
    "train_labels": f"{hdfs_base_path}/train-labels-idx1-ubyte",  # Path to training label data
    "test_images": f"{hdfs_base_path}/t10k-images-idx3-ubyte",    # Path to testing image data
    "test_labels": f"{hdfs_base_path}/t10k-labels-idx1-ubyte"     # Path to testing label data
}


def print_versions():
    """
    Print the versions of key libraries and tools (Python, PySpark, PyTorch, and Hadoop).
    """
    # Define a border for the box to format the output neatly
    border = "=" * 50

    # Print the bordered box with version information
    print(border)
    print(f"Python version: {sys.version}")
    print(f"PySpark version: {pyspark.__version__}")
    print(f"PyTorch version: {torch.__version__}")

    # Attempt to print Hadoop version, if available
    try:
        hadoop_version_output = subprocess.check_output(["hadoop", "version"]).decode("utf-8")
        # Extract the first line (which typically contains the version information)
        hadoop_version_line = hadoop_version_output.splitlines()[0]
        print(f"Hadoop version: {hadoop_version_line}")
    except FileNotFoundError:
        print("Hadoop is not installed or not in the PATH.")  # Print a message if Hadoop is not found

    # Print closing border
    print(border)

def generate_timestamp(format: str = "%Y%m%d_%H%M%S") -> str:
    """
    Generate the current timestamp in the specified format.
    """
    return datetime.now().strftime(format)  # Get the current timestamp formatted as a string

def generate_checkpoint_filename(prefix: str = "model_checkpoint", timestamp_format: str = "%Y%m%d_%H%M%S") -> str:
    """
    Generate a filename for saving a model checkpoint, including a timestamp.
    """
    timestamp = generate_timestamp(timestamp_format)  # Generate the timestamp
    return f"{prefix}_{timestamp}.pth"  # Return the formatted filename with timestamp

def initialize_spark(app_name="MyApp", master="local[*]"):
    """
    Initialize a SparkContext and SparkSession for Spark applications.
    """
    sc = SparkContext.getOrCreate()  # Create or retrieve an existing SparkContext
    spark = SparkSession.builder \
        .appName(app_name) \
        .master(master) \
        .getOrCreate()  # Create or retrieve an existing SparkSession

    # Reduce the verbosity of Spark logs to only show errors
    spark.sparkContext.setLogLevel("ERROR")

    return sc, spark  # Return the SparkContext and SparkSession objects


# Parse MNIST image data (28x28 pixels)
def parse_mnist_image(raw_bytes):
    """
    Parse the image data from a raw byte sequence in the IDX file format.

    This function skips the first 16 bytes (header) and reshapes the remaining data into
    a 3D array representing multiple images (each 28x28 pixels).

    Parameters:
    - raw_bytes (bytes): Raw byte data representing the image information.

    Returns:
    - numpy.ndarray: A 3D array of shape (num_images, 28, 28) representing the images.
    """
    # Parse the raw byte data into an array of uint8 values
    data = np.frombuffer(raw_bytes, dtype=np.uint8)
    # Skip the first 16 bytes (header) and reshape the remaining bytes into 28x28 images
    images = data[16:].reshape(-1, 28, 28)  # Skip the header and reshape
    return images

# Parse MNIST label data
def parse_mnist_labels(raw_bytes):
    """
    Parse the label data from a raw byte sequence in the IDX file format.

    This function skips the first 8 bytes (header) and returns the remaining data as labels.

    Parameters:
    - raw_bytes (bytes): Raw byte data representing the labels.

    Returns:
    - numpy.ndarray: A 1D array of labels corresponding to the images.
    """
    # Parse the raw byte data into an array of uint8 values
    data = np.frombuffer(raw_bytes, dtype=np.uint8)
    # Skip the first 8 bytes (header) and return the labels
    labels = data[8:]  # Skip the header and return the labels
    return labels

def print_image_by_column(image):
    """
    Print the image (28x28) by columns, ensuring each column is 3 characters wide.

    This function prints each pixel in a 28x28 image in a grid format where each pixel
    value is displayed with a width of 3 characters for readability.

    Parameters:
    - image (numpy.ndarray): A 28x28 array representing a single image.

    Output:
    - The image is printed in a readable format where each line corresponds to a row of pixels.
    """
    for row in range(image.shape[0]):  # For each row in the image
        for col in range(image.shape[1]):  # For each column in the image
            # Print each pixel value with a width of 3 characters for uniform formatting
            print(f"{image[row, col]:3d}", end=" ")  # Print pixel value with 3-character width
        print()  # Move to the next line after printing one row
    print("\n")  # Add an extra newline for separation between different images


def get_mnist_data(images_rdd, labels_rdd, is_train):
    """
    Load and preprocess MNIST data from RDDs (Resilient Distributed Datasets).

    This function performs the following:
    - Parses the image and label data from RDDs.
    - Optionally prints one image and label if `is_train` is True.
    - Pairs each image with its corresponding label.
    - Calculates the frequency of each label in the dataset.

    Parameters:
    - images_rdd (RDD): RDD containing image data in byte format.
    - labels_rdd (RDD): RDD containing label data in byte format.
    - is_train (bool): Flag to indicate whether the data is for training; used for debugging outputs.

    Returns:
    - labeled_image_rdd (RDD): An RDD of (image, label) pairs.
    """
    # Parse image data by using flatMap to expand each image entry
    images_rdd = images_rdd.flatMap(lambda x: parse_mnist_image(x[1]))  # Parse image data into usable format
    labels_rdd = labels_rdd.flatMap(lambda x: parse_mnist_labels(x[1]))  # Parse label data

    if is_train:
        # Optionally print out the first image and label for debugging
        # print(images_rdd.take(1)[0])  # Uncomment to print raw image data
        print_image_by_column(images_rdd.take(1)[0])  # Print the first image in a column format for better readability
        print(labels_rdd.take(1)[0])  # Print the first label for inspection

    # Pair each image with its corresponding label
    labeled_image_rdd = images_rdd.zip(labels_rdd)

    # Calculate the frequency of each label in the dataset using reduceByKey
    label_counts = labeled_image_rdd.map(lambda x: (x[1], 1))  # Create pairs (label, 1) to count occurrences
    label_count_rdd = label_counts.reduceByKey(lambda a, b: a + b)  # Sum the occurrences for each label

    if is_train:
        # Collect the label counts and print them for debugging
        label_count_result = label_count_rdd.collect()  # Collect the result to print it
        for label, count in label_count_result:
            print(f"Label {label}: {count} times")  # Print the count of each label

    return labeled_image_rdd  # Return the RDD of (image, label) pairs


class MNISTDataset(Dataset):
    """
    PyTorch Dataset class for the MNIST dataset.

    This class takes the images and labels as input and provides methods for:
    - Retrieving the total number of samples in the dataset.
    - Accessing each image-label pair at a specific index.

    The images are normalized to the range [0, 1] and reshaped to include a channel dimension.

    Parameters:
    - images (list or array): List or array of image data (each image is 28x28).
    - labels (list or array): List or array of labels corresponding to the images.

    Attributes:
    - images (list or array): The image data.
    - labels (list or array): The label data.
    """
    def __init__(self, images, labels):
        """
        Initialize the MNIST dataset.
        """
        self.images = images
        self.labels = labels

    def __len__(self):
        """
        Return the total number of samples in the dataset.
        """
        return len(self.images)

    def __getitem__(self, idx):
        """
        Retrieve an image-label pair at a specific index.

        The image is converted to a tensor and normalized to the range [0, 1], with an added channel dimension (1, 28, 28).

        Parameters:
        - idx (int): The index of the sample to retrieve.

        Returns:
        - tuple: (image, label) where `image` is a tensor of shape (1, 28, 28) and `label` is a tensor of shape (long).
        """
        image = self.images[idx]  # Get the image at the specified index
        label = self.labels[idx]  # Get the corresponding label

        # Convert the image to a tensor, normalize, and add a channel dimension
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)  # Add channel dimension (1, 28, 28)
        image = image / 255.0  # Normalize image pixel values to the range [0, 1]

        # Convert the label to a tensor of type long
        label = torch.tensor(label, dtype=torch.long)

        return image, label  # Return the image and label as a tuple



class ImprovedCNN(nn.Module):
    def __init__(self):
        """
        Initialize the improved CNN model with additional convolutional layers, batch normalization,
        and dropout layers to improve performance and prevent overfitting.

        The network architecture includes:
        - 3 convolutional layers with increasing output channels (32, 64, 128).
        - Batch normalization after each convolutional layer.
        - Dropout to reduce overfitting.
        - Fully connected layers (fc1, fc2, fc3) for classification.
        - Swish activation function.
        """
        super(ImprovedCNN, self).__init__()

        # First convolutional layer with batch normalization
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        # Second convolutional layer with batch normalization
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        # Third convolutional layer with batch normalization (new layer)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # Fully connected layers (fc1, fc2, fc3)
        # The input size to fc1 is based on the output size of the last convolutional layer
        self.fc1 = nn.Linear(128 * 3 * 3, 256)  # Adjusted input size
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

        # Dropout layer to prevent overfitting
        self.dropout = nn.Dropout(0.5)

        # Swish activation function
        self.swish = nn.SiLU()

    def forward(self, x):
        """
        Define the forward pass of the model. The input `x` passes through the following sequence:
        1. Convolutional layers with Batch Normalization and Swish activation.
        2. Max pooling layers to reduce spatial dimensions.
        3. Fully connected layers with Swish activation and dropout.
        4. Final output layer for classification.

        Parameters:
        - x (Tensor): Input tensor with shape (batch_size, channels, height, width).

        Returns:
        - Tensor: The output tensor representing class scores.
        """
        # Pass through the first convolutional layer, batch normalization, and Swish activation
        x = self.swish(self.bn1(self.conv1(x)))
        x = torch.max_pool2d(x, 2)  # Max pooling

        # Pass through the second convolutional layer, batch normalization, and Swish activation
        x = self.swish(self.bn2(self.conv2(x)))
        x = torch.max_pool2d(x, 2)  # Max pooling

        # Pass through the third convolutional layer, batch normalization, and Swish activation
        x = self.swish(self.bn3(self.conv3(x)))  # New conv layer
        x = torch.max_pool2d(x, 2)  # Max pooling

        # Debugging: Uncomment to print the shape of `x` before flattening
        # print(x.shape)  # For debugging: Check the shape of the tensor before flattening

        # Flatten the tensor before passing it to the fully connected layers
        x = x.view(x.size(0), -1)  # Flatten the output of conv layers

        # Pass through the fully connected layers with Swish activation and dropout
        x = self.swish(self.fc1(x))  # First fully connected layer with Swish activation
        x = self.dropout(x)  # Apply dropout
        x = self.swish(self.fc2(x))  # Second fully connected layer with Swish activation
        x = self.fc3(x)  # Output layer (class scores)

        return x


def print_and_check_model_params(model, rank, world_size):
    """
    Print and verify that model parameters are consistent across all ranks during training.
    This function ensures the model parameters are synchronized across different ranks in a
    distributed training setup, especially when using multiple GPUs or nodes.
    """

    # Get the model's state_dict (which contains the model's parameters)
    model_params = model.state_dict()

    # Collect the parameters from all ranks (this part is not yet fully implemented)
    all_params = []

    # The function checks the model's parameters to ensure they are identical across all ranks.
    # Rank 0 performs the consistency check and prints any mismatches found.
    # Debugging: Uncomment to print each rank's parameters and their shapes
    # for name, param in model.named_parameters():
    #     print(f"Rank {rank}: {name} - Shape: {param.data.shape}")

    # Debugging: Uncomment to print current parameter values across all ranks for comparison
    # # Rank 0 prints parameter values from all ranks
    # if rank == 0:
    #     print(f"Rank {rank} - Parameter {name}:")
    #     for i in range(world_size):
    #         print(f"    Rank {i}: {gathered_params[i]}")

    # Rank 0 compares parameters to check consistency across ranks
    if rank == 0:
        for name, gathered_params in zip(model_params.keys(), all_params):
            # Check that parameters are the same across all ranks
            for i in range(1, world_size):
                assert torch.allclose(gathered_params[0], gathered_params[i], atol=1e-4), f"Mismatch in parameter {name} between rank 0 and rank {i}"
                print(f"Parameter {name} is consistent across all ranks.")

def load_checkpoint(model, optimizer, seed, filename="model_checkpoint.pth"):
    """
    Load the model checkpoint if it exists. If the checkpoint file does not exist,
    return the default starting epoch (0) and loss (0.0). Only rank 0 loads the checkpoint,
    and then broadcasts the epoch and loss to all other ranks.

    Parameters:
    - model (nn.Module): The model to load the checkpoint into.
    - optimizer (Optimizer): The optimizer to load the checkpoint into.
    - filename (str): The name of the checkpoint file to load from.

    Returns:
    - epoch (int): The epoch to resume training from.
    - loss (float): The loss value at the time of the checkpoint.
    """
    epoch = 0
    loss = 0.0

    # Only rank 0 should load the checkpoint
    if torch.distributed.get_rank() == 0 and os.path.exists(filename):
        checkpoint = torch.load(filename, weights_only=True)
        print(f"Loaded checkpoint from {filename}")

        epoch = checkpoint['epoch'] + 1  # Start training from the next epoch after the saved one
        loss = checkpoint.get('loss', 0.0)
        seed = checkpoint['seed']
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Checkpoint loaded from epoch {epoch - 1}, loss {loss}")

        epoch_tensor = torch.tensor(epoch).to(torch.device("cpu"))
        loss_tensor = torch.tensor(loss).to(torch.device("cpu"))

        # Broadcast the epoch and loss to all ranks
        if torch.distributed.get_rank() == 0:
            torch.distributed.broadcast(epoch_tensor, 0)
            torch.distributed.broadcast(loss_tensor, 0)
        print(f"Rank {torch.distributed.get_rank()} received epoch {epoch_tensor.item()} and loss {loss_tensor.item()}")
    else:
        print("Checkpoint not found or rank != 0, starting from scratch.")

        # Skip broadcasting and return default values when no checkpoint is found
        epoch_tensor = torch.tensor(epoch).to(torch.device("cpu"))
        loss_tensor = torch.tensor(loss).to(torch.device("cpu"))

        # No broadcast needed, as all ranks will start from scratch
        return epoch_tensor.item(), loss_tensor.item()

    return epoch_tensor.item(), loss_tensor.item()


def save_checkpoint(model, optimizer, epoch, loss, seed, filename="model_checkpoint.pth"):
    """
    Save the model and optimizer state to a checkpoint file.
    This function saves the model's parameters, optimizer state,
    current epoch, and loss. Only rank 0 performs the saving operation.

    Parameters:
    - model (nn.Module): The model to save.
    - optimizer (Optimizer): The optimizer to save.
    - epoch (int): The current epoch number.
    - loss (float): The current loss value.
    - filename (str): The name of the file to save the checkpoint.
    """
    if torch.distributed.get_rank() == 0:
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'seed': seed
        }
        torch.save(checkpoint, filename)
        print(f"Checkpoint saved to {filename} at epoch {epoch}")


def upload_checkpoint_to_hdfs(local_checkpoint_file, hdfs_checkpoint_dir):
    """
    Upload a checkpoint file from the local system to HDFS. This function checks if
    the local checkpoint file exists, creates the HDFS directory if needed, uploads
    the file with a timestamped name, and deletes the local file after a successful
    upload. It handles errors during the process.
    """
    # Check if the local checkpoint file exists before proceeding with the upload
    if not os.path.exists(local_checkpoint_file):
        print(f"Local file {local_checkpoint_file} does not exist, unable to upload.")
        return

    try:
        # Ensure the HDFS directory exists, creating it if necessary
        subprocess.run(["hdfs", "dfs", "-mkdir", "-p", hdfs_checkpoint_dir], check=True)

        # Generate the full HDFS file path, appending a timestamp for version control
        hdfs_file_path = os.path.join(hdfs_checkpoint_dir, generate_checkpoint_filename())

        # Upload the local checkpoint file to HDFS, overwriting if necessary
        subprocess.run(["hdfs", "dfs", "-put", "-f", local_checkpoint_file, hdfs_file_path], check=True)
        print(f"File successfully uploaded to HDFS: {hdfs_file_path}")

        # Remove the local checkpoint file after a successful upload to free up disk space
        os.remove(local_checkpoint_file)
        print(f"Local file {local_checkpoint_file} has been deleted.")

    except subprocess.CalledProcessError as e:
        # Handle errors related to HDFS operations
        print(f"HDFS operation failed: {e}")
    except Exception as e:
        # Handle any other exceptions that may occur
        print(f"An error occurred: {e}")


def init_process(rank, model, optimizer, train_dataset, world_size, loss_dict, accuracy_dict, batch_loss_dict,seed):
    """
    Initialize the distributed training process for multi-GPU or multi-node setup.
    This function initializes the distributed process group, sets up logging,
    prepares the model and data loader for each rank, and handles checkpointing.
    """

    # Set the environment variables for distributed training
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # Configure logging for each rank
    log_filename = f"rank_{rank}_training_log.txt"
    #logging.basicConfig(filename=log_filename, level=logging.INFO)
    logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


    # Initialize the distributed process group using Gloo backend
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    # Wrap the model in DistributedDataParallel for multi-GPU training
    model = nn.parallel.DistributedDataParallel(model)

    # Only rank 0 loads the checkpoint to avoid deadlock
    if rank == 0:
        start_epoch, loss = load_checkpoint(model, optimizer, seed, filename="model_checkpoint.pth")
    else:
        start_epoch, loss = 0, 0.0  # Other ranks start from scratch

    # Using DistributedSampler for data parallelism across ranks
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)

    # Create the data loader with the sampler to ensure correct batching across ranks
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler, num_workers=1)

    dist.barrier()  # Synchronize all ranks before training

    # Set the correct epoch for each rank to ensure proper data shuffling
    train_sampler.set_epoch(start_epoch)

    # Record the start time for training
    start_time = time.time()

    # Initialize variables to track total loss and accuracy
    total_loss = 0.0
    total_batches = 0

    # Track the loss and accuracy for each rank
    rank_loss = []  # Stores the loss for each batch in this rank
    rank_accuracy = []  # Stores the accuracy for each batch in this rank

    # Start the training loop
    for epoch in range(start_epoch, EPOCH_COUNT):
        print(f"Rank {rank} - Epoch {epoch} starts.")

        # Ensure the sampler is correctly shuffled for each epoch
        train_sampler.set_epoch(epoch)

        epoch_loss = 0
        correct = 0
        total = 0

        # Loop through the batches in the train_loader
        for batch_idx, (data, target) in enumerate(train_loader):
            # Move data and target to the CPU
            data, target = data.to("cpu"), target.to("cpu")

            # Zero out the gradients
            optimizer.zero_grad()

            # Forward pass: Get model output
            output = model(data)

            # Calculate the loss
            loss = nn.CrossEntropyLoss()(output, target)

            # Backpropagate the loss and update the model parameters
            loss.backward()
            optimizer.step()

            # Track the loss for each batch
            epoch_loss += loss.item()

            total_loss += loss.item()  # Accumulate total loss
            total_batches += 1  # Count the total number of batches

            # Calculate accuracy for this batch
            _, predicted = torch.max(output, 1)
            correct += (predicted == target).sum().item()
            total += target.size(0)

            # Store the batch loss and accuracy for the current rank
            rank_loss.append(loss.item())
            accuracy = 100 * correct / total
            rank_accuracy.append(accuracy)

            if batch_idx % 50 == 0:
                print(f"Rank {rank}, PID {os.getpid()} with {len(data)} samples, Epoch {epoch}, Batch {batch_idx}, Loss {loss.item()} ")

        # Calculate epoch accuracy
        epoch_accuracy = 100 * correct / total
        rank_accuracy.append(epoch_accuracy)  # Store the accuracy for this epoch

        # Synchronize all ranks after each epoch
        dist.barrier()

        # Store the loss and accuracy for the rank
        batch_loss_dict[rank] = rank_loss  # Store batch-wise losses for this rank
        loss_dict[rank] = epoch_loss / len(train_loader)  # Store average loss for the rank
        accuracy_dict[rank] = rank_accuracy  # Store accuracy for each rank

        # Save checkpoint only for rank 0
        if rank == 0:
            save_checkpoint(model, optimizer, epoch, loss.item(), seed, filename="model_checkpoint.pth")

        # Log epoch-wise loss and accuracy for each rank
        logging.info(f"Rank {rank} - Epoch {epoch}, Loss {epoch_loss / len(train_loader):.4f}, Accuracy {epoch_accuracy:.2f}%")

    # Record the end time for training
    end_time = time.time()
    training_time = end_time - start_time  # Total training time

    dist.barrier()  # Synchronize all ranks before printing the results

    # Aggregate total loss across all ranks
    total_loss = torch.tensor(total_loss).to("cpu")
    dist.reduce(total_loss, op=dist.ReduceOp.SUM, dst=0)

    # Aggregate accuracy across all ranks
    total_accuracy = torch.tensor(epoch_accuracy).to("cpu")
    dist.reduce(total_accuracy, op=dist.ReduceOp.SUM, dst=0)

    # Only rank 0 prints the final results and logs them
    if rank == 0:
        total_accuracy /= world_size  # Calculate average accuracy across all ranks
        avg_loss = total_loss.item() / total_batches  # Calculate average loss across all batches

        # Format and print the final training results
        print(f"\n{'='*53}")
        if world_size > 1:
            print(f"Training Results: Parallel Training - World Size: {world_size}")
        else:
            print("Training Results: Single Node Training")
        print(f"{'='*53}")
        print(f"Total Epochs: {EPOCH_COUNT}")
        print(f"Batch Size: {BATCH_SIZE}")
        print(f"Final Training Loss:       {avg_loss:.4f}")
        print(f"Final Training Accuracy:   {total_accuracy.item():.2f}%")
        print(f"Total Training Time:       {training_time:.2f} seconds")
        print(f"{'='*53}")

        # Log the results
        # logging.info(f"\n{'='*53}")
        # if world_size > 1:
        #     logging.info(f"Training Results: Parallel Training - World Size: {world_size}")
        # else:
        #     logging.info("Training Results: Single Node Training")
        # logging.info(f"Total Epochs: {EPOCH_COUNT}")
        # logging.info(f"Batch Size: {BATCH_SIZE}")
        # logging.info(f"Final Training Loss:       {avg_loss:.4f}")
        # logging.info(f"Final Training Accuracy:   {total_accuracy.item():.2f}%")
        # logging.info(f"Total Training Time:       {training_time:.2f} seconds")
        # logging.info(f"{'='*53}")

    # Save the checkpoint for rank 0 to HDFS
    if rank == 0:
        upload_checkpoint_to_hdfs(local_checkpoint_file_path, hdfs_checkpoint_dir)

    dist.barrier()  # Ensure all ranks finish before the process ends


def calculate_accuracy(model, data_loader):
    """
    Calculate the accuracy of the model on the given dataset.
    The function evaluates the model's performance by comparing the predicted labels
    with the true labels for all batches in the data loader.
    """
    # Set the model to evaluation mode
    model.eval()

    # Initialize counters for correct predictions and total samples
    correct = 0
    total = 0

    # Disable gradient calculation for evaluation
    with torch.no_grad():
        # Iterate through the dataset in the data loader
        for data, target in data_loader:
            # Get model predictions
            output = model(data)

            # Get the predicted class with the highest probability
            _, predicted = torch.max(output, 1)

            # Count the number of correct predictions
            correct += (predicted == target).sum().item()
            total += target.size(0)

    # Calculate the accuracy as a percentage
    accuracy = 100 * correct / total

    return accuracy


def show_sample_images(dataset, num_images=10):
    """
    Display a specified number of sample images from the dataset.
    The function randomly selects images and labels from the dataset and plots them.
    """
    # Check if the dataset is empty
    if len(dataset) == 0:
        print("Dataset is empty.")
        return

    # Ensure that the number of images to display does not exceed the dataset size
    num_images = min(num_images, len(dataset))

    # Randomly sample indices from the dataset
    sample_indices = random.sample(range(len(dataset)), num_images)
    images, labels = zip(*[dataset[i] for i in sample_indices])

    # Plot the sampled images
    plt.figure(figsize=(15, 5))
    for i, (image, label) in enumerate(zip(images, labels)):
        plt.subplot(1, num_images, i + 1)

        # Ensure the image is 2D by removing the channel dimension (if any)
        image = image.squeeze(0)  # Remove channel dimension (1, 28, 28) -> (28, 28)

        # Display the image in grayscale
        plt.imshow(image, cmap="gray")
        plt.title(f"Label: {label}")
        plt.axis("off")

    # Show the plot
    plt.show()


def plot_loss_curve(loss_dict):
    """
    Plot the loss curve for each rank during training.
    The function visualizes the loss over batches for each rank in a distributed training setting.
    """
    plt.figure(figsize=(12, 6))

    # Loop through each rank and plot its loss curve
    for rank, rank_loss in loss_dict.items():
        plt.plot(rank_loss, label=f"Rank {rank}")

    # Add labels and title to the plot
    plt.xlabel("Batch")
    plt.ylabel("Loss")
    plt.title("Loss Curve for Each Rank")
    plt.legend()

    # Show the plot
    plt.show()


def plot_accuracy_curve(accuracy_dict):
    """
    Plot the training accuracy curve for each rank.
    The function visualizes the accuracy over batches for each rank in a distributed training setting.
    """
    plt.figure(figsize=(10, 6))

    # Loop through each rank and plot its accuracy curve
    for rank, accuracy in accuracy_dict.items():
        plt.plot(accuracy, label=f'Rank {rank}')

    # Add labels, title, and grid to the plot
    plt.title("Training Accuracy for Each Rank")
    plt.xlabel("Batch Index")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True)

    # Show the plot
    plt.show()


def plot_confusion_matrix(model, data_loader, num_classes=10):
    """
    Plot the confusion matrix for the model's predictions.
    The function evaluates the model on the data_loader and visualizes the confusion matrix using seaborn.
    """
    # Set the model to evaluation mode
    model.eval()

    # Initialize lists to store predictions and true labels
    all_preds = []
    all_labels = []

    # Disable gradient calculation to speed up inference
    with torch.no_grad():
        # Iterate through the data_loader
        for data, target in data_loader:
            # Get the model's predictions
            output = model(data)
            _, predicted = torch.max(output, 1)

            # Store the predictions and true labels
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(target.cpu().numpy())

    # Compute the confusion matrix
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))

    # Plot the confusion matrix using seaborn
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=list(range(num_classes)), yticklabels=list(range(num_classes)))
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()


def prepare_data(train_images_path, train_labels_path, t10k_images_path, t10k_labels_path, sparkSession):
    """
    Prepare the training and testing datasets from binary files.
    The function loads MNIST data from binary files and returns the processed training and testing datasets.
    """
    # Load the binary image and label files for training and testing data
    train_images = sparkSession.binaryFiles(train_images_path)
    train_labels = sparkSession.binaryFiles(train_labels_path)
    test_images = sparkSession.binaryFiles(t10k_images_path)
    test_labels = sparkSession.binaryFiles(t10k_labels_path)

    # Process the training and testing data into RDDs
    train_data_rdd = get_mnist_data(train_images, train_labels, is_train=True)
    test_data_rdd = get_mnist_data(test_images, test_labels, is_train=False)

    # Collect the processed data into lists
    train_data = train_data_rdd.collect()
    test_data = test_data_rdd.collect()

    # Return the prepared training and testing data
    return train_data, test_data


def build_datasets(train_data, test_data):
    """
    Build PyTorch datasets from the processed MNIST data.
    The function prepares training and testing datasets from the provided data.
    """
    # Unzip the training and testing data into images and labels
    train_images, train_labels = zip(*train_data)
    test_images, test_labels = zip(*test_data)

    # Create MNISTDataset instances for training and testing datasets
    train_dataset = MNISTDataset(train_images, train_labels)
    test_dataset = MNISTDataset(test_images, test_labels)

    # Return the created datasets
    return train_dataset, test_dataset


def initialize_model(device):
    """
    Initialize the model and optimizer.
    The function sets up the model and optimizer, then returns both.
    """
    # Instantiate the ImprovedCNN2 model and move it to the specified device (CPU or GPU)
    model = ImprovedCNN().to(device)

    # Print the summary of the model with a given input size
    summary(model, input_size=(1, 28, 28))

    # Initialize the optimizer (SGD with learning rate and momentum)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    # Return the model and optimizer
    return model, optimizer


def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def train_model(train_dataset, model, optimizer, world_size):
    """
    Train the model using Distributed Data Parallel (DDP) with multiprocessing.
    """
    manager = Manager()
    seed = 42
    set_seed(seed)

    # Initialize dictionaries for batch loss, total loss, and accuracy tracking
    batch_loss_dict = manager.dict()
    loss_dict = manager.dict()
    accuracy_dict = manager.dict()

    # Spawn processes to train the model in parallel across multiple workers
    mp.spawn(
        init_process,
        args=(model, optimizer, train_dataset, world_size, loss_dict, accuracy_dict, batch_loss_dict,seed),
        nprocs=world_size,  # Set the number of processes to match the world size
        join=True  # Wait for all processes to finish
    )

    # Return dictionaries containing the results for loss and accuracy tracking
    return batch_loss_dict, loss_dict, accuracy_dict


def evaluate_model(model, test_dataset):
    """
    Evaluate the model on the test dataset and display the results.
    """
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Calculate test accuracy
    final_accuracy = calculate_accuracy(model, test_loader)

    # Create a visually framed output for the test results
    print("=" * 53)
    print(f"|{'Final Test Results':^51}|")
    print("=" * 53)
    print(f"|{'Test Accuracy:':<30} {final_accuracy:.2f}%{' ':<14}|")
    print("=" * 53)

    # Plot and display the confusion matrix
    plot_confusion_matrix(model, test_loader)

def main():
    """
    Main function to initialize, train, and evaluate the model.
    """
    # Initialize Spark
    sc, spark = initialize_spark(app_name="MNIST_Processing")

    # Execute the task
    print("SparkContext initialized.")
    print(f"Application ID: {sc.applicationId}")

    # Print version information
    print_versions()

    # Set device to CPU (can be changed to GPU if available)
    device = torch.device("cpu")

    # Data preparation
    # Prepare data using MNIST file paths and Spark context
    train_data, test_data = prepare_data(
        mnist_paths["train_images"],
        mnist_paths["train_labels"],
        mnist_paths["test_images"],
        mnist_paths["test_labels"],
        sc
    )

    # Build datasets for training and testing
    train_dataset, test_dataset = build_datasets(train_data, test_data)

    # Print the length of the training dataset
    print(f"Number of training MNISTDataset: {len(train_dataset)}")

    # Print the length of the test dataset
    print(f"Number of test MNISTDataset: {len(test_dataset)}")

    # Display some sample images from the training dataset
    show_sample_images(train_dataset, num_images=10)

    # Model initialization
    model, optimizer = initialize_model(device)

    # Distributed training (This assumes the training function handles multiple workers)
    batch_loss_dict, loss_dict, accuracy_dict = train_model(train_dataset, model, optimizer, WORKER_COUNT)

    # Only rank 0 should plot loss, accuracy curves, and evaluate the model
    if 0 == 0:
        plot_loss_curve(batch_loss_dict)  # Plot loss curve
        plot_accuracy_curve(accuracy_dict)  # Plot accuracy curve
        evaluate_model(model, test_dataset)  # Model evaluation

    # Stop SparkContext at the end of the main function
    sc.stop()

if __name__ == "__main__":
    main()
