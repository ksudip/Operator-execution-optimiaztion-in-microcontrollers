
import tensorflow as tf
from tensorflow.keras import layers
import psutil
import matplotlib.pyplot as plt

# Function to train a neural network without operator reordering
def train_without_reordering():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.reshape(-1, 784).astype("float32") / 255.0
    x_test = x_test.reshape(-1, 784).astype("float32") / 255.0
    model = tf.keras.Sequential([
        layers.Dense(512, activation="relu"),
        layers.Dense(10)
    ])
    model.compile(optimizer="adam", loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=["accuracy"])
    history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10, batch_size=128)
    return history

# Function to train a neural network with operator reordering
def train_with_reordering():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.reshape(-1, 784).astype("float32") / 255.0
    x_test = x_test.reshape(-1, 784).astype("float32") / 255.0
    model = tf.keras.Sequential([
        layers.Dense(512, activation="relu"),
        layers.Dense(10)
    ])
    model.compile(optimizer="adam", loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=["accuracy"])
    model = tf.function(model)
    history = model(x_train, training=True)
    return history

# Function to generate a memory usage graph for both models
def generate_memory_graph():
    plt.figure(figsize=(10, 6))
    plt.title("Memory Usage of Neural Network without Operator Reordering")
    plt.xlabel("Training Time (epochs)")
    plt.ylabel("Memory Usage (GB)")
    history_without_reordering = train_without_reordering()
    # history_with_reordering = train_with_reordering()
    memory_usage_without_reordering = []
    # memory_usage_with_reordering = []
    for i in range(50):
        memory_usage_without_reordering.append(psutil.Process().memory_info().rss / 1024 ** 3)
        # memory_usage_with_reordering.append(psutil.Process().memory_info().rss / 1024 ** 3)
        history_without_reordering = train_without_reordering()
        # history_with_reordering = train_with_reordering()
    plt.plot(memory_usage_without_reordering, label="Without Operator Reordering")
    # plt.plot(memory_usage_with_reordering, label="With Operator Reordering")
    plt.legend()
    plt.show()
    return memory_usage_without_reordering

# Call the function to generate the graph
mem_use_woR = generate_memory_graph()

def generate_memory_graph_for_with_operator_reordering():
    plt.figure(figsize=(10, 6))
    plt.title("Memory Usage of Neural Network with Operator Reordering")
    plt.xlabel("Training Time (epochs)")
    plt.ylabel("Memory Usage (GB)")
    # history_without_reordering = train_without_reordering()
    history_with_reordering = train_with_reordering()
    # memory_usage_without_reordering = []
    memory_usage_with_reordering = []
    for i in range(50):
        # memory_usage_without_reordering.append(psutil.Process().memory_info().rss / 1024 ** 3)
        memory_usage_with_reordering.append(psutil.Process().memory_info().rss / 1024 ** 3)
        # history_without_reordering = train_without_reordering()
        history_with_reordering = train_with_reordering()
    # plt.plot(memory_usage_without_reordering, label="Without Operator Reordering")
    plt.plot(memory_usage_with_reordering, label="With Operator Reordering" , color = 'green')
    plt.legend()
    plt.show()
    return memory_usage_with_reordering
mem_use_wR = generate_memory_graph_for_with_operator_reordering()

print(mem_use_woR)
print(mem_use_wR)

import numpy as np
lst = range(10)

plt.plot(lst,mem_use_wR , color = 'blue')
plt.plot( lst, mem_use_woR  , color = 'red')