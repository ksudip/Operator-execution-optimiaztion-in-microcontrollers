import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import onnx
import onnx_tensorrt.backend as backend
import tensorrt as trt

# Load MNIST dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Preprocess input data
x_train = x_train.reshape((60000, 28, 28, 1)).astype(np.float32) / 255.0
x_test = x_test.reshape((10000, 28, 28, 1)).astype(np.float32) / 255.0

# Define the neural network architecture
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=5)

# Evaluate the model on test data
test_loss, test_acc = model.evaluate(x_test, y_test)
print('Test accuracy:', test_acc)

# Save the trained model
tf.saved_model.save(model, "saved_model")

# Load the saved model
loaded_model = tf.saved_model.load("saved_model")

# Convert the model to ONNX format
onnx_model = onnx.convert_keras(loaded_model)

# Parse the ONNX model with TensorRT parser
parser = trt.OnnxParser(trt.Logger(trt.Logger.WARNING))
parser.parse(onnx_model.SerializeToString())

# Create a TensorRT engine
builder = trt.Builder(trt.Logger(trt.Logger.WARNING))
builder.max_workspace_size = 1 << 20  # 1 MB
network = builder.create_network()
engine = builder.build_cuda_engine(network)

# Optimize memory with operator reordering
config = trt.Config()
config.flags |= trt.BuilderFlag.FORCE_OP_REORDER
config.max_workspace_size = 1 << 20  # 1 MB
builder = trt.Builder(trt.Logger(trt.Logger.WARNING))
builder.max_batch_size = 1
builder.max_workspace_size = config.max_workspace_size
engine_with_reordering = builder.build_engine(network, config)

# Get memory usage for both engines
context = engine.create_execution_context()
bindings = [np.empty_like(x_test)]
context.execute_async_v2(bindings=bindings, stream_handle=0)
memory_with_reordering = context.get_binding_shape(0)[0] * np.dtype(np.float32).itemsize
print("Memory usage with operator reordering:", memory_with_reordering)

context = engine_with_reordering.create_execution_context()
bindings = [np.empty_like(x_test)]
context.execute_async_v2(bindings=bindings, stream_handle=0)
memory_without_reordering = context.get_binding_shape(0)[0] * np.dtype(np.float32).itemsize
print("Memory usage without operator reordering:", memory_without_reordering)

# Plot memory difference
labels = ['With Operator Reordering', 'Without Operator Reordering']
memory_usage = [memory_with_reordering, memory_without_reordering]
plt.bar(labels, memory_usage)
plt.ylabel("Memory Usage (bytes)")
plt.show()
