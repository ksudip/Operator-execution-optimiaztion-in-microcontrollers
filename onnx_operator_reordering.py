
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

# Load the trained model
path__ = '/content/drive/MyDrive/my_model.h5'
model = tf.keras.models.load_model(path__)

# Create a TensorRT engine
trt_logger = trt.Logger(trt.Logger.WARNING)
trt_builder = trt.Builder(trt_logger)
trt_network = trt_builder.create_network()
trt_parser = trt.OnnxParser(trt_network, trt_logger)

# Convert the Keras model to ONNX format
onnx_model_path = 'my_model.onnx'
tf.saved_model.save(model, 'temp')
converter = tf.compat.v1.lite.TFLiteConverter.from_saved_model('temp')
converter.allow_custom_ops = True
tflite_model = converter.convert()
with open(onnx_model_path, 'wb') as f:
    f.write(tflite_model)

# Parse the ONNX model and build the TensorRT engine
with open(onnx_model_path, 'rb') as f:
    onnx_model = f.read()
trt_parser.parse(onnx_model)
trt_builder.max_batch_size = 1

config = trt_builder.create_builder_config()
config.max_workspace_size = 2 * 1024 * 1024 * 1024
# config = trt_builder.create_builder_config()

# # set the maximum workspace size
# max_workspace_size_bytes = 2 * 1024 * 1024 * 1024  # 2GB
# config.max_workspace_size = max_workspace_size_bytes

# create the TensorRT engine

# with trt.Runtime(trt_logger) as runtime:
#   trt_engine = runtime.deserialize_cuda_engine(plan)
# return trt_engine
# trt_builder.max_workspace_size = 1 << 30  # 1GB
trt_engine = trt_builder.buildICudaengine(trt_network,config)
# type(trt_engine)
# Create an execution context and allocate memory
# trt_engine.
# type
trt_context = trt_engine.create_execution_context_without_device_memory()
inputs, outputs, bindings = [], [], []
for binding in trt_engine:
    size = trt.volume(trt_engine.get_binding_shape(binding)) * trt_engine.max_batch_size
    dtype = trt.nptype(trt_engine.get_binding_dtype(binding))
    host_mem = cuda.pagelocked_empty(size, dtype)
    device_mem = cuda.mem_alloc(host_mem.nbytes)
    bindings.append(int(device_mem))
    if trt_engine.binding_is_input(binding):
        inputs.append((host_mem, device_mem))
    else:
        outputs.append((host_mem, device_mem))
stream = cuda.Stream()

# Define a function to run inference and measure memory usage
def run_inference(inputs, outputs, bindings, stream):
    # Transfer input data to device memory
    for inp, (host_mem, device_mem) in zip(inputs, bindings):
        np.copyto(host_mem, inp.reshape(-1))
        cuda.memcpy_htod_async(device_mem, host_mem, stream)
    # Run inference
    # trt_context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    # Transfer output data to host memory
    results = [np.empty_like(out[0]) for out in outputs]
    for out, (host_mem, device_mem) in zip(results, outputs):
        cuda.memcpy_dtoh_async(host_mem, device_mem, stream)
    # Synchronize the stream and return the output data
    stream.synchronize()
    return results

# Run inference and measure memory usage
prev_mem_usage = cuda.mem_get_info()[0]
inputs = [np.random.randn(*input_shape).astype(np.float32) for input_shape in model.input_shape]
outputs = run_inference(inputs, outputs, bindings, stream)
latest_mem_usage = cuda.mem_get_info()[0]

# Plot memory usage graphs
fig, ax = plt.subplots()
ax.plot([0, 1], [prev_mem_usage, latest_mem_usage])
ax.set_xlabel('Inference run')
ax.set_ylabel('Memory usage (bytes)')
plt.show()

# pip install keras2onnx

# Load the Keras model
path__ = '/content/drive/MyDrive/my_model.h5'
keras_model_path = path__
keras_model = load_model(keras_model_path)



# Convert the Keras model to ONNX format
onnx_model_path = 'my_model.onnx'
tf.saved_model.save(model, 'temp')
converter = tf.compat.v1.lite.TFLiteConverter.from_saved_model('temp')
converter.allow_custom_ops = True
tflite_model = converter.convert()
with open(onnx_model_path, 'wb') as f:
    f.write(tflite_model)
# :
# # Convert the Keras model to ONNX format
# onnx_model_path = "my_model.onnx"
# tf.keras.backend.set_learning_phase(0)
# onnx_model = onnx.convert_keras(keras_model, keras_model.name)
# onnx.save(onnx_model, onnx_model_path)

# Create a TensorRT builder and network from the ONNX model
builder = trt.Builder(trt.Logger(trt.Logger.INFO))
builder.max_batch_size = 1
# builder.max_workspace_size = 1 << 30
network = builder.create_network()
parser = trt.OnnxParser(network, builder.logger)
with open(onnx_model_path, 'rb') as model:
    parser.parse(model.read())
    
# Get the original memory usage without operator reordering
builder_config = builder.create_builder_config()
builder_config.flags = 1 << int(trt.BuilderFlag.FP16)
builder_config.max_workspace_size = 1 << 30
min_memory_without_reordering = builder.get_refittable_tensor_size(network, builder_config, 1)

# Get the memory usage with operator reordering
builder_config = builder.create_builder_config()
builder_config.flags = 1 << int(trt.BuilderFlag.FP16) | 1 << int(trt.BuilderFlag.INTE8)
builder_config.max_workspace_size = 1 << 30
builder_config.set_flag(trt.BuilderFlag.STRICT_TYPES)
min_memory_with_reordering = builder.refit_cuda_graph(network, builder_config)

# Plot the results
labels = ["Without Operator Reordering", "With Operator Reordering"]
memory_usages = [min_memory_without_reordering, min_memory_with_reordering]
plt.bar(labels, memory_usages)
plt.ylabel("Memory Usage (Bytes)")
plt.title("Memory Usage with and without Operator Reordering")
plt.show()
