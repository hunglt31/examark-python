import onnx
from onnx import helper, numpy_helper, TensorProto
import numpy as np

# Load the original model
model = onnx.load("./assets/models/content/weights/best.onnx")
graph = model.graph
input0 = helper.make_tensor_value_info("input0", TensorProto.UINT8, [8, 640, 640, 3])
graph.input.remove(graph.input[0])
graph.input.insert(0, input0)

# --- Step 1: Add pre-process layers ---
# Cast layer
cast_node = helper.make_node(
    "Cast",
    inputs=["input0"],
    outputs=["input_float"],
    to=TensorProto.FLOAT
)
graph.node.insert(0, cast_node)

# Transpose layer
transpose_node = helper.make_node(
    "Transpose",
    inputs=["input_float"],  
    outputs=["input_nchw"], 
    perm=[0, 3, 1, 2] 
)
graph.node.insert(1, transpose_node)  

# Normalize layer
scale_value = np.array(1.0 / 255.0, dtype=np.float32)

graph.initializer.append(numpy_helper.from_array(scale_value, name="scale"))
mul_node = helper.make_node("Mul",
                            inputs=["input_nchw", "scale"],
                            outputs=["normalized_image"])
graph.node.insert(2, mul_node)

# Swap color layers
split_node = helper.make_node(
    'Split',
    inputs=['normalized_image'],
    outputs=['b_channel', 'g_channel', 'r_channel'],
    axis=1
)
graph.node.insert(3, split_node)

concat_node = helper.make_node(
    'Concat',
    inputs=['r_channel', 'g_channel', 'b_channel'],
    outputs=['rgb_image'],
    axis=1
)
graph.node.insert(4, concat_node)

# --- Step 2: Update input ---
for node in graph.node:
    for i, inp in enumerate(node.input):
        if inp == "images":
            node.input[i] = "rgb_image"
            print(f"Node {node.name} input updated to 'rgb_image'")

# --- Step 3: Add post-process layers ---
# Get model output
model_output_name = graph.output[0].name
shuffle_node = helper.make_node(
    "Transpose",
    inputs=[model_output_name],  
    outputs=["shuffle_output"], 
    perm=[0, 2, 1] 
)
graph.node.append(shuffle_node)

# Add bboxes layer
MAX_INT = np.iinfo(np.int64).max
print(MAX_INT)
bbox_starts = np.array([0, 0, 0], dtype=np.int64)
bbox_ends = np.array([8, 8400, 4], dtype=np.int64)
bbox_axes = np.array([0, 1, 2], dtype=np.int64)
bbox_steps = np.array([1, 1, 1], dtype=np.int64)

graph.initializer.extend([
    numpy_helper.from_array(bbox_starts, name="bbox_starts"),
    numpy_helper.from_array(bbox_ends, name="bbox_ends"),
    numpy_helper.from_array(bbox_axes, name="bbox_axes"),
    numpy_helper.from_array(bbox_steps, name="bbox_steps")
])

bbox_slice_node = helper.make_node(
    "Slice",
    inputs=["shuffle_output", "bbox_starts", "bbox_ends", "bbox_axes", "bbox_steps"],
    outputs=["bbox_slice"]
)
graph.node.append(bbox_slice_node)

# Add scores layer
score_starts = np.array([0, 0, 4], dtype=np.int64)
score_ends = np.array([8, 8400, 6], dtype=np.int64)
score_axes = np.array([0, 1, 2], dtype=np.int64)
score_steps = np.array([1, 1, 1], dtype=np.int64)

graph.initializer.extend([
    numpy_helper.from_array(score_starts, name="score_starts"),
    numpy_helper.from_array(score_ends, name="score_ends"),
    numpy_helper.from_array(score_axes, name="score_axes"),
    numpy_helper.from_array(score_steps, name="score_steps")
])

score_slice_node = helper.make_node(
    "Slice",
    inputs=["shuffle_output", "score_starts", "score_ends", "score_axes", "score_steps"],
    outputs=["score_slice"]
)
graph.node.append(score_slice_node)

# --- Step 4: Replace outputs and save ---
graph.output.remove(graph.output[0])
bbox_output = helper.make_tensor_value_info(
    "bbox_slice",  
    TensorProto.FLOAT,  
    [8, 8400, 4] 
)
graph.output.append(bbox_output)

score_output = helper.make_tensor_value_info(
    "score_slice", 
    TensorProto.FLOAT,  
    [8, 8400, 2] 
)
graph.output.append(score_output)

# --- Step 5: Save model --- 
model.opset_import[0].version = max(model.opset_import[0].version, 11)
onnx.save(model, "./assets/models/content.onnx")
print(onnx.helper.printable_graph(model.graph))
