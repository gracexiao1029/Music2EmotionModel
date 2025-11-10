import onnx

onnx_model = onnx.load("emotion_model.onnx")
print("\n=== ONNX Model Info ===")
for input_tensor in onnx_model.graph.input:
    dims = [d.dim_value for d in input_tensor.type.tensor_type.shape.dim]
    print(f"Input: {input_tensor.name}, shape={dims}")

for output_tensor in onnx_model.graph.output:
    dims = [d.dim_value for d in output_tensor.type.tensor_type.shape.dim]
    print(f"Output: {output_tensor.name}, shape={dims}")
print("========================\n")