"""Export embedding model to ONNX format for cross-platform consistency."""
import sys
sys.path.insert(0, 'training')

import torch
import numpy as np
from model import CardEmbeddingModel

print("Loading PyTorch model...")
checkpoint = torch.load('training/checkpoints/final_model.pt', map_location='cpu', weights_only=False)
model = CardEmbeddingModel(
    num_classes=checkpoint["num_classes"],
    embedding_dim=checkpoint["embedding_dim"],
    pretrained=False,
)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print("Exporting to ONNX...")
dummy_input = torch.randn(1, 3, 224, 224)

# Export the embedding extraction part
torch.onnx.export(
    model,
    dummy_input,
    "training/output/card_embedding_model.onnx",
    export_params=True,
    opset_version=14,
    do_constant_folding=True,
    input_names=['image'],
    output_names=['embedding'],
    dynamic_axes={
        'image': {0: 'batch_size'},
        'embedding': {0: 'batch_size'}
    }
)

print("Verifying ONNX model...")
import onnxruntime as ort

# Test ONNX model
session = ort.InferenceSession("training/output/card_embedding_model.onnx")
test_input = np.random.randn(1, 3, 224, 224).astype(np.float32)

# PyTorch output
with torch.no_grad():
    pytorch_out = model.get_embedding(torch.from_numpy(test_input)).numpy()

# ONNX output
onnx_out = session.run(None, {'image': test_input})[0]

# Compare
diff = np.abs(pytorch_out - onnx_out).max()
print(f"Max difference between PyTorch and ONNX: {diff:.8f}")

if diff < 1e-5:
    print("SUCCESS: ONNX model matches PyTorch!")
else:
    print("WARNING: Significant difference detected")

print(f"\nONNX model saved to: training/output/card_embedding_model.onnx")
