import torch.onnx
from pytorch_model import Classifier, BasicBlock

# Check if GPU is available, otherwise use CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the PyTorch model weights
model_path = 'models/pytorch_model_weights.pth'
mtailor = Classifier(BasicBlock, [2, 2, 2, 2])
mtailor.load_state_dict(torch.load(model_path, map_location=device))

# Set the model to evaluation mode
mtailor.eval()

# Define the input size for the model
input_size = (1, 3, 224, 224)

# Convert the PyTorch model to ONNX
torch.onnx.export(
    model=mtailor, 
    args=torch.randn(input_size), 
    f='models/pytorch_model_weights.onnx', 
    verbose=True
)
