import torch
import onnx
import onnxruntime
import sys
import os

project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_dir)

from src.model import MultiTaskBertModel
from src.data_loader import load_dataset
from src.utils import bert_config

checkpoint_path = "lit-wandb/a5yyvgve/checkpoints/epoch=0-step=35.ckpt"
checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))

# Extract the state dictionary
state_dict = checkpoint['state_dict']

# Initialize your model
config = bert_config()  # Assuming you have a function bert_config() to get your model configuration
dataset = load_dataset("training_dataset")  # Assuming you have a function load_dataset() to get your dataset
model = MultiTaskBertModel(config, dataset)

# Load the state dictionary into your model
model.load_state_dict(state_dict)

# Save the model's state dictionary as a bin file
torch.save(model.state_dict(), "pytorch_model.bin")