import torch
print("cuda runtime:", torch.version.cuda)
print("cuda available?", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))