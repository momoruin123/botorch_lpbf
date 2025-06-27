import torch

if torch.cuda.is_available():
    print("✅ CUDA is available!")
    print("GPU Name:", torch.cuda.get_device_name(0))
    print("Number of GPUs:", torch.cuda.device_count())
    print("Current device:", torch.cuda.current_device())
else:
    print("❌ CUDA is NOT available. Running on CPU.")
