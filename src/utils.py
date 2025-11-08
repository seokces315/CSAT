import os
import random
import numpy as np
import torch


# Seed configuration
def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Option
    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(True)


# Check if the current CUDA device supports bfloat16
def supports_bf16():
    if not torch.cuda.is_available():
        return False

    major, _ = torch.cuda.get_device_capability()
    return major >= 8
