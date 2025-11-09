import torch
def my_device() -> torch.device: 
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("gpu")
    else :
        return torch.device("cpu")
