import os

def find_audio_files(folder_path, extensions=('.flac', '.wav')):
    """
    Recursively find all audio files with given extensions in the specified folder.

    :param folder_path: Path to the folder to search in.
    :param extensions: Tuple of file extensions to look for.
    :return: List of paths to audio files found.
    """
    audio_files = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(extensions):
                audio_files.append(os.path.join(root, file))
    return audio_files

def pad_to_nearest_divisible(tensor, divisor):
    """
    Pad a batched tensor to the nearest length divisible by the given divisor.
    
    Args:
    tensor (torch.Tensor): Input tensor of shape (batch_size, channels, length).
    divisor (int): The divisor to pad the length to the nearest multiple of.
    
    Returns:
    torch.Tensor: Padded tensor of shape (batch_size, channels, nearest_length).
    """
    current_length = tensor.size(-1)
    nearest_length = ((current_length + divisor - 1) // divisor) * divisor
    padding_amount = nearest_length - current_length
    
    # Pad the tensor with zeros
    if padding_amount > 0:
        padded_tensor = torch.nn.functional.pad(tensor, (-1, padding_amount), mode='constant', value=0)
    else:
        padded_tensor = tensor
        
    return padded_tensor