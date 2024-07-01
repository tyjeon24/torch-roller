import numpy as np
import pandas as pd
import torch


def roll(target, length: int):
    if isinstance(target, pd.DataFrame):
        tensor = torch.FloatTensor(target.values)
    elif isinstance(target, np.ndarray):
        tensor = torch.FloatTensor(target)
    elif not isinstance(target, torch.Tensor):
        raise TypeError("The input type should be pandas dataframe, numpy ndarray or torch tensor.")
    return tensor.unfold(0, length, 1).transpose(1, 2)


def unroll(target: torch.Tensor):
    if not isinstance(target, torch.Tensor):
        raise TypeError("The input type should be torch tensor.")

    num_features = target.shape[-1]
    concatenated_tensor = torch.cat([*target], dim=1)
    zero_pad = torch.empty(concatenated_tensor.shape) * float("nan")
    padded_tensor = torch.cat([zero_pad, concatenated_tensor, zero_pad], dim=1)
    strided_tensor = torch.as_strided(padded_tensor, padded_tensor.shape, (padded_tensor.stride(0) - num_features, 1))
    tensor_2d = strided_tensor.nanmedian(dim=0)
    tensor_2d_values = tensor_2d.values
    array_2d = tensor_2d_values.numpy()
    x = array_2d[~np.isnan(array_2d)].reshape(-1, num_features)
    return pd.DataFrame(x)
