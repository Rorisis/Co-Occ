import torch

def find_first_and_last_nonzeros(tensor):
    """
    find first non zero value in a tensor along a D dim
    tensor: [N, D, C]
    """
    tensor_01 = (tensor != 0).all(dim=-1).float()
    
    idx = torch.arange(tensor.shape[1], 0, -1).to(tensor.device)[None, :]
    tmp = tensor_01 * idx
    max_indices = torch.argmax(tmp, dim=1) # [N]
    max_indices = torch.stack([torch.arange(tensor.shape[0]).to(tensor.device), max_indices], dim=-1)
    first_nonzeros = tensor[max_indices[..., 0], max_indices[..., 1]]
    
    idx = torch.arange(tensor.shape[1]).to(tensor.device)[None, :]
    tmp = tensor_01 * idx
    max_indices = torch.argmax(tmp, dim=1) # [N]
    max_indices = torch.stack([torch.arange(tensor.shape[0]).to(tensor.device), max_indices], dim=-1)
    last_nonzeros = tensor[max_indices[..., 0], max_indices[..., 1]]
    
    last_nonzeros = torch.maximum(last_nonzeros, first_nonzeros + 1e-5)
    valid_mask = (first_nonzeros != 0).all(dim=-1)
    return first_nonzeros, last_nonzeros, valid_mask


if __name__ == '__main__':
    tensor = torch.rand(2, 10, 3)
    tensor[tensor < 0.1] = 0
    print('tensor', tensor)
    first_nonzeros, last_nonzeros = find_first_and_last_nonzeros(tensor)
    print('first_nonzeros', first_nonzeros, first_nonzeros.shape)
    print('last_nonzeros', last_nonzeros, last_nonzeros.shape)