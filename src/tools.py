import torch
from torch_geometric.utils import structured_negative_sampling
def negative_sampling(edge_index, num_negatives):
    """
    Args:
        edge_index (torch.Tensor): 边关系
        num_negatives (int): 负采样数量
    Returns:
        torch.Tensor: 负采样物品ID
    """
    edges = structured_negative_sampling(edge_index)
    edges = torch.stack(edges, dim=0)
    return edges[2]


x = torch.tensor([[0, 1, 2, 3, 4, 5], 
                  [1, 2, 3, 4, 5, 0]])
print(negative_sampling(x, 2))
