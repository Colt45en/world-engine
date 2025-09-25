import torch

def build_batched_edges(batch_edges, lengths):
    """
    batch_edges: List of (edge_index_b: Long[2, E_b], edge_type_b: Long[E_b]) in sentence-local indices
    lengths:     List[int] token counts per sentence
    Returns global (edge_index, edge_type) with offsets applied.
    """
    offsets = torch.tensor([0] + list(torch.cumsum(torch.tensor(lengths[:-1]), dim=0)), dtype=torch.long)
    all_src, all_dst, all_type = [], [], []
    for b, (edge_index_b, edge_type_b) in enumerate(batch_edges):
        off = offsets[b].item()
        all_src.append(edge_index_b[0] + off)
        all_dst.append(edge_index_b[1] + off)
        all_type.append(edge_type_b)
    edge_index = torch.stack([torch.cat(all_src), torch.cat(all_dst)], dim=0)
    edge_type  = torch.cat(all_type)
    return edge_index, edge_type