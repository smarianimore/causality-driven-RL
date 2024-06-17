import torch
from causalnex.structure import StructureModel

" ******************************************************************************************************************** "


def detach_and_cpu(tensor):
    """Detach a tensor from the computation graph and move it to the CPU."""
    return tensor.detach().cpu()


def detach_dict(d, func=detach_and_cpu):
    """Recursively apply a function to all tensors in a dictionary, list, or tuple."""
    if isinstance(d, dict):
        return {k: detach_dict(v, func) for k, v in d.items()}
    elif isinstance(d, list):
        return [detach_dict(v, func) for v in d]
    elif isinstance(d, tuple):
        return tuple(detach_dict(v, func) for v in d)
    elif isinstance(d, torch.Tensor):
        return func(d)
    else:
        return d


" ******************************************************************************************************************** "


def define_causal_graph(list_for_causal_graph: list):
    # Create a StructureModel
    sm = StructureModel()

    # Add edges to the StructureModel
    for relationship in list_for_causal_graph:
        cause, effect = relationship
        sm.add_edge(cause, effect)

    return sm


" ******************************************************************************************************************** "
