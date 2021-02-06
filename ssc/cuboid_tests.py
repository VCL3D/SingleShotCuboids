import torch

__TESTS__ = { }

def get_tests():
    for k, v in __TESTS__.items():
        yield k, v

def register(test):
    name = getattr(test, "__name__", test.__class__.__name__)
    if name in __TESTS__:
        raise RuntimeError(f'Encountered a test name collision "{name}"')
    __TESTS__[name] = test
    return test

@register
def test1():
    unaligned = torch.Tensor([
        [-0.9743, -0.2491],
        [-0.5582, -0.1589],
        [-0.2159, -0.1677],
        [ 0.1593, -0.3002],
        [-0.9743,  0.2714],
        [-0.5582,  0.1491],
        [-0.2159,  0.1377],
        [ 0.1593,  0.2662]
    ])
    return torch.chunk(unaligned.unsqueeze(0), 2, dim=-2)

@register
def test2():
    unaligned = torch.Tensor([
        [-0.9791, -0.3541],
        [-0.5517, -0.1809],
        [-0.2686, -0.1822],
        [ 0.2237, -0.4052],
        [-0.9791,  0.2998],
        [-0.5517,  0.1574],
        [-0.2686,  0.1590],
        [ 0.2237,  0.3572]
    ])
    return torch.chunk(unaligned.unsqueeze(0), 2, dim=-2)

@register
def test3():
    unaligned = torch.Tensor([
        [-0.9882, -0.0736],
        [-0.7437, -0.0662],
        [-0.2522, -0.0995],
        [ 0.2172, -0.1664],
        [-0.9882,  0.2589],
        [-0.7437,  0.1070],
        [-0.2522,  0.1835],
        [ 0.2172,  0.2766]
    ])
    return torch.chunk(unaligned.unsqueeze(0), 2, dim=-2)

@register
def test4():
    unaligned = torch.Tensor([
        [-0.9846, -0.5418],
        [ 0.0205, -0.2194],
        [ 0.2252, -0.2062],
        [ 0.5488, -0.2653],
        [-0.9846,  0.4780],
        [ 0.0205,  0.2186],
        [ 0.2252,  0.1963],
        [ 0.5488,  0.1743]
    ])
    return torch.chunk(unaligned.unsqueeze(0), 2, dim=-2)

@register
def test5():
    unaligned = torch.Tensor([
        [-0.9825, -0.1939],
        [-0.7482, -0.1596],
        [-0.1407, -0.2930],
        [ 0.2218, -0.4294],
        [-0.9825,  0.1199],
        [-0.7482,  0.1063],
        [-0.1407,  0.1077],
        [ 0.2218,  0.1338]
    ])
    return torch.chunk(unaligned.unsqueeze(0), 2, dim=-2)

@register
def test6():
    unaligned = torch.Tensor([
        [-0.9225, -0.2832],
        [-0.7755, -0.2755],
        [-0.1321, -0.4738],
        [ 0.2905, -0.6044],
        [-0.9225,  0.2201],
        [-0.7755,  0.2172],
        [-0.1321,  0.4091],
        [ 0.2905,  0.5314]
    ])
    return torch.chunk(unaligned.unsqueeze(0), 2, dim=-2)

@register
def test7():
    unaligned = torch.Tensor([
        [-0.9369, -0.4814],
        [-0.4945, -0.1954],
        [-0.3008, -0.2248],
        [ 0.2783, -0.4565],
        [-0.9369,  0.4577],
        [-0.4945,  0.1532],
        [-0.3008,  0.1621],
        [ 0.2783,  0.3577]
    ])
    return torch.chunk(unaligned.unsqueeze(0), 2, dim=-2)