from torch import optim

OPTIMIZERS = {
    'sgd': optim.SGD,
    'adam': optim.Adam,
}

def select_optimizer(name, parameters, lr, kwargs=None):
    if kwargs is None:
        kwargs = {}
        
    try:
        optim_cls = OPTIMIZERS[name.lower()]
    except:
        raise ValueError(f"Invalid optimizer name: {name}")
    
    optim = optim_cls(parameters, lr=lr, **kwargs)
    
    return optim