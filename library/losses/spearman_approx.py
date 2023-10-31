import torch
from torch.functional import F

def relu_sign(X):
    return F.softsign(F.relu(X))
    

NONLIN_F = {
    'relu': F.relu,
    'softsign': F.softsign, #? DO NOT USE THIS ONE, DOES AND WILL NOT WORK
    'relu_sign': relu_sign
}

MIN_ORDER=1

def k_order_spearman_approx(
    Y:torch.tensor, 
    order:int=MIN_ORDER, 
    nonlin='relu', 
    X:torch.tensor=None
)->torch.tensor:
    
    """    
    Approximate the spearmann correlation metric as a loss function.
    Spearman Correlation computes how the rank of two random variables X and Y correlates.
    This function expects that the random variables X (ground truth) is sorted frow low to high and the values of Y is sorted according to Y.           

    Args:
        Y (torch.tensor): Prediction sorted according to the value of ground truth X.
        order (int): Order of spearman approximation.
        nonlin (str, optional): Name of the non-linear function for the approximation. Defaults to 'relu'.
        X (torch.tensor, optional): Sorted ground truth from low to high for checking purpose. It is disabled if None. Defaults to None.

    Returns:
        torch.tensor: loss
    """
    
    assert order >= 1, f"Order must be greater equal than zero!"
    assert order < len(Y), f"Order must be less than the number of Y values"
    
    nonlin_f = NONLIN_F[nonlin]
    
    if X is not None:
        assert (X.diff() >= 0).all(), "X is not (correctly) sorted!"
    
    loss = 0.0
    for i in range(MIN_ORDER, order+1):
        diff = Y[:-i] - Y[i:]
        loss += nonlin_f(diff).mean()
        
    return loss