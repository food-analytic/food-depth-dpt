import torch


def SILog(yhat, y, mask, lambda_=0):
    d = torch.log(yhat[mask] + 1e-8) - torch.log(y[mask])
    return (d**2).mean() - lambda_ * d.mean() ** 2
