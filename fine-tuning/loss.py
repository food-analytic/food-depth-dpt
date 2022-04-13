import torch


def SILog(yhat, y, mask, lambda_term=0):
    d = torch.log(yhat[mask] + 1e-8) - torch.log(y[mask] + 1e-8)
    return (d ** 2).mean() - lambda_term * d.mean() ** 2
