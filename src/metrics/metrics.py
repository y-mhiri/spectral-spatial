import torch


def RNMSE(A,B):
    assert A.shape == B.shape, "A and B must have the same shape"

    return torch.norm(A-B)/torch.prod(torch.tensor(A.shape)).sqrt()

def CC(A,B):
    assert A.shape == B.shape, "A and B must have the same shape"

    A = A - A.mean()
    B = B - B.mean()

    return (A*B).sum()/(torch.norm(A)*torch.norm(B))

def SNR(A,B):
    assert A.shape == B.shape, "A and B must have the same shape"

    return 10*torch.log10(1/(RNMSE(A,B)**2))

def SAM(A,B):
    assert A.shape == B.shape, "A and B must have the same shape"

    A = A/A.norm(dim=0)
    B = B/B.norm(dim=0)

    return torch.acos((A*B).sum(dim=0)).mean()



def compute_metrics(gt,est, numpy=False):

    if numpy==True:
        return {'RNMSE': RNMSE(gt,est).cpu().numpy().tolist(),
                'CC': CC(gt,est).cpu().numpy().tolist(),
                'SNR': SNR(gt,est).cpu().numpy().tolist(),
                'SAM': SAM(gt,est).cpu().numpy().tolist()}

    return {'RNMSE': RNMSE(gt,est),
            'CC': CC(gt,est),
            'SNR': SNR(gt,est),
            'SAM': SAM(gt,est)}