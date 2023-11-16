from src.colour.soft_encode import soft_encode
import torch.nn.functional as F

def ab_to_z(ab,hull):
    ab = F.interpolate(ab, scale_factor=0.25)
    ab = ab.permute(0, 2, 3, 1)
    return soft_encode(ab, centroids=hull)