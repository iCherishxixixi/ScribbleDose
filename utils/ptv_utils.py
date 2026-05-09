import torch
import torch.nn.functional as F

def dose_ranking_loss(coarse_seg, dose):
    prob = torch.softmax(coarse_seg, dim=1)

    ptv_mask = prob[:,0]
    oar_mask = prob[:,1]

    ptv_dose = (dose * ptv_mask).mean()
    oar_dose = (dose * oar_mask).mean()

    return F.relu(oar_dose - ptv_dose)