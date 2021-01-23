import torch
import torch.nn.functional as F

# DCGAN loss


def loss_dcgan_dis(dis_fake, dis_real, clip):
    L1 = torch.mean(F.softplus(-dis_real))
    L2 = torch.mean(F.softplus(dis_fake))
    return L1, L2


def loss_dcgan_gen(dis_fake):
    loss = torch.mean(F.softplus(-dis_fake))
    return loss


# Hinge Loss
def loss_hinge_dis(dis_fake, dis_real, clip):
    loss_real = torch.clamp(torch.mean(F.relu(1. - dis_real)), -clip, clip)
    loss_fake = torch.clamp(torch.mean(F.relu(1. + dis_fake)), -clip, clip)
    return loss_real, loss_fake
# def loss_hinge_dis(dis_fake, dis_real): # This version returns a single loss
    # loss = torch.mean(F.relu(1. - dis_real))
    # loss += torch.mean(F.relu(1. + dis_fake))
    # return loss


def loss_hinge_gen(dis_fake):
    loss = -torch.mean(dis_fake)
    return loss


def vae_recon_loss(G_z, x):
    """vae reconstruction and kld loss"""
    # reconstruction
    recon_loss = F.mse_loss(G_z, x)
    # print(f"chcek inside losses {type(recon_loss)} {recon_loss} {recon_loss.item()}")
    return recon_loss

def vae_kld_loss(mu, log_var, clip):
    # kld loss
    log_var = torch.clamp(log_var, -clip, clip)
    kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
    return kld_loss


def mask_complement_loss(mask_x):
    """mask_x: [n, k, h, w]"""
    return (mask_x.sum(1) - 1.).abs().mean()

def mask_independent_loss(mask_x, device='cuda'):
    """mask_x: [n, k, h, w]
    """
    n, k, h, w = mask_x.shape
    mask_x = mask_x.view(n, k, -1)
    mask_x_TT = torch.matmul(mask_x, torch.transpose(mask_x, 1, 2)) # n, k, k

    diag_mask = torch.eye(k).to(device) # k, k
    positive_term = mask_x_TT * diag_mask[None, :, :] # n, k, k
    positive_term = positive_term.sum((1, 2)) # n

    denom_term = mask_x_TT.sum((1, 2)) # n 

    contrastive_loss = - (positive_term / denom_term).mean() # to maximize

    return contrastive_loss

def mask_loss(mask_x_all, complement_weight=1, contrastive_weight=1, device='cuda'):
    """mask_x_all: [[n, k, h, w],...]
    regularize mask x to be
    1) sum in k dim is equal to 1
    2) let every mask to be different with each other
    """
    
    # sum to 1 (minimize)
    complement_loss = torch.cat([mask_complement_loss(mask_x).unsqueeze(0) for mask_x in mask_x_all]).sum()
    # orthogonoal by contrastive (maximize)
    contrastive_loss = torch.cat([mask_independent_loss(mask_x, device).unsqueeze(0) for mask_x in mask_x_all]).sum()

    losses = complement_loss * complement_weight + contrastive_loss * contrastive_weight

    return losses



# Default to hinge loss
generator_loss = loss_hinge_gen
discriminator_loss = loss_hinge_dis
