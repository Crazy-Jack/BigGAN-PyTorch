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
    return recon_loss

def vae_kld_loss(mu, log_var, clip):
    # kld loss
    log_var = torch.clamp(log_var, -clip, clip)
    kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
    return kld_loss



# Default to hinge loss
generator_loss = loss_hinge_gen
discriminator_loss = loss_hinge_dis
