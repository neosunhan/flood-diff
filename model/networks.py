import functools
import logging
import torch
import torch.nn as nn
from torch.nn import init
logger = logging.getLogger('base')

####################
# initialize
####################


def weights_init_normal(m, std=0.02):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.normal_(m.weight.data, 0.0, std)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0.0, std)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, std)  # BN also uses norm
        init.constant_(m.bias.data, 0.0)


def weights_init_kaiming(m, scale=1):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        m.weight.data *= scale
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        m.weight.data *= scale
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.constant_(m.weight.data, 1.0)
        init.constant_(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.orthogonal_(m.weight.data, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.orthogonal_(m.weight.data, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.constant_(m.weight.data, 1.0)
        init.constant_(m.bias.data, 0.0)


def init_weights(net, init_type='kaiming', scale=1, std=0.02):
    # scale for 'kaiming', std for 'normal'.
    logger.info('Initialization method [{:s}]'.format(init_type))
    if init_type == 'normal':
        weights_init_normal_ = functools.partial(weights_init_normal, std=std)
        net.apply(weights_init_normal_)
    elif init_type == 'kaiming':
        weights_init_kaiming_ = functools.partial(weights_init_kaiming, scale=scale)
        net.apply(weights_init_kaiming_)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError(f'initialization method [{init_type}] not implemented')


####################
# define network
####################


# Generator
def define_generator(opt):
    model_opt = opt['model']
    if model_opt['which_model_G'] == 'ddpm':
        from model.ddpm_modules import diffusion, unet, autoencoder
    elif model_opt['which_model_G'] == 'sr3':
        from model.sr3_modules import diffusion, unet, autoencoder

    model = unet.UNet(
        in_channel=model_opt['unet']['in_channel'],
        out_channel=model_opt['unet']['out_channel'],
        inner_channel=model_opt['unet']['inner_channel'],
        norm_groups=model_opt['unet']['norm_groups'],
        channel_mults=model_opt['unet']['channel_multiplier'],
        attn_res=model_opt['unet']['attn_res'],
        res_blocks=model_opt['unet']['res_blocks'],
        dropout=model_opt['unet']['dropout'],
        image_size=model_opt['diffusion']['image_size']
    )

    vae, vae_dem = None, None
    if opt["latent"] and opt["phase"] == 'test':
        vae = autoencoder.VAE()
        if model_opt['dem']:
            vae_dem = autoencoder.VAE()
        
    net_generator = diffusion.GaussianDiffusion(
        model,
        loss_type=model_opt['diffusion']['loss_type'],
        conditional=model_opt['diffusion']['conditional'],
        dem=model_opt['dem'],
    )

    if opt['phase'] == 'train':
        # init_weights(net_generator, init_type='kaiming', scale=0.1)
        init_weights(net_generator, init_type='orthogonal')

    if opt['distributed']:
        assert torch.cuda.is_available()
        net_generator = nn.DataParallel(net_generator)
        if vae is not None:
            vae = nn.DataParallel(vae)
        if vae_dem is not None:
            vae_dem = nn.DataParallel(vae_dem)

    return net_generator, (vae, vae_dem)
