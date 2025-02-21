import torch
from torch import nn
from functools import partial
import numpy as np
from tqdm import tqdm


def make_beta_schedule(schedule, n_timestep, linear_start=1e-6, linear_end=1e-2):
    if schedule == 'linear':
        betas = np.linspace(linear_start, linear_end, n_timestep, dtype=np.float64)
    else:
        raise NotImplementedError(schedule)
    return betas


# gaussian diffusion trainer class
class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        denoise_fn,
        loss_type='l1',
        conditional=True,
    ):
        super().__init__()
        self.denoise_fn = denoise_fn
        self.conditional = conditional
        self.loss_type = loss_type
        self.vae = None
        self.vae_dem = None

    def set_loss(self, device):
        if self.loss_type == 'l1':
            self.loss_func = nn.L1Loss(reduction='sum').to(device)
        elif self.loss_type == 'l2':
            self.loss_func = nn.MSELoss(reduction='sum').to(device)
        else:
            raise NotImplementedError()
        
    def load_vae(self, vae, vae_dem, vae_checkpoint, vae_dem_checkpoint):
        self.vae = vae
        self.vae_dem = vae_dem

        vae_state = torch.load(vae_checkpoint, weights_only=True)
        self.vae.load_state_dict(vae_state["model_state_dict"])

        vae_dem_state = torch.load(vae_dem_checkpoint, weights_only=True)
        self.vae_dem.load_state_dict(vae_dem_state["model_state_dict"])

        self.vae.requires_grad_(False)
        self.vae_dem.requires_grad_(False)

    def set_new_noise_schedule(self, schedule_opt, device):
        to_torch = partial(torch.tensor, dtype=torch.float32, device=device)
        self.num_timesteps = schedule_opt['n_timestep']
        self.backward_start = schedule_opt.get('backward_start')

        betas = make_beta_schedule(
            schedule=schedule_opt['schedule'],
            n_timestep=schedule_opt['n_timestep'],
            linear_start=schedule_opt['linear_start'],
            linear_end=schedule_opt['linear_end'])
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)

        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas', to_torch(alphas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        
    
    @torch.no_grad()
    def sample_prev_timestep(self, xt, noise_pred, t):
        
        """ 
        Sample x_(t-1) given x_t and noise predicted by model.

        :param xt: Image tensor at timestep t of shape -> B x C x H x W
        :param noise_pred: Noise Predicted by model of shape -> B x C x H x W
        :param t: Current time step
        """
        # Original Image Prediction at timestep t
        # x0 = xt - torch.sqrt(1 - self.alphas_cumprod[t]) * noise_pred
        # x0 = x0 / torch.sqrt(self.alphas_cumprod[t])
        # x0 = torch.clamp(x0, -1., 1.) 
        
        # mean of x_(t-1)
        mean = xt - (1 - self.alphas[t]) * noise_pred / torch.sqrt(1 - self.alphas_cumprod[t])
        mean = mean / torch.sqrt(self.alphas[t])
        
        # only return mean
        if t == 0:
            return mean
        else:
            variance =  (1 - self.alphas_cumprod[t-1]) / (1 - self.alphas_cumprod[t])
            variance = variance * self.betas[t]
            sigma = variance ** 0.5
            z = torch.randn_like(xt)
            return mean + sigma * z
    
    @torch.no_grad()
    def add_noise(self, original, noise, t):
        
        """
        Adds noise to a batch of original images at time-step t.
        
        :param original: Input Image Tensor
        :param noise: Random Noise Tensor sampled from Normal Dist N(0, 1)
        :param t: timestep of the forward process of shape -> (B, )
        
        Note: time-step t may differ for each image inside the batch.
        """      
        sqrt_alpha_bar_t = self.sqrt_alphas_cumprod[t]
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alphas_cumprod[t]
        
        # Broadcast to multiply with the original image.
        sqrt_alpha_bar_t = sqrt_alpha_bar_t[:, None, None, None]
        sqrt_one_minus_alpha_bar_t = sqrt_one_minus_alpha_bar_t[:, None, None, None]
        
        # Return
        return sqrt_alpha_bar_t * original + sqrt_one_minus_alpha_bar_t * noise
    
    @torch.no_grad()
    def super_resolution(self, lr_img, dem):
        lr_img, _, _ = self.vae.encode(lr_img)
        dem, _, _ = self.vae_dem.encode(dem)
        if self.backward_start is None:
            backward_start = self.num_timesteps
            noise_img = torch.randn_like(lr_img)
        else:
            backward_start = self.backward_start
            noise = torch.randn_like(lr_img)
            t = torch.tensor(backward_start).repeat(lr_img.shape[0])
            noise_img = self.add_noise(lr_img, noise, t)

        for t in reversed(range(backward_start)):
            t_tensor = torch.tensor(t, dtype=torch.long).repeat(lr_img.shape[0]).to(lr_img.device)
            input_unet = torch.cat([noise_img, lr_img, dem], dim=1).to(lr_img.device)
            noise_pred = self.denoise_fn(input_unet, t_tensor)
            noise_img = self.sample_prev_timestep(noise_img.detach(), noise_pred, t)

        pred_imgs = self.vae.decode(noise_img)
        return pred_imgs
        

    def forward(self, x, *args, **kwargs):
        hr_img, lr_img, dem, = x['HR'], x['SR'], x['DEM']

        noise = torch.randn_like(hr_img)
        t = torch.randint(self.num_timesteps, (hr_img.shape[0],)).to(hr_img.device)
        noisy_images = self.add_noise(hr_img, noise, t)

        input_unet = torch.cat([noisy_images, lr_img, dem], dim=1)
        noise_pred = self.denoise_fn(input_unet, t)
        return self.loss_func(noise_pred, noise)
