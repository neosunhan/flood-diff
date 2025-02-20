import math
import torch
from torch import nn
from functools import partial
import numpy as np


def make_beta_schedule(schedule, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
    if schedule == 'quad':
        betas = np.linspace(linear_start ** 0.5, linear_end ** 0.5, n_timestep, dtype=np.float64) ** 2
    elif schedule == 'linear':
        betas = np.linspace(linear_start, linear_end, n_timestep, dtype=np.float64)
    elif schedule == 'const':
        betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    elif schedule == 'jsd':  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1. / np.linspace(n_timestep, 1, n_timestep, dtype=np.float64)
    elif schedule == "cosine":
        timesteps = torch.arange(n_timestep + 1, dtype=torch.float64) / n_timestep + cosine_s
        alphas = timesteps / (1 + cosine_s) * math.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = betas.clamp(max=0.999)
    else:
        raise NotImplementedError(f"{schedule} noise schedule not implemented")
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
        self.loss_type = loss_type
        self.conditional = conditional
        self.vae = None
        self.vae_dem = None

    def set_loss(self, device):
        if self.loss_type == 'l1':
            self.loss_func = nn.L1Loss(reduction='sum').to(device)
        elif self.loss_type == 'l2':
            self.loss_func = nn.MSELoss(reduction='sum').to(device)
        else:
            raise NotImplementedError(f"{self.loss_type} loss not implemented")
        
    def load_vae(self, vae, vae_dem, vae_checkpoint, vae_dem_checkpoint):
        # assert self.vae is not None and self.vae_dem is not None, "VAE missing"
        self.vae = vae
        self.vae_dem = vae_dem

        if "enc_dec" in vae_checkpoint:
            dec_checkpoint = vae_checkpoint.replace("encoder", "decoder")
            enc_checkpoint = vae_checkpoint
            enc_state = torch.load(enc_checkpoint, weights_only=True)
            self.vae.encoder = torch.nn.DataParallel(self.vae.encoder)
            self.vae.encoder.load_state_dict(enc_state["model_state_dict"])
            dec_state = torch.load(dec_checkpoint, weights_only=True)
            self.vae.decoder = torch.nn.DataParallel(self.vae.decoder)
            self.vae.decoder.load_state_dict(dec_state["model_state_dict"])

            enc_dem_state = torch.load(vae_dem_checkpoint, weights_only=True)
            self.vae_dem.encoder = torch.nn.DataParallel(self.vae_dem.encoder)
            self.vae_dem.encoder.load_state_dict(enc_dem_state["model_state_dict"])
            self.vae_dem.decoder = torch.nn.DataParallel(self.vae_dem.decoder)
        else:
            vae_state = torch.load(vae_checkpoint, weights_only=True)
            self.vae.load_state_dict(vae_state["model_state_dict"])

            vae_dem_state = torch.load(vae_dem_checkpoint, weights_only=True)
            self.vae_dem.load_state_dict(vae_dem_state["model_state_dict"])

        self.vae.requires_grad_(False)
        self.vae_dem.requires_grad_(False)

    def set_new_noise_schedule(self, schedule_opt, device):
        self.num_timesteps = schedule_opt['n_timestep']
        to_torch = partial(torch.tensor, dtype=torch.float32, device=device)

        betas = make_beta_schedule(
            schedule=schedule_opt['schedule'],
            n_timestep=schedule_opt['n_timestep'],
            linear_start=schedule_opt['linear_start'],
            linear_end=schedule_opt['linear_end'])
        betas = betas.detach().cpu().numpy() if isinstance(betas, torch.Tensor) else betas
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])
        self.sqrt_alphas_cumprod_prev = np.sqrt(np.append(1., alphas_cumprod))

        self.backward_start = schedule_opt.get('backward_start')
        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior p(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance', to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch((1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

    def predict_start_from_noise(self, x_t, t, noise):
        return self.sqrt_recip_alphas_cumprod[t] * x_t - self.sqrt_recipm1_alphas_cumprod[t] * noise

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = self.posterior_mean_coef1[t] * x_start + self.posterior_mean_coef2[t] * x_t
        posterior_log_variance_clipped = self.posterior_log_variance_clipped[t]
        return posterior_mean, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, clip_denoised, condition_x, dem):
        batch_size = x.shape[0]
        noise_level = torch.FloatTensor([self.sqrt_alphas_cumprod_prev[t+1]]).repeat(batch_size, 1).to(x.device)
        x_recon = self.predict_start_from_noise(
            x, 
            t=t, 
            noise=self.denoise_fn(torch.cat([condition_x, x, dem], dim=1), noise_level)
        )

        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, t, condition_x, dem, clip_denoised=True):
        model_mean, model_log_variance = self.p_mean_variance(
            x=x, t=t, clip_denoised=clip_denoised, condition_x=condition_x, dem=dem)
        noise = torch.randn_like(x) if t > 0 else torch.zeros_like(x)
        return model_mean + noise * (0.5 * model_log_variance).exp()

    @torch.no_grad()
    def p_sample_loop(self, x_in, dem):
        device = self.betas.device
        x = x_in
        if self.vae is not None and self.vae_dem is not None:
            x, _, _ = self.vae.encode(x)
            dem, _, _ = self.vae_dem.encode(dem)
        shape = x.shape
        if self.backward_start is None:
            backward_start = self.num_timesteps
            img = torch.randn(shape, device=device)
        else:
            x_start = x
            [b, c, h, w] = x_start.shape
            backward_start = self.backward_start
            continuous_sqrt_alpha_cumprod = torch.FloatTensor(
                np.random.uniform(
                    self.sqrt_alphas_cumprod_prev[backward_start-1],
                    self.sqrt_alphas_cumprod_prev[backward_start],
                    size=b
                )
            ).to(x_start.device)
            continuous_sqrt_alpha_cumprod = continuous_sqrt_alpha_cumprod.view(b, -1)
            noise = torch.randn_like(x_start)
            img = self.q_sample(
                x_start=x_start, continuous_sqrt_alpha_cumprod=continuous_sqrt_alpha_cumprod.view(-1, 1, 1, 1), noise=noise)            
            
        for i in reversed(range(backward_start)):
            img = self.p_sample(img, i, condition_x=x, dem=dem)
        
        if self.vae is not None and self.vae_dem is not None:
            img = self.vae.decode(img)

        return img

    @torch.no_grad()
    def super_resolution(self, x_in, dem):
        return self.p_sample_loop(x_in, dem)

    def q_sample(self, x_start, continuous_sqrt_alpha_cumprod, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        # random gama
        return continuous_sqrt_alpha_cumprod * x_start + (1 - continuous_sqrt_alpha_cumprod ** 2).sqrt() * noise

    def p_losses(self, x_in, noise=None):
        x_start = x_in['HR']
        lr_imgs = x_in['SR']
        dem_imgs = x_in["DEM"]

        if self.vae is not None and self.vae_dem is not None:
            x_start, _, _ = self.vae.encode(x_start)
            lr_imgs, _, _ = self.vae.encode(lr_imgs)
            dem_imgs, _, _ = self.vae_dem.encode(dem_imgs)

        [b, c, h, w] = x_start.shape
        t = np.random.randint(1, self.num_timesteps + 1)
        continuous_sqrt_alpha_cumprod = torch.FloatTensor(
            np.random.uniform(
                self.sqrt_alphas_cumprod_prev[t-1],
                self.sqrt_alphas_cumprod_prev[t],
                size=b
            )
        ).to(x_start.device)
        continuous_sqrt_alpha_cumprod = continuous_sqrt_alpha_cumprod.view(b, -1)

        if noise is None:
            noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(
            x_start=x_start, continuous_sqrt_alpha_cumprod=continuous_sqrt_alpha_cumprod.view(-1, 1, 1, 1), noise=noise)

        if not self.conditional:
            x_recon = self.denoise_fn(x_noisy, continuous_sqrt_alpha_cumprod)
        else:
            x_recon = self.denoise_fn(
                torch.cat([lr_imgs, x_noisy, dem_imgs], dim=1), continuous_sqrt_alpha_cumprod)

        loss = self.loss_func(noise, x_recon)
        return loss

    def forward(self, x, *args, **kwargs):
        return self.p_losses(x, *args, **kwargs)
