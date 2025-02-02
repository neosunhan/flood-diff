import logging
from collections import OrderedDict
import torch
import torch.nn as nn
import os
import model.networks as networks
from model.base_model import BaseModel
logger = logging.getLogger('base')
from torch.amp import autocast
scaler = torch.amp.GradScaler('cuda')

class DDPM(BaseModel):
    def __init__(self, opt):
        super(DDPM, self).__init__(opt)
        # define network and load pretrained models
        self.netG = self.set_device(networks.define_generator(opt))
        self.schedule_phase = None
        self.saved_checkpoints = []

        if opt['latent']:
            self.netG.load_vae(opt['path']['vae_state'], opt['path']['vae_dem_state'])

        # set loss and load resume state
        self.set_loss()
        self.set_new_noise_schedule(opt['model']['beta_schedule']['train'], schedule_phase='train')
        self.amp = self.opt['amp']
        if self.opt['phase'] == 'train':
            self.netG.train()
            # find the parameters to optimize
            if opt['model']['finetune_norm']:
                optim_params = []
                for k, v in self.netG.named_parameters():
                    v.requires_grad = False
                    if k.find('transformer') >= 0:
                        v.requires_grad = True
                        v.data.zero_()
                        optim_params.append(v)
                        logger.info(f'Params [{k}] initialized to 0 and will optimize.')
            else:
                optim_params = [param for param in self.netG.parameters() if param.requires_grad]

            self.optG = torch.optim.Adam(optim_params, lr=opt['train']["optimizer"]["lr"])
            self.log_dict = OrderedDict()
        self.load_network()
        self.print_network()

    def feed_data(self, data):
        self.data = self.set_device(data)

    def optimize_parameters(self):
        self.optG.zero_grad()
        if self.amp:
            with torch.autocast('cuda'):
                l_pix = self.netG(self.data)
                b, c, h, w = self.data['HR'].shape
                l_pix = l_pix.sum() / int(b * c * h * w)
            scaler.scale(l_pix).backward()
            scaler.step(self.optG)
            scaler.update()
        else:
            l_pix = self.netG(self.data)
            # need to average in multi-gpu
            b, c, h, w = self.data['HR'].shape
            l_pix = l_pix.sum() / int(b * c * h * w)
            l_pix.backward()
            self.optG.step()

        # set log
        self.log_dict['l_pix'] = l_pix.item()

    def test(self):
        self.netG.eval()
        with torch.no_grad():
            if self.amp:
                with torch.autocast('cuda'):
                    if isinstance(self.netG, nn.DataParallel):
                        self.SR = self.netG.module.super_resolution(self.data['SR'], self.data['DEM'])
                    else:
                        self.SR = self.netG.super_resolution(self.data['SR'], self.data['DEM'])
            else:
                if isinstance(self.netG, nn.DataParallel):
                    self.SR = self.netG.module.super_resolution(self.data['SR'], self.data['DEM'])
                else:
                    self.SR = self.netG.super_resolution(self.data['SR'], self.data['DEM'])
        self.netG.train()

    def set_loss(self):
        if isinstance(self.netG, nn.DataParallel):
            self.netG.module.set_loss(self.device)
        else:
            self.netG.set_loss(self.device)

    def set_new_noise_schedule(self, schedule_opt, schedule_phase='train'):
        if self.schedule_phase is None or self.schedule_phase != schedule_phase:
            self.schedule_phase = schedule_phase
            if isinstance(self.netG, nn.DataParallel):
                self.netG.module.set_new_noise_schedule(schedule_opt, self.device)
            else:
                self.netG.set_new_noise_schedule(schedule_opt, self.device)

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_LR=True, sample=False):
        out_dict = OrderedDict()
        out_dict["filenames"] = self.data['filename']
        out_dict["profiles"] = self.data['profile']
        if sample:
            out_dict['SAM'] = self.SR.detach().float().cpu()
        else:
            out_dict['SR'] = self.SR.detach().float().cpu()
            out_dict['INF'] = self.data['SR'].detach().float().cpu()
            out_dict['HR'] = self.data['HR'].detach().float().cpu()
            if need_LR and 'LR' in self.data:
                out_dict['LR'] = self.data['LR'].detach().float().cpu()
            else:
                out_dict['LR'] = out_dict['INF']
        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel):
            net_struc_str = f'{self.netG.__class__.__name__} - {self.netG.module.__class__.__name__}'
        else:
            net_struc_str = f'{self.netG.__class__.__name__}'

        logger.info(f'Network G structure: {net_struc_str}, with parameters: {n:,d}')
        logger.info(s)

    def save_network(self, epoch, iter_step):
        if (iter_step, epoch) in self.saved_checkpoints:
            return
        gen_path = os.path.join(self.opt['path']['checkpoint'], f'I{iter_step}_E{epoch}_gen.pth')
        opt_path = os.path.join(self.opt['path']['checkpoint'], f'I{iter_step}_E{epoch}_opt.pth')
        # gen
        network = self.netG
        if isinstance(self.netG, nn.DataParallel):
            network = network.module
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, gen_path)
        # opt
        opt_state = {
            'epoch': epoch, 
            'iter': iter_step,
            'scheduler': None, 
            'optimizer': None
        }
        opt_state['optimizer'] = self.optG.state_dict()
        torch.save(opt_state, opt_path)

        logger.info('Saved model in [{:s}] ...'.format(gen_path))
        self.saved_checkpoints.append((iter_step, epoch))

    def load_network(self):
        load_path = self.opt['path']['resume_state']
        if load_path is not None:
            logger.info(f'Loading pretrained model for G [{load_path}] ...')
            gen_path = f'{load_path}_gen.pth'
            opt_path = f'{load_path}_opt.pth'
            # gen
            network = self.netG
            if isinstance(network, nn.DataParallel):
                network = network.module
            network.load_state_dict(torch.load(gen_path, weights_only=True), strict=(not self.opt['model']['finetune_norm']))
            # network.load_state_dict(torch.load(gen_path), strict=False)
            if self.opt['phase'] == 'train':
                # optimizer
                opt = torch.load(opt_path, weights_only=True)
                self.optG.load_state_dict(opt['optimizer'])
                self.begin_step = opt['iter']
                self.begin_epoch = opt['epoch']

        if self.opt['latent']:
            enc_dec_states = [self.opt['path'][checkpoint] for checkpoint in ('encoder_state', 'encoder_lr_state', 'encoder_dem_state', 'decoder_state')]
            if any(checkpoint is not None for checkpoint in enc_dec_states):
                self.netG.load_enc_dec(*enc_dec_states)