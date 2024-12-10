import torch
import data as Data
import model as Model
import argparse
import logging
import core.logger as Logger
import core.metrics as Metrics
from core.wandb_logger import WandbLogger
from tqdm import tqdm
from tensorboardX import SummaryWriter
from torch.nn.parallel import DataParallel
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/sr_sr3_16_128.json',
                        help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['train', 'test'],
                        help='Run either train(training) or test(generation)', default='train')
    parser.add_argument('-debug', '-d', action='store_true')
    parser.add_argument('-enable_wandb', action='store_true')
    parser.add_argument('-log_wandb_ckpt', action='store_true')
    parser.add_argument('-log_eval', action='store_true')
    parser.add_argument('-output_dir', type=str)

    # parse configs
    args = parser.parse_args()
    opt = Logger.parse(args)
    # Convert to NoneDict, which return None for missing key.
    opt = Logger.dict_to_nonedict(opt)

    # logging
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    Logger.setup_logger(None, opt['path']['log'],
                        'train', level=logging.INFO, screen=True)
    Logger.setup_logger('test', opt['path']['log'], 'test', level=logging.INFO)
    logger = logging.getLogger('base')
    logger.info(Logger.dict2str(opt))
    tb_logger = SummaryWriter(log_dir=opt['path']['tb_logger'])

    # Initialize WandbLogger
    if opt['enable_wandb']:
        import wandb
        wandb_logger = WandbLogger(opt)
        wandb.define_metric('validation/val_step')
        wandb.define_metric('epoch')
        wandb.define_metric("validation/*", step_metric="val_step")
        val_step = 0
    else:
        wandb_logger = None

    # dataset
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train' == args.phase:
            train_set = Data.create_dataset(dataset_opt, phase, opt["datasets"]["meta"])
            train_loader = Data.create_dataloader(train_set, dataset_opt, phase)
        elif phase == 'test':
            test_set = Data.create_dataset(dataset_opt, phase, opt["datasets"]["meta"])
            test_loader = Data.create_dataloader(test_set, dataset_opt, phase)
    logger.info('Initial Dataset Finished')

    # model
    diffusion = Model.create_model(opt)
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        logger.info(f"Using {num_gpus} GPUs")
        diffusion = DataParallel(diffusion)
    logger.info('Initial Model Finished')

    # Train
    current_step = diffusion.begin_step
    current_epoch = diffusion.begin_epoch
    n_iter = opt['train']['n_iter']

    diffusion.set_new_noise_schedule(opt['model']['beta_schedule'][opt['phase']], schedule_phase=opt['phase'])
    if opt['phase'] == 'train':
        if opt['path']['resume_state']:
            logger.info(f'Resuming training from epoch: {current_epoch}, iter: {current_step}.')
        while current_step < n_iter:
            current_epoch += 1
            for train_data in train_loader:
                current_step += 1
                if current_step > n_iter:
                    break
                diffusion.feed_data(train_data)
                diffusion.optimize_parameters()
                # log
                if current_step % opt['train']['print_freq'] == 0:
                    logs = diffusion.get_current_log()
                    message = '<epoch:{:3d}, iter:{:8,d}> '.format(
                        current_epoch, current_step)
                    for k, v in logs.items():
                        message += '{:s}: {:.4e} '.format(k, v)
                        tb_logger.add_scalar(k, v, current_step)
                    logger.info(message)

                    if wandb_logger:
                        wandb_logger.log_metrics(logs)

                # validation
                if current_step % opt['train']['val_freq'] == 0:
                    avg_psnr = 0.0
                    avg_mse_input, avg_mse_predicted = 0.0, 0.0
                    mse_loss = torch.nn.MSELoss()
                    idx = 0
                    # result_path = f'{opt['path']['results']}/{current_epoch}'
                    # os.makedirs(result_path, exist_ok=True)

                    diffusion.set_new_noise_schedule(opt['model']['beta_schedule']['test'], schedule_phase='test')
                    for val_data in test_loader:
                        idx += 1
                        diffusion.feed_data(val_data)
                        diffusion.test(continous=False)
                        visuals = diffusion.get_current_visuals()
                        sr_image = visuals['SR'].squeeze().clamp(
                            opt["datasets"]["meta"]["norm_min"], 
                            opt["datasets"]["meta"]["norm_max"]
                        )

                        loss_input = mse_loss(visuals['INF'].squeeze(), visuals['HR'].squeeze())
                        loss_predicted = mse_loss(sr_image, visuals['HR'].squeeze())

                        avg_mse_input += loss_input.item()
                        avg_mse_predicted += loss_predicted.item()

                        # sr_img = Metrics.tensor2img(visuals['SR'])  # uint8
                        # hr_img = Metrics.tensor2img(visuals['HR'])  # uint8
                        # lr_img = Metrics.tensor2img(visuals['LR'])  # uint8
                        # fake_img = Metrics.tensor2img(visuals['INF'])  # uint8

                        # generation
                        # Metrics.save_img(hr_img, '{}/{}_{}_hr.png'.format(result_path, current_step, idx))
                        # Metrics.save_img(sr_img, '{}/{}_{}_sr.png'.format(result_path, current_step, idx))
                        # Metrics.save_img(lr_img, '{}/{}_{}_lr.png'.format(result_path, current_step, idx))
                        # Metrics.save_img(fake_img, '{}/{}_{}_inf.png'.format(result_path, current_step, idx))
                        # tb_logger.add_image(
                        #     'Iter_{}'.format(current_step),
                        #     np.transpose(np.concatenate(
                        #         (fake_img, sr_img, hr_img), axis=1), [2, 0, 1]),
                        #     idx)
                        # avg_psnr += Metrics.calculate_psnr(sr_img, hr_img)

                        # if wandb_logger:
                        #     wandb_logger.log_image(
                        #         f'validation_{idx}', 
                        #         np.concatenate((fake_img, sr_img, hr_img), axis=1)
                        #     )

                    # avg_psnr = avg_psnr / idx
                    avg_mse_input = avg_mse_input / 4 / (opt["model"]["diffusion"]["channels"] * opt["model"]["diffusion"]["image_size"]**2) / len(test_loader.dataset) * (opt["datasets"]["meta"]["max_depth"]**2)
                    avg_mse_predicted = avg_mse_predicted / 4 / (opt["model"]["diffusion"]["channels"] * opt["model"]["diffusion"]["image_size"]**2) / len(test_loader.dataset) * (opt["datasets"]["meta"]["max_depth"]**2)
                    diffusion.set_new_noise_schedule(opt['model']['beta_schedule']['train'], schedule_phase='train')
                    # log
                    # logger.info('# Validation # PSNR: {:.4e}'.format(avg_psnr))
                    logger.info(f"# Validation # MSE (Input to Target): {avg_mse_input:.4f}")
                    logger.info(f"# Validation # MSE (Prediction to Target): {avg_mse_predicted:.4f}")
                    logger_test = logging.getLogger('test')  # validation logger
                    # logger_test.info('<epoch:{:3d}, iter:{:8,d}> psnr: {:.4e}'.format(
                    #     current_epoch, current_step, avg_psnr))
                    # tensorboard logger
                    tb_logger.add_scalar('psnr', avg_psnr, current_step)

                    if wandb_logger:
                        wandb_logger.log_metrics({
                            'validation/val_psnr': avg_psnr,
                            'validation/val_step': val_step
                        })
                        val_step += 1

                if current_step % opt['train']['save_checkpoint_freq'] == 0:
                    logger.info('Saving models and training states.')
                    diffusion.save_network(current_epoch, current_step)

                    if wandb_logger and opt['log_wandb_ckpt']:
                        wandb_logger.log_checkpoint(current_epoch, current_step)

            if wandb_logger:
                wandb_logger.log_metrics({'epoch': current_epoch-1})

        # save model
        logger.info('End of training.')
    else:
        logger.info('Begin Model Evaluation.')
        avg_psnr, avg_ssim = 0.0, 0.0
        total_mse_input, total_mse_predicted = 0.0, 0.0
        mse_loss_sum = torch.nn.MSELoss(reduction="sum")
        result_path = opt['path']['results']
        
        logger.info(f"Num test batches: {len(test_loader)}")
        for val_data in tqdm(test_loader, desc=f"Test batch"):
            diffusion.feed_data(val_data)
            diffusion.test(continous=False)
            visuals = diffusion.get_current_visuals()
            sr_image = visuals['SR'].squeeze().clamp(
                opt["datasets"]["meta"]["norm_min"], 
                opt["datasets"]["meta"]["norm_max"]
            )

            loss_input = mse_loss_sum(visuals['INF'].squeeze(), visuals['HR'].squeeze())
            loss_predicted = mse_loss_sum(sr_image, visuals['HR'].squeeze())

            total_mse_input += loss_input.item()
            total_mse_predicted += loss_predicted.item()

            # hr_img = Metrics.tensor2img(visuals['HR'])  # uint8
            # lr_img = Metrics.tensor2img(visuals['LR'])  # uint8
            # fake_img = Metrics.tensor2img(visuals['INF'])  # uint8

            # sr_img_mode = 'grid'
            # if sr_img_mode == 'single':
            #     # single img series
            #     sr_img = visuals['SR']  # uint8
            #     sample_num = sr_img.shape[0]
            #     for iter in range(0, sample_num):
            #         Metrics.save_img(Metrics.tensor2img(sr_img[iter]), '{}/{}_{}_sr_{}.png'.format(result_path, current_step, idx, iter))
            # else:
            #     # grid img
            #     sr_img = Metrics.tensor2img(visuals['SR'])  # uint8
            #     Metrics.save_img(sr_img, '{}/{}_{}_sr_process.png'.format(result_path, current_step, idx))
            #     Metrics.save_img(Metrics.tensor2img(visuals['SR'][-1]), '{}/{}_{}_sr.png'.format(result_path, current_step, idx))

            if opt["test"]["save_results"]:
                # logger.info(f"sr_image shape: {sr_image.shape}")
                sr_flood_map = Metrics.tensor2floodmap(
                    sr_image, 
                    opt["datasets"]["meta"]["max_depth"], 
                    opt["test"]["threshold"], 
                    min_max=(opt["datasets"]["meta"]["norm_min"], opt["datasets"]["meta"]["norm_max"])
                )
                # logger.info(f"sr_flood_map shape: {sr_flood_map.shape}")

                filenames = visuals['filenames']
                profiles = visuals['profiles']
                # logger.info(f"filenames: {filenames}, {type(filenames)}")
                # logger.info(f"profiles: {profiles}, {type(profiles)}")
                for i in range(opt["datasets"]["test"]["batch_size"]):
                    filename = filenames[i]
                    # logger.info(f"image: {filename}")
                    Metrics.save_flood_map(sr_flood_map[i], os.path.join(result_path, filename), profiles[i])

            # Metrics.save_img(hr_img, '{}/{}_{}_hr.png'.format(result_path, current_step, idx))
            # Metrics.save_img(lr_img, '{}/{}_{}_lr.png'.format(result_path, current_step, idx))
            # Metrics.save_img(fake_img, '{}/{}_{}_inf.png'.format(result_path, current_step, idx))

            # generation
            # eval_psnr = Metrics.calculate_psnr(Metrics.tensor2img(visuals['SR'][-1]), hr_img)
            # eval_ssim = Metrics.calculate_ssim(Metrics.tensor2img(visuals['SR'][-1]), hr_img)

            # avg_psnr += eval_psnr
            # avg_ssim += eval_ssim

            # if wandb_logger and opt['log_eval']:
            #     wandb_logger.log_eval_data(fake_img, Metrics.tensor2img(visuals['SR'][-1]), hr_img, eval_psnr, eval_ssim)

        # avg_psnr = avg_psnr / idx
        # avg_ssim = avg_ssim / idx
        total_mse_input = total_mse_input / 4 / (opt["model"]["diffusion"]["channels"] * opt["model"]["diffusion"]["image_size"]**2) / len(test_loader.dataset) * (opt["datasets"]["meta"]["max_depth"]**2)
        total_mse_predicted = total_mse_predicted / 4 / (opt["model"]["diffusion"]["channels"] * opt["model"]["diffusion"]["image_size"]**2) / len(test_loader.dataset) * (opt["datasets"]["meta"]["max_depth"]**2)

        # log
        # logger.info('# Validation # PSNR: {:.4e}'.format(avg_psnr))
        # logger.info('# Validation # SSIM: {:.4e}'.format(avg_ssim))
        logger.info(f"# Validation # MSE (LR to HR): {total_mse_input:.4f}")
        logger.info(f"# Validation # MSE (SR to HR): {total_mse_predicted:.4f}")
        logger_test = logging.getLogger('test')  # validation logger
        # logger_test.info(f'<epoch:{current_epoch:3d}, iter:{current_step:8,d}> psnr: {avg_psnr:.4e}, ssim：{avg_ssim:.4e}')

        if wandb_logger:
            if opt['log_eval']:
                wandb_logger.log_eval_table()
            wandb_logger.log_metrics({
                'PSNR': float(avg_psnr),
                'SSIM': float(avg_ssim)
            })
