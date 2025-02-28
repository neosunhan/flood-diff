import torch
import data as Data
import model as Model
import argparse
import logging
import os
import logging
import core.logger as Logger
import core.metrics as Metrics
from tqdm import tqdm


def train(diffusion, opt, train_loader, valid_loader=None):
    diffusion.set_new_noise_schedule(opt['model']['beta_schedule']['train'], schedule_phase='train')
    current_step = diffusion.begin_step
    current_epoch = diffusion.begin_epoch
    n_iter = opt['train']['n_iter']

    logger = logging.getLogger('base')
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
                message = f'<epoch:{current_epoch:3d}, iter:{current_step:8,d}> '
                for k, v in logs.items():
                    message += f'{k:s}: {v:.4e} '
                logger.info(message)

            # validation
            if valid_loader is not None and current_step % opt['train']['val_freq'] == 0:
                test(diffusion, opt, valid_loader)
                diffusion.set_new_noise_schedule(opt['model']['beta_schedule']['train'], schedule_phase='train')

            if current_step % opt['train']['save_checkpoint_freq'] == 0:
                logger.info('Saving models and training states.')
                diffusion.save_network(current_epoch, current_step)
    
    return diffusion


def test(diffusion, opt, test_loader):
    diffusion.set_new_noise_schedule(opt['model']['beta_schedule']['test'], schedule_phase='test')
    total_mse_input, total_mse_predicted = 0.0, 0.0
    mse_loss_sum = torch.nn.MSELoss(reduction="sum")

    result_path = opt['path']['results']
    n_channels = opt["datasets"]["meta"]["channels"]
    image_size = opt["datasets"]["meta"]["image_size"]
    max_depth = opt["datasets"]["meta"]["max_depth"]
    norm_range = (opt["datasets"]["meta"]["norm_min"], opt["datasets"]["meta"]["norm_max"])
    mse_coeff = (norm_range[1] - norm_range[0]) ** 2
    logger = logging.getLogger('base')
    
    for val_data in tqdm(test_loader, desc=f"Test batch"):
        diffusion.feed_data(val_data)
        diffusion.test()
        visuals = diffusion.get_current_visuals()
        sr_image = visuals['SR'].squeeze().clamp(
            opt["datasets"]["meta"]["norm_min"], 
            opt["datasets"]["meta"]["norm_max"]
        )

        total_mse_input += mse_loss_sum(visuals['INF'].squeeze(), visuals['HR'].squeeze()).item()
        total_mse_predicted += mse_loss_sum(sr_image, visuals['HR'].squeeze()).item()

        if opt["phase"] == "test" and opt["test"]["save_results"]:
            sr_flood_map = Metrics.tensor2floodmap(
                sr_image, 
                max_depth, 
                opt["test"]["threshold"], 
                min_max=norm_range
            )

            filenames, profiles = visuals['filenames'], visuals['profiles']
            for i in range(opt["datasets"]["test"]["batch_size"]):
                filename = filenames[i]
                Metrics.save_flood_map(sr_flood_map[i], os.path.join(result_path, filename), profiles[i])

    total_mse_input /= n_channels * (image_size ** 2)
    total_mse_input /= len(test_loader.dataset)
    total_mse_input *= max_depth ** 2
    total_mse_input /= mse_coeff

    total_mse_predicted /= n_channels * (image_size ** 2)
    total_mse_predicted /= len(test_loader.dataset)
    total_mse_predicted *= max_depth ** 2
    total_mse_predicted /= mse_coeff

    # log
    logger.info(f"# Validation # MSE (CG to FG): {total_mse_input:.4f}")
    logger.info(f"# Validation # MSE (SR to FG): {total_mse_predicted:.4f}")

    return total_mse_input, total_mse_predicted


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/sr_sr3_16_128.json',
                        help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['train', 'test'],
                        help='Run either train(training) or test(generation)', default='train')
    parser.add_argument('-debug', '-d', action='store_true')
    parser.add_argument('-output_dir', type=str)
    args = parser.parse_args()

    # parse configs
    opt = Logger.parse(args)
    # Convert to NoneDict, which return None for missing key.
    opt = Logger.dict_to_nonedict(opt)

    # logging
    logger = Logger.setup_logger('base', opt['path']['log'], 'train', level=logging.INFO, screen=True)
    logger.info(Logger.dict2str(opt))

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    # dataset
    if args.phase == 'train':
        train_set = Data.create_dataset(opt['datasets']['train'], 'train', opt["datasets"]["meta"], opt["latent"])
        train_loader = Data.create_dataloader(train_set, opt['datasets']['train'], 'train')
        if opt["latent"]:
            valid_loader = None
        else:
            valid_set = Data.create_dataset(opt['datasets']['valid'], 'test', opt["datasets"]["meta"], opt["latent"])
            valid_loader = Data.create_dataloader(valid_set, opt['datasets']['valid'], 'test')

    elif args.phase == 'test':
        test_set = Data.create_dataset(opt['datasets']['test'], 'test', opt["datasets"]["meta"], latent=False)
        test_loader = Data.create_dataloader(test_set, opt['datasets']['test'], 'test')
    logger.info('Initial Dataset Finished')

    # model
    diffusion = Model.create_model(opt)
    logger.info('Initial Model Finished')

    if opt['phase'] == 'train':
        logger.info('Begin training.')
        train(diffusion, opt, train_loader, valid_loader)
        logger.info('End of training.')
    else:
        logger.info('Begin testing.')
        num_epochs = opt['datasets']['test']['epochs']
        input_mse, predicted_mse = 0, 0
        for i in range(1, num_epochs + 1):
            logger.info(f"Test epoch {i}/{num_epochs}:")
            epoch_input_mse, epoch_predicted_mse = test(diffusion, opt, test_loader)
            input_mse += epoch_input_mse
            predicted_mse += epoch_predicted_mse
        input_mse /= num_epochs
        predicted_mse /= num_epochs
        logger.info(f"# Average MSE (CG to FG): {input_mse:.4f}")
        logger.info(f"# Average MSE (SR to FG): {predicted_mse:.4f}")
        logger.info('End of testing.')
