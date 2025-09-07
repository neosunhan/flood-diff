import torch
import data as Data
import model as Model
import argparse
import logging
import numpy as np
import os
import time
import datetime
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


def test(diffusion, opt, test_loader, progress_bar=True):
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
    
    if progress_bar:
        test_iterator = tqdm(test_loader, desc=f"Test batch")
    else:
        test_iterator = test_loader
    start_time = time.time()
    # imgs = []
    for val_data in test_iterator:
        diffusion.feed_data(val_data)
        diffusion.test()
        visuals = diffusion.get_current_visuals()
        sr_image = visuals['SR'].squeeze().clamp(norm_range[0], norm_range[1])
        fg_image = visuals['HR'].squeeze()
        cg_image = visuals['INF'].squeeze()
        
        # imgs.append(sr_image.cpu().detach().numpy())

        total_mse_input += mse_loss_sum(cg_image, fg_image).item()
        total_mse_predicted += mse_loss_sum(sr_image, fg_image).item()

        if opt["test"]["thresholds"]:
            fg_flood_map = Metrics.unnormalize(fg_image, max_depth, min_max=norm_range)
            cg_flood_map = Metrics.unnormalize(cg_image, max_depth, min_max=norm_range)
            sr_flood_map = Metrics.unnormalize(sr_image, max_depth, min_max=norm_range)

            classification_results = {
                threshold: {
                    map_type: {
                        "tp": 0,
                        "fp": 0,
                        "fn": 0,
                        "tn": 0
                    } for map_type in ("CG", "SR")
                } for threshold in opt["test"]["thresholds"]
            }

            for threshold in opt["test"]["thresholds"]:
                fg_bin = torch.where(fg_flood_map < threshold, False, True)
                cg_bin = torch.where(cg_flood_map < threshold, False, True)
                sr_bin = torch.where(sr_flood_map < threshold, False, True)

                classification_results[threshold]["CG"]["tp"] += torch.sum(torch.logical_and(cg_bin, fg_bin))
                classification_results[threshold]["CG"]["fp"] += torch.sum(torch.logical_and(cg_bin, torch.logical_not(fg_bin)))
                classification_results[threshold]["CG"]["fn"] += torch.sum(torch.logical_and(torch.logical_not(cg_bin), fg_bin))
                classification_results[threshold]["CG"]["tn"] += torch.sum(torch.logical_and(torch.logical_not(cg_bin), torch.logical_not(fg_bin)))

                classification_results[threshold]["SR"]["tp"] += torch.sum(torch.logical_and(sr_bin, fg_bin))
                classification_results[threshold]["SR"]["fp"] += torch.sum(torch.logical_and(sr_bin, torch.logical_not(fg_bin)))
                classification_results[threshold]["SR"]["fn"] += torch.sum(torch.logical_and(torch.logical_not(sr_bin), fg_bin))
                classification_results[threshold]["SR"]["tn"] += torch.sum(torch.logical_and(torch.logical_not(sr_bin), torch.logical_not(fg_bin)))
        
        if opt["phase"] == "test" and opt["test"]["save_results"]:
            sr_flood_map = Metrics.tensor2floodmap(
                sr_image, 
                max_depth, 
                opt["test"]["thresholds"][0], 
                min_max=norm_range
            )

            filenames, profiles = visuals['filenames'], visuals['profiles']
            for i in range(opt["datasets"]["test"]["batch_size"]):
                filename = filenames[i]
                Metrics.save_flood_map(sr_flood_map[i], os.path.join(result_path, filename), profiles[i])

    end_time = time.time()
    time_taken = datetime.timedelta(seconds=end_time - start_time)

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
    if opt["test"]["thresholds"]:
        for threshold in classification_results:
            for map_type, results_dict in classification_results[threshold].items():
                results_dict["pod"] = results_dict["tp"] / (results_dict["tp"] + results_dict["fn"])
                results_dict["rfa"] = results_dict["fp"] / (results_dict["tp"] + results_dict["fp"])
                results_dict["csi"] = results_dict["tp"] / (results_dict["tp"] + results_dict["fn"] + results_dict["fp"])
                logger.info(f"# Threshold: {threshold}cm # POD ({map_type} to FG): {results_dict['pod']:.4f}")
                logger.info(f"# Threshold: {threshold}cm # RFA ({map_type} to FG): {results_dict['rfa']:.4f}")
                logger.info(f"# Threshold: {threshold}cm # CSI ({map_type} to FG): {results_dict['csi']:.4f}")
    logger.info(f"# Time taken: {str(time_taken)}")

    if opt["test"]["thresholds"]:
        return total_mse_input, total_mse_predicted, classification_results, time_taken
    else:
        return total_mse_input, total_mse_predicted, None, time_taken
    # return total_mse_input, total_mse_predicted, time_taken


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
        train_set = Data.create_dataset(opt['datasets']['train'], 'train', opt["datasets"]["meta"], opt["latent"], opt["dem"])
        train_loader = Data.create_dataloader(train_set, opt['datasets']['train'], 'train')
        if opt["latent"]:
            valid_loader = None
        else:
            valid_set = Data.create_dataset(opt['datasets']['valid'], 'test', opt["datasets"]["meta"], opt["latent"], opt["dem"])
            valid_loader = Data.create_dataloader(valid_set, opt['datasets']['valid'], 'test')

    elif args.phase == 'test':
        test_set = Data.create_dataset(opt['datasets']['test'], 'test', opt["datasets"]["meta"], latent=False, dem=opt["dem"])
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
        if num_epochs is None or num_epochs == 1:
            test(diffusion, opt, test_loader)
        else:
            input_mse, predicted_mse = [], []
            threshold_csi = {
                threshold: {
                    map_type: {
                        "pod": 0,
                        "rfa": 0,
                        "csi": 0
                    } for map_type in ("CG", "SR")
                } for threshold in opt["test"]["thresholds"]
            }
            total_duration = datetime.timedelta()
            all_maps = []
            for i in range(1, num_epochs + 1):
                logger.info(f"Test epoch {i}/{num_epochs}:")
                epoch_input_mse, epoch_predicted_mse, epoch_classification_results, duration = test(diffusion, opt, test_loader, progress_bar=False)
                input_mse.append(epoch_input_mse)
                predicted_mse.append(epoch_predicted_mse)
                total_duration += duration
                if opt["test"]["thresholds"]:
                    for threshold in threshold_csi:
                        for map_type in threshold_csi[threshold]:
                            threshold_csi[threshold][map_type]['pod'] += epoch_classification_results[threshold][map_type]['pod']
                            threshold_csi[threshold][map_type]['rfa'] += epoch_classification_results[threshold][map_type]['rfa']
                            threshold_csi[threshold][map_type]['csi'] += epoch_classification_results[threshold][map_type]['csi']
                # all_maps.append(epoch_maps)
            average_time = total_duration / num_epochs
            average_input_mse = sum(input_mse) / len(input_mse)
            average_predicted_mse = sum(predicted_mse) / len(predicted_mse)
            logger.info(f"# Average MSE (CG to FG): {average_input_mse:.4f}")
            logger.info(f"# Average MSE (SR to FG): {average_predicted_mse:.4f}")
            if opt["test"]["thresholds"]:
                for threshold in threshold_csi:
                    for map_type in threshold_csi[threshold]:
                        threshold_csi[threshold][map_type]['pod'] /= num_epochs
                        threshold_csi[threshold][map_type]['rfa'] /= num_epochs
                        threshold_csi[threshold][map_type]['csi'] /= num_epochs
                        logger.info(f"# Threshold {threshold}cm # Average POD ({map_type} to FG): {threshold_csi[threshold][map_type]['pod']:.4f}")
                        logger.info(f"# Threshold {threshold}cm # Average RFA ({map_type} to FG): {threshold_csi[threshold][map_type]['rfa']:.4f}")
                        logger.info(f"# Threshold {threshold}cm # Average CSI ({map_type} to FG): {threshold_csi[threshold][map_type]['csi']:.4f}")
            # variance = np.var(all_maps, axis=0)
            # standard_dev = np.std(all_maps, axis=0)
            # logger.info(f"Variance ({variance.shape}): {np.mean(variance):.2E}, Standard Deviation ({standard_dev.shape}): {np.mean(standard_dev):.2E}")
            logger.info(f"Average time taken: {str(average_time)}")
        logger.info('End of testing.')
