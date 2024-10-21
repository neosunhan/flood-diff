'''create dataset and dataloader'''
import logging
import torch.utils.data
import os


def create_dataloader(dataset, dataset_opt, phase):
    '''create dataloader '''
    if phase == 'train':
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=dataset_opt['batch_size'],
            shuffle=dataset_opt['use_shuffle'],
            pin_memory=True,
            drop_last=True
        )
    elif phase == 'test':
        return torch.utils.data.DataLoader(
            dataset, 
            batch_size=dataset_opt['batch_size'], 
            shuffle=dataset_opt['use_shuffle'], 
            num_workers=1, 
            pin_memory=True
        )
    else:
        raise NotImplementedError(
            'Dataloader [{:s}] is not found.'.format(phase))


def create_dataset(dataset_opt, phase, meta):
    '''create dataset'''
    # from data.LRHR_dataset import LRHRDataset as D
    from data.flood_depth_dataset import FloodDepthDatasetWithDEM as D
    # dataset = D(dataroot=dataset_opt['dataroot'],
    #             datatype=dataset_opt['datatype'],
    #             l_resolution=dataset_opt['l_resolution'],
    #             r_resolution=dataset_opt['r_resolution'],
    #             split=phase,
    #             data_len=dataset_opt['data_len'],
    #             need_LR=(dataset_opt['mode'] == 'LRHR')
    #             )
    dataset = D(
        low_res_folder=os.path.join(dataset_opt["dataroot"], f"{phase}_lr"),
        high_res_folder=os.path.join(dataset_opt["dataroot"], f"{phase}_hr"),
        dem_folder=os.path.join(dataset_opt["dataroot"], "cropped_dems"),
        max_value=meta["max_depth"],
        max_value_dem=meta["dem_max_value"],
        min_value_dem=meta["dem_min_value"],
        data_len=dataset_opt["data_len"],
        norm_range=(meta["norm_min"], meta["norm_max"])
    )
    logger = logging.getLogger('base')
    logger.info(f"Dataset [{dataset.__class__.__name__} - {dataset_opt['catchment']} ({phase})] is created.")
    return dataset
