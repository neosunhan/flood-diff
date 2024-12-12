import logging
from model.model import DDPM


def create_model(opt):
    m = DDPM(opt)
    logger = logging.getLogger('base')
    logger.info('Model [{:s}] is created.'.format(m.__class__.__name__))
    return m
