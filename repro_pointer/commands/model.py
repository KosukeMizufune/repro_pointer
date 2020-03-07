import logging

from repro_pointer import models


def get_net(n_class, name, params={}, logger=None):
    logger = logger or logging.getLogger(__name__)
    return getattr(models, name)(n_class=n_class, logger=logger, **params)


def get_loss(name, params={}, logger=None):
    logger = logger or logging.getLogger(__name__)
    return getattr(models, name)(logger=logger, **params)
