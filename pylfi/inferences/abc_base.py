#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
from abc import ABCMeta, abstractmethod
from typing import Callable, Union

import colorlog
from pylfi.distances import euclidean
from pylfi.utils.checks import check_distance_str


def setup_logger(name):
    """Return a logger with a default ColoredFormatter."""
    formatter = colorlog.ColoredFormatter(
        "%(log_color)s%(levelname)-8s%(reset)s %(blue)s%(message)s",
        datefmt=None,
        reset=True,
        log_colors={
            "DEBUG": "cyan",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "red",
        },
    )

    logger = logging.getLogger(name)
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    return logger


# class ABCBase(metaclass=ABCMeta):
class ABCBase:
    def __init__(self, observation, simulator, statistics_calculator, priors, distance_metric, seed):
        """
        simulator : callable
            simulator model
        summary_calculator : callable, defualt None
            summary statistics calculator. If None, simulator should output
            sum stat
        distance : str
            Can be a custom function or one of l1, l2, mse
        distance_metric : callable
            discrepancy measure
        """
        self._obs_data = observation
        self._stat_calc = statistics_calculator
        self._simulator = simulator
        self._priors = priors
        self._seed = seed

        self._obs_sumstat = self._stat_calc(self._obs_data)

        # Select distance function.
        if callable(distance_metric):
            self._distance_metric = distance_metric
        elif isinstance(distance_metric, str):
            check_distance_str(distance_metric)
            self._distance_metric = self._choose_distance(distance_metric)
        else:
            raise TypeError()

        #self.logger = setup_logger(self.__class__.__name__)
        #self.logger = colorlog.getLogger(self.__class__.__name__)

    @abstractmethod
    def sample(self):
        """To be overwritten by sub-class: should implement sampling from
        inference scheme and return journal.

        Returns
        -------
        pylfi.journal
            Journal
        """

        raise NotImplementedError

    @staticmethod
    def _choose_distance(distance):
        """Return distance function for given distance type."""
        if distance == 'l1':
            return None
        elif distance == 'l2':
            return euclidean
        elif distance == 'mse':
            return None

    @staticmethod
    def run_lra():
        """Linear regression adjustment as in Beaumont et al. 2002.
        """
        pass
