#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np


class StatisticsCalculator:
    """
    Collection of summary statistic calculators
    """

    def __init__(self):
        pass

    @staticmethod
    def mean(data):
        return np.mean(data)

    @staticmethod
    def variance(data):
        return np.var(data)
