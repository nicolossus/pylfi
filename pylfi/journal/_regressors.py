#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod

import numpy as np


class RegressionBase(metaclass=ABCMeta):
    r"""Regressor base
    """

    def __init__(self, standardize=False, penalty=1.0):
        self._coef = None
        self._intercept = None
        self._standardize = standardize
        self._penalty = penalty

    def _process_data(self, X, y):
        r"""Process data.

        * Ensure correct shape of data `X`
        * Augment with a column of ones
        * Standardize data and target if kw `standardize=True`

        Parameters
        ----------
        X : array_like
            Data
        y : array_like
            Target

        Returns
        -------
        X : ndarray
            Processed data
        y : ndarray
            Processed target
        """
        X = np.asarray(X)
        y = np.asarray(y)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        # augment with column of ones
        X = np.c_[np.ones(X.shape[0]), X]

        if self._standardize:
            X_mean = np.mean(X[:, 1:], axis=0)
            X_sd = np.std(X_train[:, 1:], axis=0)
            X_stand = (X[:, 1:] - X_mean[np.newaxis, :]) / X_sd[np.newaxis, :]
            X = np.c_[np.ones(X.shape[0]), X_stand]

            y_mean = np.mean(y)
            y_sd = np.std(y)
            y_stand = (y - y_mean) / y_sd
            y = y_stand

        self._data = X
        self._target = y

    @abstractmethod
    def fit(self, X, y):
        r"""To be overwritten by sub-class;
        """

        raise NotImplementedError

    @abstractmethod
    def predict(self, X):
        r"""To be overwritten by sub-class;"""

        raise NotImplementedError

    @abstractmethod
    def posterior_mean(self):
        r"""To be overwritten by sub-class;
        """

        raise NotImplementedError

    @property
    def coef(self):
        return self._coef

    @property
    def intercept(self):
        return self._intercept

    @property
    def penalty(self):
        return self._penalty

    @penalty.setter
    def penalty(self, value):
        self._penalty = value


class OLS():

    def __init__(self, standardize=False):
        super.init(standardize=standardize)

    def fit(self, X, y):

        X, y = self._process_data(X, y)

        xTx = np.dot(X.T, X)
        inverse_xTx = np.linalg.pinv(xTx)
        xTy = np.dot(X.T, y)
        coef = np.dot(inverse_xTx, xTy)

        self._intercept = coef[0]
        self._coef = coef[1:]

    def predict(self, X):
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return self.intercept + np.dot(X, self.coef)

    def posterior_mean():
        return self.intercept


class LocLinear():
    pass


'''
class OLS(StatMetrics, MLModelTools):
    """
    Linear Model Using Ordinary Least Squares (OLS).
    Parameters
    ----------
    fit_intercept : boolean, optional, default True
        whether to calculate the intercept for this model. If set to False, no
        intercept will be used in calculations.
    Attributes
    ----------
    coef_ : array, shape (n_features,)
        Estimated coefficients for the linear regression problem
    intercept_ : float
        Independent term in the linear model
    """

    def __init__(self, fit_intercept=True):
        self.coef_ = None
        self.intercept_ = None
        self._fit_intercept = fit_intercept

    def fit(self, X, y):
        """
        Fit the model according to the given training data.
        Parameters
        ----------
        X : array, shape = (n_samples) or shape = (n_samples, n_features)
            Training samples
        y : array, shape = (n_samples)
            Target values
        Returns
        -------
        Estimated coefficients for the linear regression problem : array, shape (n_features,)
        """

        self.data = X
        self.target = y

        # check if X is 1D or 2D array
        if len(self.data.shape) == 1:
            X = self.data.reshape(-1, 1)
        else:
            X = self.data
        # add bias if fit_intercept
        if self._fit_intercept:
            X = np.c_[np.ones(X.shape[0]), X]

        xTx = np.dot(X.T, X)
        inverse_xTx = np.linalg.pinv(xTx)
        xTy = np.dot(X.T, self.target)
        coef = np.dot(inverse_xTx, xTy)

        # set attributes
        if self._fit_intercept:
            self.intercept_ = coef[0]
            self.coef_ = coef[1:]
        else:
            self.intercept_ = 0
            self.coef_ = coef

        return self.coef_

    def predict(self, X):
        """
        Predicts the value after the model has been trained.
        Parameters
        ----------
        X : array, shape (n_samples) or shape (n_samples, n_features)
            Test samples
        Returns
        -------
        Predicted values : array, shape (n_samples,)
        """

        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        return self.intercept_ + np.dot(X, self.coef_)


class Ridge(StatMetrics, MLModelTools):
    """
    Linear Model Using Ridge Regression.
    Parameters
    ----------
    lmbda : float, optional, default 1.0
        regularization (penalty) parameter; must be a positive float.
        Regularization improves the conditioning of the problem and reduces
        the variance of the estimates. Larger values specify stronger
        regularization.
    fit_intercept : boolean, optional, default True
        whether to calculate the intercept for this model. If set to False, no
        intercept will be used in calculations.
    Attributes
    ----------
    coef_ : array, shape (n_features,)
        Estimated coefficients for the linear regression problem
    intercept_ : float
        Independent term in the linear model
    """

    def __init__(self, lmbda=1.0, fit_intercept=True):
        self.coef_ = None
        self.intercept_ = None
        self._lmbda = lmbda
        self._fit_intercept = fit_intercept

    def fit(self, X, y):
        """
        Fit the model according to the given training data.
        Parameters
        ----------
        X : array, shape = (n_samples) or shape = (n_samples, n_features)
            Training samples
        y : array, shape = (n_samples)
            Target values
        Returns
        -------
        Estimated coefficients for the linear regression problem : array, shape (n_features,)
        """

        self.data = X
        self.target = y

        # check if X is 1D or 2D array
        if len(self.data.shape) == 1:
            X = self.data.reshape(-1, 1)
        else:
            X = self.data
        # add bias if fit_intercept
        if self._fit_intercept:
            X = np.c_[np.ones(X.shape[0]), X]

        xTx = np.dot(X.T, X)
        N = xTx.shape[0]
        inverse_xTx = np.linalg.pinv(xTx + self._lmbda * np.identity(N))
        xTy = np.dot(X.T, self.target)
        coef = np.dot(inverse_xTx, xTy)

        # set attributes
        if self._fit_intercept:
            self.intercept_ = coef[0]
            self.coef_ = coef[1:]
        else:
            self.intercept_ = 0
            self.coef_ = coef

        return self.coef_

    def predict(self, X):
        """
        Predicts the value after the model has been trained.
        Parameters
        ----------
        X : array, shape (n_samples) or shape (n_samples, n_features)
            Test samples
        Returns
        -------
        Predicted values : array, shape (n_samples,)
        """
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        return self.intercept_ + np.dot(X, self.coef_)

    def set_penalty(self, lmbda):
        """
        Set regularization parameter.
        Parameters
        ----------
        lmbda : float
            Value of regularization parameter
        Returns
        -------
        object : self
        """
        self._lmbda = lmbda
        return self
'''
