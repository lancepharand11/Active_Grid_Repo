# -*- coding: utf-8 -*-
"""
Created on Thu Jan  8 09:22:34 2026

@author: ctoppings
"""
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

def get_model_polynomial(order):
    model = Pipeline([
        ('poly', PolynomialFeatures(degree=order)),
        ('linear', LinearRegression(fit_intercept=True))
    ])
    return model