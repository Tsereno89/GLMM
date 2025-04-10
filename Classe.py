# Databricks notebook source
from pyiris.ingestion.config.file_system_config import FileSystemConfig
from pyiris.ingestion.extract import ExtractService, FileReader
from pyiris.ingestion.load import LoadService, FileWriter
from pyiris.infrastructure import Spark

import pyspark.sql.functions as f 
from pyspark.sql import  Row, Window

import pyspark as ps
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns


from sklearn.model_selection import RandomizedSearchCV,GridSearchCV,cross_validate
from scipy.stats import uniform, randint, loguniform #,quniform
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error,make_scorer,mean_absolute_error, mean_squared_error,r2_score

from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
# from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Lasso

from sklearn.svm import SVR
from sklearn.ensemble import AdaBoostRegressor

from sklearn.preprocessing import StandardScaler, MinMaxScaler,LabelEncoder
from sklearn.base import clone

import hyperopt
from hyperopt import fmin, tpe, hp, SparkTrials, STATUS_OK, Trials
 
import mlflow
from mlflow import MlflowClient

# import pymer4
# from pymer4.models import Lmer

from datetime import datetime
import os, sys
import json

import logging
logging.getLogger("urllib3").setLevel(logging.ERROR)
pyiris_spark = Spark()

# COMMAND ----------

def read_base_lake(file: str) -> ps.sql.dataframe.DataFrame:
    """
    Lê uma base de dados do data lake e retorna um DataFrame.
    ---
    file (str): Nome do arquivo a ser lido.
    ---
    return:
    base (ps.sql.dataframe.DataFrame): DataFrame com os dados lidos do data lake.
    """
    path_to_file = (r"Sales/InteligenciaDeMercado/" + file)
    base = (
        FileReader(table_id=file.lower(),
                   data_lake_zone='consumezone',
                   country='Brazil',
                   path=path_to_file,
                   format='parquet')
        .consume(spark=pyiris_spark)
    )
    return base
 
def write_lake(base: ps.sql.dataframe.DataFrame, file: str) -> None:
    """
    Escreve um DataFrame no data lake.
    ---
    base (ps.sql.dataframe.DataFrame): DataFrame a ser salvo no data lake.
    file (str): Nome do arquivo para salvar.
    ---
    return: None
    """
    path_to_file = (r"Sales/InteligenciaDeMercado/" + file)
    file_config = FileSystemConfig(
        mount_name='consumezone',
        country='Brazil',
        path=path_to_file,
        mode='overwrite',
        format='parquet'
    )
    writer = [FileWriter(config=file_config)]
    load_service = LoadService(writers=writer)
    load_service.commit(dataframe=base)


# COMMAND ----------

import re
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.optim as optim
from scipy.special import gammaln  # for Poisson and Gamma likelihoods
import math
import matplotlib.pyplot as plt

# -----------------------
# Helper: Inverse Link Functions
# -----------------------
def inverse_link(eta, link):
    if link == 'identity':
        return eta
    elif link == 'log':
        return torch.exp(eta)
    elif link == 'logit':
        return torch.sigmoid(eta)
    else:
        raise ValueError("Link not recognized.")

# -----------------------
# Helper: Log-likelihood functions for different distributions
# -----------------------
def loglik_gaussian(y, mu, sigma_y):
    ll = -0.5 * torch.log(2 * math.pi * sigma_y**2) - 0.5 * ((y - mu)**2) / (sigma_y**2)
    return ll

def loglik_binomial(y, mu):
    eps = 1e-6
    mu = torch.clamp(mu, eps, 1 - eps)
    ll = y * torch.log(mu) + (1 - y) * torch.log(1 - mu)
    return ll

def loglik_poisson(y, mu):
    ll = -mu + y * torch.log(mu) - torch.lgamma(y + 1)
    return ll

def loglik_gamma(y, mu, phi):
    # Parameterization: E(y)=mu, shape=phi.
    eps = 1e-6
    mu = torch.clamp(mu, eps, None)
    ll = (phi * torch.log(phi/mu) - torch.lgamma(phi)) + (phi - 1)*torch.log(y) - (phi*y/mu)
    return ll

# -----------------------
# Helper: Parse R-style Formula
# -----------------------
def parse_formula(formula, data):
    """
    Parse a simple R-style formula.
    Expected form: "y ~ x1 + x2 + (1 + x2 | group)"
    
    Returns:
      X: fixed effects design matrix as numpy array (including an intercept)
      y: response vector (numpy array)
      groups: grouping variable (numpy array)
      random_effect_cols: list of indices (in X) corresponding to random slopes.
          (Index 0 is reserved for the intercept.)
      fixed_colnames: list of column names in X.
    """
    # Split formula at '~'
    lhs, rhs = formula.split("~")
    response = lhs.strip()
    rhs = rhs.strip()
    
    # Find random effect terms: look for patterns like "( ... | ... )"
    rand_term = None
    rand_pattern = r"\((.*?)\)"
    m = re.search(rand_pattern, rhs)
    if m:
        rand_term = m.group(1)  # e.g., "1 + x2 | group"
        # Remove the random term from rhs
        rhs = re.sub(rand_pattern, "", rhs).strip()
    
    # Fixed effects: split remaining rhs by '+' and remove any empty parts
    fixed_terms = [term.strip() for term in rhs.split("+") if term.strip() and term.strip() != "-1"]
    # Always include an intercept if not explicitly removed:
    fixed_colnames = ["Intercept"] + fixed_terms

    # Build fixed effects design matrix X.
    n = data.shape[0]
    # Create intercept column
    intercept = np.ones((n, 1))
    if fixed_terms:
        X_fixed = data[fixed_terms].values  # assume these columns exist in data
        X = np.hstack([intercept, X_fixed])
    else:
        X = intercept.copy()
    # Get response vector y.
    y = data[response].values
    groups = None
    random_effect_cols = []  # list of indices in fixed design matrix to use as random slopes.
    
    if rand_term is not None:
        # rand_term should be of the form "1 + x2 | group" or "x1 | group"
        # Split at '|'
        parts = rand_term.split("|")
        if len(parts) != 2:
            raise ValueError("Random effects term not in expected format: '(...|group)'")
        rand_predictors = parts[0].strip()  # e.g., "1 + x2"
        group_var = parts[1].strip()        # e.g., "group"
        # Split rand_predictors by '+'
        tokens = [tok.strip() for tok in rand_predictors.split("+") if tok.strip()]
        # For each token, map to an index in fixed_colnames.
        for tok in tokens:
            if tok == "1":
                # random intercept corresponds to intercept column index 0.
                if 0 not in random_effect_cols:
                    random_effect_cols.append(0)
            else:
                if tok in fixed_colnames:
                    idx = fixed_colnames.index(tok)
                    random_effect_cols.append(idx)
                else:
                    raise ValueError(f"Random effect term '{tok}' not found among fixed effects: {fixed_colnames}")
        # Get groups from the data:
        groups = data[group_var].values
    else:
        # If no random term, default is random intercept (index 0) and groups must be provided separately.
        raise ValueError("No random effects term found in the formula. Please include at least a random intercept term using (1|group).")
        
    return X, y, groups, random_effect_cols, fixed_colnames

# -----------------------
# Updated GLMM Class with R-formula interface and Regularization options
# -----------------------
class GLMM:
    def __init__(self, X=None, y=None, groups=None, distribution='gaussian', link='identity',
                 random_effect_cols=None, formula=None, data=None, regularization=None, reg_lambda=0.0):
        """
        Parameters:
         - Either provide X (n x p numpy array or DataFrame), y (n,), groups, and random_effect_cols (list of indices)
           OR supply a formula (R-style string) and a pandas DataFrame (data).
         - distribution: one of 'gaussian', 'binomial', 'gamma', 'poisson'
         - link: one of 'identity', 'log', 'logit'
         - random_effect_cols: list of indices in fixed design matrix (if not using formula).
         - formula: R-style formula (e.g., "y ~ x1 + x2 + (1 + x2 | group)")
         - data: pandas DataFrame (required if formula is provided)
         - regularization: None, 'L1', or 'L2'
         - reg_lambda: regularization coefficient (nonnegative float)
        """
        self.distribution = distribution.lower()
        self.link = link.lower()
        self.regularization = regularization
        self.reg_lambda = reg_lambda
        
        if formula is not None and data is not None:
            # Use our formula parser.
            self.formula = formula # ------------------ ADD
            X, y, groups, random_effect_cols, fixed_colnames = parse_formula(formula, data)
            self.fixed_colnames = fixed_colnames
        else:
            if X is None or y is None or groups is None:
                raise ValueError("Either supply (X, y, groups, random_effect_cols) or (formula and data)")
            # If X is provided as a DataFrame, get its column names.
            if isinstance(X, pd.DataFrame):
                self.fixed_colnames = list(X.columns)
                X = X.values
            else:
                # If X is a numpy array, we assume columns are numbered.
                self.fixed_colnames = [f"X{i}" for i in range(X.shape[1])]
            random_effect_cols = random_effect_cols if random_effect_cols is not None else [0]
        
        # Store fixed effects design matrix and response.
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1)
        self.n, self.p = self.X.shape
        
        # Process groups.
        groups = np.array(groups)
        unique_groups, group_idx = np.unique(groups, return_inverse=True)
        self.group_idx = torch.tensor(group_idx, dtype=torch.long)
        self.n_groups = len(unique_groups)
        self.unique_groups = unique_groups
        
        # Build random-effect design matrix Z.
        # Z has d columns: always 1 for random intercept plus additional columns for each random slope.
        self.random_effect_cols = random_effect_cols  # list of indices in fixed effect design matrix.
        d = len(random_effect_cols)
        self.d = d
        n = self.n
        # For each observation, Z_i: each column j is the fixed design matrix column at index random_effect_cols[j]
        Z = np.zeros((n, d))
        for j, col_idx in enumerate(random_effect_cols):
            # For random intercept, the column should be ones.
            if col_idx == 0:
                Z[:, j] = 1.0
            else:
                Z[:, j] = self.X[:, col_idx].detach().numpy().ravel()
        self.Z = torch.tensor(Z, dtype=torch.float32)
        
        # Initialize fixed effects beta (p x 1)
        self.beta = nn.Parameter(torch.zeros(self.p, 1, dtype=torch.float32))
        # Initialize random effects b for each group (n_groups x d)
        self.b = nn.Parameter(torch.zeros(self.n_groups, d, dtype=torch.float32))
        # Initialize random effects std dev per component via log_sigma_b (d-dimensional vector)
        self.log_sigma_b = nn.Parameter(torch.zeros(d, dtype=torch.float32))
        
        # For gaussian, also estimate residual sigma_y; for gamma, estimate shape phi.
        if self.distribution == 'gaussian':
            self.log_sigma_y = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        elif self.distribution == 'gamma':
            self.log_phi = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
            
        # Collect parameters to optimize.
        self.params = [self.beta, self.b, self.log_sigma_b]
        if self.distribution == 'gaussian':
            self.params.append(self.log_sigma_y)
        elif self.distribution == 'gamma':
            self.params.append(self.log_phi)
            
        self.loss_history = []

    def model_loglik(self):
        """
        Compute the negative joint log-likelihood:
         - observation log-likelihood based on distribution and link.
         - penalty for random effects assuming b_g ~ N(0, diag(sigma_b^2))
         - plus optional regularization penalty on beta and b.
        """
        b_obs = self.b[self.group_idx]  # shape (n, d)
        # Compute contribution from random effects via design matrix Z:
        rand_part = torch.sum(self.Z * b_obs, dim=1, keepdim=True)
        eta = self.X @ self.beta + rand_part
        mu = inverse_link(eta, self.link)
        
        if self.distribution == 'gaussian':
            sigma_y = torch.exp(self.log_sigma_y)
            ll_obs = loglik_gaussian(self.y, mu, sigma_y)
        elif self.distribution == 'binomial':
            ll_obs = loglik_binomial(self.y, mu)
        elif self.distribution == 'poisson':
            ll_obs = loglik_poisson(self.y, mu)
        elif self.distribution == 'gamma':
            phi = torch.exp(self.log_phi)
            ll_obs = loglik_gamma(self.y, mu, phi)
        else:
            raise ValueError("Distribution not recognized.")
        ll_obs_sum = torch.sum(ll_obs)
        
        # Random effects penalty: each b_{g,j} ~ N(0, sigma_b[j]^2)
        sigma_b = torch.exp(self.log_sigma_b)  # shape (d,)
        ll_rand = -0.5 * torch.sum((self.b**2) / (sigma_b**2)) \
                  - self.n_groups * torch.sum(torch.log(sigma_b)) \
                  - 0.5 * self.n_groups * self.d * math.log(2 * math.pi)
                  
        total_ll = ll_obs_sum + ll_rand
        
        # Regularization penalty (if any) on fixed effects and random effects:
        reg_penalty = 0.0
        if self.regularization is not None and self.reg_lambda > 0:
            if self.regularization.lower() == 'l1':
                reg_penalty = self.reg_lambda * (torch.sum(torch.abs(self.beta)) + torch.sum(torch.abs(self.b)))
            elif self.regularization.lower() == 'l2':
                reg_penalty = self.reg_lambda * (torch.sum(self.beta**2) + torch.sum(self.b**2))
            else:
                raise ValueError("regularization must be either 'L1', 'L2', or None")
        
        return -total_ll + reg_penalty

    def fit(self, lr=0.01, epochs=5000, verbose=True):
        optimizer = optim.Adam(self.params, lr=lr)
        for epoch in range(epochs):
            optimizer.zero_grad()
            loss = self.model_loglik()
            loss.backward()
            optimizer.step()
            self.loss_history.append(loss.item())
            if verbose and ((epoch % 500 == 0) or (epoch == epochs-1)):
                print(f"Epoch {epoch}: Negative Log-Likelihood = {loss.item():.4f}")
                
    def predict(self, new_data=None, X_new = None, groups_new = None):# ------------------ CHANGE (ADD new_data)
        """
        Predict on new data.
        new_data is optional. If provided, it should be a DataFrame with the same columns as used in training.
        X_new: fixed-effects design matrix (numpy array or DataFrame) with same columns as used in training.
        groups_new: group labels for each observation.
        For new groups (not seen in training), random effects are set to 0.
        """
        if self.formula is not None and new_data is not None:# ------------------ ADD
            X_new, y, groups_new, random_effect_cols, fixed_colnames = parse_formula(self.formula, new_data)# ------------------ ADD
        elif isinstance(X_new, pd.DataFrame):# ------------------ CHANGE (if to elif)
            X_new = X_new[self.fixed_colnames[1:]]  # skip intercept column name
            # Prepend a column of ones for intercept.
            X_new = np.hstack([np.ones((X_new.shape[0], 1)), X_new.values])
        else:
            X_new = np.array(X_new)
        X_new = torch.tensor(X_new, dtype=torch.float32)
        n_new = X_new.shape[0]
        # Build new Z matrix using the same random_effect_cols.
        d = self.d
        Z_new = np.zeros((n_new, d))
        for j, col_idx in enumerate(self.random_effect_cols):
            if col_idx == 0:
                Z_new[:, j] = 1.0
            else:
                Z_new[:, j] = X_new[:, col_idx].detach().numpy().ravel()
        Z_new = torch.tensor(Z_new, dtype=torch.float32)
        # Map groups_new to indices.
        b_new_list = []
        for g in groups_new:
            if g in self.unique_groups:
                idx = np.where(self.unique_groups == g)[0][0]
                b_new_list.append(self.b[idx])
            else:
                b_new_list.append(torch.zeros(self.d, dtype=torch.float32))
        b_new = torch.stack(b_new_list)  # shape (n_new, d)
        eta_new = X_new @ self.beta.detach() + torch.sum(Z_new * b_new, dim=1, keepdim=True)
        mu_new = inverse_link(eta_new, self.link)
        return mu_new.detach().numpy()

    def summary(self):
        print("Fixed Effects (beta):")
        print(self.beta.detach().numpy().ravel())
        print("Random Effects (first 5 groups):")
        print(self.b.detach().numpy()[:5])
        sigma_b = torch.exp(self.log_sigma_b).detach().numpy()
        print("Random Effects Std Dev (sigma_b) per component:")
        print(sigma_b)
        if self.distribution == 'gaussian':
            sigma_y = torch.exp(self.log_sigma_y).item()
            print(f"Residual Std Dev (sigma_y): {sigma_y:.4f}")
        elif self.distribution == 'gamma':
            phi = torch.exp(self.log_phi).item()
            print(f"Gamma shape (phi): {phi:.4f}")

    def plot_loss(self):
        plt.plot(self.loss_history)
        plt.xlabel("Epoch")
        plt.ylabel("Negative Log-Likelihood")
        plt.title("Loss History")
        plt.show()

    def plot_residuals(self):
        with torch.no_grad():
            b_obs = self.b[self.group_idx]
            rand_part = torch.sum(self.Z * b_obs, dim=1, keepdim=True)
            eta = self.X @ self.beta + rand_part
            mu = inverse_link(eta, self.link)
            residuals = self.y - mu
        fitted = mu.detach().numpy().ravel()
        resid = residuals.detach().numpy().ravel()
        plt.scatter(fitted, resid, alpha=0.6)
        plt.xlabel("Fitted values")
        plt.ylabel("Residuals")
        plt.title("Residuals vs Fitted")
        plt.axhline(0, color='red', linestyle='--')
        plt.show()

# ---------------------------
# Example: Simulate Data with Random Intercept and Slope (Gaussian GLMM)
# ---------------------------
def simulate_gaussian_glmm_random_slopes(n_groups=20, group_size=30, beta_true=[1.0, -0.5],
                                         sigma_b_intercept=1.0, sigma_b_slope=0.5,
                                         sigma_y_true=0.5, seed=42):
    np.random.seed(seed)
    n = n_groups * group_size
    # Fixed effects: intercept and one predictor "x"
    X = np.ones((n, 2))
    X[:,1] = np.random.normal(0, 1, size=n)
    groups = np.repeat(np.arange(n_groups), group_size)
    b_intercepts = np.random.normal(0, sigma_b_intercept, size=n_groups)
    b_slopes = np.random.normal(0, sigma_b_slope, size=n_groups)
    eta = beta_true[0] + beta_true[1]*X[:,1] + b_intercepts[groups] + b_slopes[groups]*X[:,1]
    y = eta + np.random.normal(0, sigma_y_true, size=(n, 1))
    return X, y.ravel(), groups