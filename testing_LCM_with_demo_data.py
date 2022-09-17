import pandas as pd
import numpy as np

from dcm_ard_libs import minimize, neglog_DCM

theta = pd.read_csv('~/DCM-ARD/theta.csv', header=None).values
X = pd.read_csv('~/DCM-ARD/X.csv', header=None).values
Y = pd.read_csv('~/DCM-ARD/Y.csv', header=None).values
Y = (Y == 1).astype(int)[:,[0]]
Y_onehot = pd.read_csv('~/DCM-ARD/Y_onehot.csv', header=None).values
Y_onehot = Y_onehot[:, [0]]
availableChoices = pd.read_csv('~/DCM-ARD/availableChoices.csv', header=None).values
availableChoices = availableChoices[:, [0]]

theta_optim_full = minimize( theta, neglog_DCM, -20000, X, Y, Y_onehot, availableChoices)

print('done')