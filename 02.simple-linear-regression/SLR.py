
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# data preprocessing
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]

