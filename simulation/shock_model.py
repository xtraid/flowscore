import scipy as sp
import numpy as np
import pandas as pd


def read_data(file):
    df = pd.read_csv(file)
    return df[['id', 'monthly_income_hist', 'monthly_fixed_exp_hist', 'monthly_variable_exp_hist', 'monthly_saving_hist', 'debt_hist', 'bnpl_exposure']]






def main ():
    df=read_data('data/profiles.csv')
    print(df.head())


main()