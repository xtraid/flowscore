from scipy.integrate import solve_ivp
import numpy as np
import pandas as pd


def read_data(file):
    df = pd.read_csv(file)
    return df[['id', 'monthly_income_hist', 'monthly_fixed_exp_hist', 'monthly_variable_exp_hist', 'monthly_saving_hist', 'debt_hist', 'bnpl_exposure']]

def shock(df): 
    #parametri fissi 
    INTEREST_RATE=0.015
    T_MAX=12
    N_STEPS=200

    #ODE
    income




def main ():
    df=read_data('data/profiles.csv')
    print(df.head())


main()