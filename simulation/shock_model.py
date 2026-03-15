from scipy.integrate import solve_ivp
import numpy as np
import pandas as pd
import ast
import matplotlib.pyplot as pl


def parse_hist(hist_str):
    """Converte stringa storico in array numpy"""
    if isinstance(hist_str, str):
        try:
            # Prova formato lista Python: "[100, 105, 110]"
            return np.array(ast.literal_eval(hist_str))
        except:
            # Altrimenti formato CSV: "100,105,110"
            return np.array([float(x.strip()) for x in hist_str.split(',')])
    
    # Se già è un array/lista, convertilo
    return np.array(hist_str)


def read_data(file):
    """Legge CSV e parsa tutti gli storici in array NumPy"""
    df = pd.read_csv(file)
    if 'profile_id' in df.columns:
        df = df.rename(columns={'profile_id': 'id'})
    
    df = df[['id', 'monthly_income_hist', 'monthly_fixed_exp_hist', 
             'monthly_variable_exp_hist', 'monthly_saving_hist', 
             'debt_hist', 'bnpl_exposure']]
    
    # PARSING: converti tutte le colonne storici da stringhe ad array
    df['monthly_income_hist'] = df['monthly_income_hist'].apply(parse_hist)
    df['monthly_fixed_exp_hist'] = df['monthly_fixed_exp_hist'].apply(parse_hist)
    df['monthly_variable_exp_hist'] = df['monthly_variable_exp_hist'].apply(parse_hist)
    df['monthly_saving_hist'] = df['monthly_saving_hist'].apply(parse_hist)
    df['debt_hist'] = df['debt_hist'].apply(parse_hist)
    
    return df


def derive_features(row):
    """Deriva feature per un singolo profilo"""
    # Estrai storici (già array numpy)
    income_hist = row['monthly_income_hist']
    fixed_exp_hist = row['monthly_fixed_exp_hist']
    variable_exp_hist = row['monthly_variable_exp_hist']
    saving_hist = row['monthly_saving_hist']
    debt_hist = row['debt_hist']
    
    # Calcola feature
    income_mean = np.mean(income_hist)
    income_volatility = np.std(income_hist) / income_mean  # Coefficient of variation
    
    fixed_exp_mean = np.mean(fixed_exp_hist)
    variable_exp_mean = np.mean(variable_exp_hist)
    expenses = fixed_exp_mean + variable_exp_mean
    
    fixed_cost_ratio = fixed_exp_mean / income_mean
    liquidity_buffer = np.mean(saving_hist)
    debt_0 = max(0.0, debt_hist[-1])  # Ultimo valore, non negativo
    
    # Income trend (pendenza regressione lineare)
    t = np.arange(len(income_hist))
    income_trend = np.polyfit(t, income_hist, 1)[0]
    
    return {
        'id': row['id'],
        'income_mean': income_mean,
        'income_volatility': income_volatility,
        'expenses': expenses,
        'fixed_cost_ratio': fixed_cost_ratio,
        'liquidity_buffer': liquidity_buffer,
        'debt_0': debt_0,
        'income_trend': income_trend,
        'bnpl_exposure': row['bnpl_exposure']
    }


def processing(df): 
   
    # Deriva feature per tutti i profili
    features_list = []
    for idx, row in df.iterrows():
        features = derive_features(row)
        features_list.append(features)
    
    df_processed = pd.DataFrame(features_list)
    
    print(f"\nFeature derivate per {len(df_processed)} profili:")
    print(df_processed.head())
    
    return df_processed
    


def ode_system(t, y, expenses, interest, shok_intensity, t_shock_start, t_shock_end, income_0):
    debt, income=y
    if t_shock_start <= t <= t_shock_end:
        shock = shok_intensity*income_0
    else:
        shock = 0.0

    d_debt = expenses + interest*debt - income
    d_income = -shock

    return [d_debt,d_income]

def simulate_profile(profile_id, features, shock_scenario):
     # Parametri fissi 
    INTEREST_RATE = 0.015
    T_MAX = 12
    N_STEPS = 200
    
    income_mean = features['income_mean']
    expenses = features['expenses']
    debt_0 = features['debt_0']

    # parametri schock 

    shock_intensity = shock_scenario['intensity']
    shock_start=shock_scenario['start']
    shock_end=shock_scenario['end']

    
    
    debt_trajectory = example([0,1000,1200,1100,1000,1200],400,60,[0,0,0,0,0,0],8,0.5)
    

    
    max_debt = np.max(debt_trajectory)
    fragility_index = max_debt / income_mean
    final_debt = debt_trajectory[-1]
    convergences = final_debt<max_debt*0.9
    num_points = 12
    return pd.DataFrame({
        'profile_id': [profile_id] * num_points,
        'scenario': [shock_scenario['name']] * num_points,
        't': range(12),
        'debt': debt_trajectory,
        'max_debt': [max_debt] * num_points,
        'final_debt': [final_debt] * num_points,
        'fragility_index': [fragility_index] * num_points,
        'convergence': [convergences] * num_points,
    })


def run_simulation(df_processed):
    """Simula tutti i profili con tutti gli scenari"""
    
    # 3 scenari di shock
    scenarios = [
        {'name': 'lieve', 'intensity': 0.20, 'start': 2, 'end': 4},
        {'name': 'medio', 'intensity': 0.40, 'start': 2, 'end': 6},
        {'name': 'grave', 'intensity': 0.60, 'start': 2, 'end': 8}
    ]
    
    print("\n=== Simulazione shock ===")
    print(f"Scenari: {len(scenarios)}")
    print(f"Profili: {len(df_processed)}")
    
    # Raccoglie tutti i risultati
    all_results = []
    
    # Loop su profili e scenari
    for idx, row in df_processed.iterrows():
        profile_id = int(row['id'])
        
        for scenario in scenarios:
            # Simula questo profilo + scenario
            result_df = simulate_profile(profile_id, row, scenario)
            all_results.append(result_df)
    
    # Concatena tutti i DataFrame
    df_final = pd.concat(all_results, ignore_index=True)
    
    # Salva CSV
    df_final.to_csv('simulation/simulation_output.csv', index=False)
    
    print(f"\n✓ Salvate {len(df_final)} righe in simulation/simulation_output.csv")
    print(f"  {len(df_processed)} profili × {len(scenarios)} scenari × 200 step")
    
    return df_final








def example(income_hist, exp_mean, std_exp, debt_hist, t_lost, etha):
    salario = income_hist[-1]
    initial_debt = debt_hist[-1]
    
    # Crea array di 12 mesi: salario fino a t_lost, poi 0
    pred_income = np.array([salario if i < t_lost else 0.0 for i in range(12)])
    pred_exp = np.random.normal(loc=exp_mean, scale=std_exp, size=12)
    
    debt = np.zeros(12)
    debt[0] = initial_debt

    for i in range(1, 12):
        
        difference = pred_exp[i] - pred_income[i]
        if difference > 0:
            debt[i] = debt[i-1] + difference  # ← aggiungi anche debt precedente
        else:
            debt[i] = debt[i-1] + etha * difference
            if debt[i] < 0: 
                debt[i]=0
                break
    
    return debt
    






def main():
    df = read_data('data/profiles.csv')
    print("=== Dati caricati ===")
    print(df.head())
    print(f"\nNumero profili: {len(df)}")
    
    print("\n=== Test parsing ===")
    print("Tipo prima colonna storico:", type(df['monthly_income_hist'].iloc[0]))
    print("Primo storico reddito:", df['monthly_income_hist'].iloc[0])
    
    print("\n=== Derivazione feature ===")
    df_processed = processing(df)
    
    print("\n=== Esecuzione simulazione ===")
    df_final = run_simulation(df_processed)
    
    print("\n✓ FATTO! CSV pronto per Chris (Layer 3)")


# main()


debt_trajectory = example([0,1000,1200,1100,1000,1200],400,60,[0,0,0,0,0,2000],4,0.5)
pl.figure()
pl.plot(range(12), debt_trajectory)
pl.show()