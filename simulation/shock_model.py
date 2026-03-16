import numpy as np
import pandas as pd
import ast
import matplotlib.pyplot as pl


# ── Parsing ──────────────────────────────────────────────────────────────────

def parse_hist(hist_str):
    """Converte stringa storico in array numpy"""
    if isinstance(hist_str, str):
        try:
            return np.array(ast.literal_eval(hist_str))
        except:
            return np.array([float(x.strip()) for x in hist_str.split(',')])
    return np.array(hist_str)


def read_data(file):
    """Legge CSV e parsa tutti gli storici in array NumPy"""
    df = pd.read_csv(file)
    if 'profile_id' in df.columns:
        df = df.rename(columns={'profile_id': 'id'})

    df = df[['id', 'monthly_income_hist', 'monthly_fixed_exp_hist',
             'monthly_variable_exp_hist', 'monthly_saving_hist',
             'debt_hist', 'bnpl_exposure', 'debito_richiesto']]

    hist_cols = ['monthly_income_hist', 'monthly_fixed_exp_hist',
                 'monthly_variable_exp_hist', 'monthly_saving_hist', 'debt_hist']
    for col in hist_cols:
        df[col] = df[col].apply(parse_hist)

    return df


# ── Feature engineering ───────────────────────────────────────────────────────

def derive_features(row):
    """Deriva feature per un singolo profilo"""
    income_hist   = row['monthly_income_hist']
    fixed_hist    = row['monthly_fixed_exp_hist']
    variable_hist = row['monthly_variable_exp_hist']
    saving_hist   = row['monthly_saving_hist']
    debt_hist     = row['debt_hist']

    income_mean      = np.mean(income_hist)
    income_volatility = np.std(income_hist) / income_mean  # Coefficient of variation
    fixed_exp_mean   = np.mean(fixed_hist)
    variable_exp_mean = np.mean(variable_hist)

    # Income trend (pendenza regressione lineare)
    income_trend = np.polyfit(np.arange(len(income_hist)), income_hist, 1)[0]

    return {
        'id':               row['id'],
        'income_mean':      income_mean,
        'income_volatility': income_volatility,
        'expenses':         fixed_exp_mean + variable_exp_mean,
        'fixed_cost_ratio': fixed_exp_mean / income_mean,
        'liquidity_buffer': np.mean(saving_hist),
        'debt_0':           max(0.0, debt_hist[-1]),  # Ultimo valore, non negativo
        'income_trend':     income_trend,
        'bnpl_exposure':    row['bnpl_exposure'],
        'debito_richiesto': row['debito_richiesto'],
    }


def processing(df):
    df_processed = pd.DataFrame(df.apply(derive_features, axis=1).tolist())
    print(f"\nFeature derivate per {len(df_processed)} profili:")
    print(df_processed.head())
    return df_processed


# ── Simulazione ───────────────────────────────────────────────────────────────

def example(income_hist, exp_mean, std_exp, debt_hist, t_lost, t_end, intensity, etha, debito_richiesto=0.0):
    """
    Simula la traiettoria del debito su 12 mesi.
    - t_lost:           mese in cui inizia lo shock
    - t_end:            mese in cui il reddito si riprende
    - intensity:        frazione di reddito persa durante lo shock (0.2 = −20%)
    - etha:             tasso di rimborso quando spesa < reddito (0=nessuno, 1=totale)
    - debito_richiesto: credito richiesto, sommato al debito iniziale (t=0)
    """
    salario      = income_hist[-1]
    initial_debt = debt_hist[-1] + debito_richiesto

    # Reddito: normale prima dello shock, ridotto durante, ripreso dopo
    pred_income = np.array([
        salario                   if i < t_lost else
        salario * (1 - intensity) if i < t_end  else
        salario
        for i in range(12)
    ])
    pred_exp    = np.random.normal(loc=exp_mean, scale=std_exp, size=12)

    debt    = np.zeros(12)
    debt[0] = initial_debt

    for i in range(1, 12):
        difference = pred_exp[i] - pred_income[i]
        if difference > 0:
            debt[i] = debt[i-1] + difference
        else:
            debt[i] = debt[i-1] + etha * difference
            if debt[i] < 0:
                debt[i] = 0
                break

    return debt


def simulate_profile(profile_id, features, shock_scenario):
    income_mean       = features['income_mean']
    expenses          = features['expenses']
    debt_0            = features['debt_0']
    debito_richiesto  = features['debito_richiesto']

    shock_intensity = shock_scenario['intensity']
    shock_start     = shock_scenario['start']
    shock_end       = shock_scenario['end']

    # etha: tasso di rimborso (0=nessun rimborso, 1=rimborso totale) — modifica qui
    etha    = 0.5
    std_exp = expenses * features['income_volatility']
    debt_trajectory = example(
        income_hist=[income_mean],
        exp_mean=expenses,
        std_exp=std_exp,
        debt_hist=[debt_0],
        t_lost=shock_start,
        t_end=shock_end,
        intensity=shock_intensity,
        etha=etha,
        debito_richiesto=debito_richiesto,
    )

    max_debt        = np.max(debt_trajectory)
    final_debt      = debt_trajectory[-1]
    fragility_index = max_debt / income_mean
    convergence     = final_debt < max_debt * 0.9

    # Percentuale del debito finale rispetto al debito richiesto
    perc_debito_finale = (final_debt / debito_richiesto) if debito_richiesto > 0 else 0.0

    n = 12
    return pd.DataFrame({
        'profile_id':         [profile_id] * n,
        'scenario':           [shock_scenario['name']] * n,
        't':                  range(n),
        'debt':               debt_trajectory,
        'max_debt':           [max_debt] * n,
        'final_debt':         [final_debt] * n,
        'debito_richiesto':   [debito_richiesto] * n,
        'perc_debito_finale': [round(perc_debito_finale, 4)] * n,
        'fragility_index':    [fragility_index] * n,
        'convergence':        [convergence] * n,
    })


def run_simulation(df_processed):
    """Simula tutti i profili con tutti gli scenari"""
    scenarios = [
        {'name': 'best_case', 'intensity': 0.00, 'start': 12, 'end': 12},  # nessuno shock
        {'name': 'lieve',     'intensity': 0.20, 'start': 2,  'end': 4 },
        {'name': 'medio',     'intensity': 0.40, 'start': 2,  'end': 6 },
        {'name': 'grave',     'intensity': 0.60, 'start': 2,  'end': 8 },
    ]

    print(f"\n=== Simulazione shock ===")
    print(f"Scenari: {len(scenarios)} | Profili: {len(df_processed)}")

    all_results = []
    for _, row in df_processed.iterrows():
        profile_id = int(row['id'])
        for scenario in scenarios:
            all_results.append(simulate_profile(profile_id, row, scenario))

    df_final = pd.concat(all_results, ignore_index=True)
    df_final.to_csv('simulation/simulation_output.csv', index=False)

    print(f"\n✓ Salvate {len(df_final)} righe in simulation/simulation_output.csv")
    return df_final


# ── Entry point ───────────────────────────────────────────────────────────────

def generate_paradox_report(sim_path='simulation/simulation_output.csv',
                            scores_path='model/scores_output.csv',
                            out_path='simulation/profili_paradosso.txt'):
    """
    Individua e riporta profili con:
      A) Tutti i profili con perc_debito_finale >= 500% nello scenario medio
      B) Credit score GLM basso (< 40) ma ottima gestione del debito sotto shock
         (perc_debito_finale < 500% nello scenario medio)
    """
    sim    = pd.read_csv(sim_path)
    scores = pd.read_csv(scores_path)

    # Usa solo scenario medio, solo l'ultimo mese (t=11) per avere i valori finali
    medio = sim[(sim['scenario'] == 'medio') & (sim['t'] == 11)].copy()
    medio = medio.merge(scores, left_on='profile_id', right_on='id')

    # Soglie fisse
    FLOWSCORE_BASSO = 40.0
    SOGLIA_PERC     = 5.0   # 500%: debito finale >= 5× il debito richiesto

    gruppo_A = medio[medio['perc_debito_finale'] >= SOGLIA_PERC].copy()
    gruppo_B = medio[(medio['flowscore'] < FLOWSCORE_BASSO) & (medio['perc_debito_finale'] < SOGLIA_PERC)]

    lines = []
    lines.append("=" * 65)
    lines.append("FLOWSCORE — PROFILI PARADOSSO: GLM vs GESTIONE SHOCK")
    lines.append("=" * 65)
    lines.append(f"Scenario analizzato: MEDIO  |  Soglie usate:")
    lines.append(f"  perc_debito_finale 'pessima' >= {SOGLIA_PERC:.0f}x  (>= 500% del debito richiesto)")
    lines.append(f"  perc_debito_finale 'ottima'  <  {SOGLIA_PERC:.0f}x  (<  500% del debito richiesto)")
    lines.append("")

    lines.append("─" * 65)
    lines.append("GRUPPO A — Tutti i profili con perc_debito_finale >= 500% (scenario medio)")
    lines.append(f"  ({len(gruppo_A)} profili)")
    lines.append("─" * 65)
    for _, r in gruppo_A.sort_values('perc_debito_finale', ascending=False).iterrows():
        lines.append(
            f"  Profilo {int(r['profile_id']):<4}  |  FlowScore: {r['flowscore']:.1f}"
            f"  |  debito_richiesto: €{r['debito_richiesto']:.0f}"
            f"  |  final_debt: €{r['final_debt']:.0f}"
            f"  |  perc_finale: {r['perc_debito_finale']*100:.1f}%"
            f"  |  fragility: {r['fragility_index']:.2f}"
        )

    lines.append("")
    lines.append("─" * 65)
    lines.append("GRUPPO B — Credit score GLM BASSO (< 40) + OTTIMA gestione debito (< 500%)")
    lines.append(f"  ({len(gruppo_B)} profili)")
    lines.append("─" * 65)
    for _, r in gruppo_B.sort_values('perc_debito_finale').iterrows():
        lines.append(
            f"  Profilo {int(r['profile_id']):<4}  |  FlowScore: {r['flowscore']:.1f}"
            f"  |  debito_richiesto: €{r['debito_richiesto']:.0f}"
            f"  |  final_debt: €{r['final_debt']:.0f}"
            f"  |  perc_finale: {r['perc_debito_finale']*100:.1f}%"
            f"  |  fragility: {r['fragility_index']:.2f}"
        )

    lines.append("")
    lines.append("=" * 65)
    lines.append(f"Totale profili paradosso: {len(gruppo_A) + len(gruppo_B)}"
                 f"  (A={len(gruppo_A)} >= 500%, B={len(gruppo_B)} score basso + < 500%)")
    lines.append("=" * 65)

    text = "\n".join(lines)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(text)

    print(f"\n✓ Report paradosso salvato in {out_path}")
    print(f"  Gruppo A (score alto, gestione pessima): {len(gruppo_A)} profili")
    print(f"  Gruppo B (score basso, gestione ottima): {len(gruppo_B)} profili")


def main():
    df = read_data('data/profiles.csv')
    print(f"=== Dati caricati: {len(df)} profili ===")

    df_processed = processing(df)
    run_simulation(df_processed)
    generate_paradox_report()
    print("\n✓ FATTO! CSV pronto per Chris (Layer 3)")


main()


# ── Test visivo example ───────────────────────────────────────────────────────
# debt_trajectory = example([0,1000,1200,1100,1000,1200], 400, 60, [0,0,0,0,0,2000], 4, 0.5)
# pl.figure()
# pl.plot(range(12), debt_trajectory)
# pl.show()
