"""
data/generate_synthetic_data.py
================================
Generates synthetic financial profiles for FlowScore.
Target population: young Italian workers (20-28), entry-level jobs,
                   scarce traditional credit history.

Output
------
  data/profiles.csv  –  500 profiles with 6-month cash-flow histories

Schema
------
  id                   int
  monthly_income_hist  str  JSON list of 6 monthly income values (€)
  monthly_fixed_exp    float  average monthly fixed costs (€)
  monthly_variable_exp float  average monthly variable spending (€)
  monthly_saving_hist  str  JSON list of 6 running savings balances (€)
  working_category     str  gig | part_time | freelance | fixed_term
  pay_on_time_bills    float  [0,1]  fraction of months balance > -50 €
  int_defaults         int   0 / 1 / 2   historical default count
  bnpl_exposure        float  average monthly BNPL instalment cost (€)

Run
---
  cd flowscore/
  python data/generate_synthetic_data.py

Dependencies
------------
  pip install simpy numpy pandas
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple

try:
    import simpy
except ImportError:
    sys.exit("SimPy not installed.  Run: pip install simpy")

import numpy as np
import pandas as pd

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

SEED       = 42
N_PROFILES = 500
N_MONTHS   = 6
MONTH_DAYS = 30
SIM_DAYS   = N_MONTHS * MONTH_DAYS   # 180 days

EMPLOYMENT_TYPES   = ['gig', 'part_time', 'freelance', 'fixed_term']
EMPLOYMENT_WEIGHTS = [0.40,  0.30,        0.20,        0.10]

TYPE_INCOME_RANGE: Dict[str, Tuple[float, float]] = {
    'gig':        (400,  1200),
    'freelance':  (600,  1800),
    'part_time':  (600,  1000),
    'fixed_term': (900,  1500),
}

TYPE_CV_RANGE: Dict[str, Tuple[float, float]] = {
    'gig':        (0.40, 0.70),
    'freelance':  (0.20, 0.40),
    'part_time':  (0.08, 0.15),
    'fixed_term': (0.03, 0.08),
}

GIG_ZERO_MONTH_PROB = 0.12   # probability of a near-zero income month (injury/no gigs)


# ──────────────────────────────────────────────────────────────────────────────
# Profile parameters
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ProfileParams:
    profile_id:       int
    working_category: str
    base_income:      float   # monthly target (€)
    income_cv:        float   # intra-type income variability
    fixed_exp:        float   # monthly fixed costs (€)
    initial_saving:   float   # starting balance (€)
    bnpl_exposure:    float   # monthly BNPL instalment (€, 0 if unused)


# ──────────────────────────────────────────────────────────────────────────────
# SimPy cash-flow simulation
# ──────────────────────────────────────────────────────────────────────────────

class CashFlowSimulation:
    """
    Discrete-event simulation of 6 months of personal cash flow.
    Time unit = 1 day.  Two concurrent processes:
      • income   – type-specific stochastic arrivals
      • variable – Poisson variable spending events
    Fixed costs and BNPL are applied post-simulation during aggregation.
    """

    def __init__(self, params: ProfileParams, rng: np.random.Generator):
        self.p   = params
        self.rng = rng
        self.env = simpy.Environment()
        self.monthly_income:  List[float] = [0.0] * N_MONTHS
        self.monthly_var_exp: List[float] = [0.0] * N_MONTHS

    # ── Income processes ──────────────────────────────────────────────────────

    def _gig_income(self):
        """
        Gig workers: 3-5 irregular payments per month (deliveries, tasks).
        ~12 % chance of a near-zero month (injury, platform ban, no orders).
        """
        p = self.p
        for month_idx in range(N_MONTHS):
            month_start = month_idx * MONTH_DAYS
            month_end   = month_start + MONTH_DAYS

            if self.rng.random() < GIG_ZERO_MONTH_PROB:
                # Advance to end of this month and skip income
                wait = month_end - self.env.now
                if wait > 0:
                    yield self.env.timeout(wait)
                continue

            n_payments = int(self.rng.integers(3, 6))
            offsets = sorted(self.rng.uniform(1.0, MONTH_DAYS - 0.5, size=n_payments))
            base_per_event = p.base_income / n_payments

            for offset in offsets:
                pay_day = month_start + offset
                wait = pay_day - self.env.now
                if wait > 0:
                    yield self.env.timeout(wait)
                amount = float(self.rng.lognormal(
                    mean=math.log(max(base_per_event, 1.0)),
                    sigma=p.income_cv,
                ))
                self.monthly_income[month_idx] += max(0.0, amount)

            # Advance to end of month boundary
            remaining = month_end - self.env.now
            if remaining > 0:
                yield self.env.timeout(remaining)

    def _freelance_income(self):
        """
        Freelance: 1-2 invoices per month, paid after a 5-20 day delay.
        All payment days are pre-sorted to guarantee non-negative timeouts.
        """
        p = self.p
        payments: List[Tuple[float, float]] = []

        for month_idx in range(N_MONTHS):
            month_start = month_idx * MONTH_DAYS
            n_inv = int(self.rng.integers(1, 3))
            for _ in range(n_inv):
                invoice_day = month_start + self.rng.uniform(0.0, MONTH_DAYS * 0.7)
                pay_day     = invoice_day + self.rng.uniform(5.0, 20.0)
                if pay_day >= SIM_DAYS:
                    continue
                per_inv = p.base_income / n_inv
                amount  = max(0.0, float(self.rng.normal(per_inv, per_inv * p.income_cv)))
                payments.append((pay_day, amount))

        payments.sort()
        for pay_day, amount in payments:
            wait = pay_day - self.env.now
            if wait > 0:
                yield self.env.timeout(wait)
            m = min(int(self.env.now / MONTH_DAYS), N_MONTHS - 1)
            self.monthly_income[m] += amount

    def _fixed_income(self):
        """
        Part-time / fixed-term: monthly salary paid around day 25-28.
        """
        p = self.p
        payments: List[Tuple[float, float]] = []

        for month_idx in range(N_MONTHS):
            pay_day = month_idx * MONTH_DAYS + self.rng.uniform(24.0, 29.0)
            if pay_day >= SIM_DAYS:
                break
            amount = max(0.0, float(self.rng.normal(p.base_income, p.base_income * p.income_cv)))
            payments.append((pay_day, amount))

        for pay_day, amount in payments:
            wait = pay_day - self.env.now
            if wait > 0:
                yield self.env.timeout(wait)
            m = min(int(self.env.now / MONTH_DAYS), N_MONTHS - 1)
            self.monthly_income[m] += amount

    # ── Variable expense process ──────────────────────────────────────────────

    def _variable_expenses(self, monthly_target: float):
        """
        Poisson variable purchases (food delivery, aperitivi, clothing…).
        ~10 events/month (mean inter-arrival 3 days).
        Amount ~ LogNormal calibrated to monthly_target.
        """
        mean_per_event = max(monthly_target / 10.0, 1.0)
        while True:
            yield self.env.timeout(self.rng.exponential(3.0))
            if self.env.now >= SIM_DAYS:
                break
            m = min(int(self.env.now / MONTH_DAYS), N_MONTHS - 1)
            amount = float(self.rng.lognormal(
                mean=math.log(mean_per_event),
                sigma=0.5,
            ))
            self.monthly_var_exp[m] += amount

    # ── Run ───────────────────────────────────────────────────────────────────

    def run(self, monthly_var_target: float) -> None:
        income_proc = {
            'gig':        self._gig_income,
            'freelance':  self._freelance_income,
            'part_time':  self._fixed_income,
            'fixed_term': self._fixed_income,
        }[self.p.working_category]

        self.env.process(income_proc())
        self.env.process(self._variable_expenses(monthly_var_target))
        self.env.run(until=SIM_DAYS)


# ──────────────────────────────────────────────────────────────────────────────
# Profile parameter sampler (causal structure)
# ──────────────────────────────────────────────────────────────────────────────

def sample_profile_params(profile_id: int, rng: np.random.Generator) -> ProfileParams:
    """
    Causal chain:
      working_category → base_income, income_cv
      working_category, age → P(uses_bnpl)
      family_support (30%) → higher initial_saving
    """
    cat = str(rng.choice(EMPLOYMENT_TYPES, p=EMPLOYMENT_WEIGHTS))
    age = int(rng.integers(20, 29))

    lo, hi = TYPE_INCOME_RANGE[cat]
    base_income = float(rng.uniform(lo, hi))
    income_cv   = float(rng.uniform(*TYPE_CV_RANGE[cat]))

    fixed_exp = float(rng.uniform(250.0, 550.0))

    # BNPL: more common among gig workers and age 22-26
    p_bnpl = 0.25
    if cat == 'gig':
        p_bnpl += 0.20
    if 22 <= age <= 26:
        p_bnpl += 0.12
    p_bnpl = min(p_bnpl, 0.70)

    uses_bnpl     = bool(rng.random() < p_bnpl)
    bnpl_exposure = float(rng.uniform(30.0, 200.0)) if uses_bnpl else 0.0

    # Initial savings: 70% start near zero (paycheck-to-paycheck), 30% have family help
    if rng.random() < 0.70:
        initial_saving = float(rng.uniform(0.0, 500.0))
    else:
        initial_saving = float(rng.uniform(500.0, 2000.0))

    return ProfileParams(
        profile_id=profile_id,
        working_category=cat,
        base_income=base_income,
        income_cv=income_cv,
        fixed_exp=fixed_exp,
        initial_saving=initial_saving,
        bnpl_exposure=bnpl_exposure,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Default label – true logistic DGP
# ──────────────────────────────────────────────────────────────────────────────

def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def assign_defaults(
    saving_hist:   List[float],
    income_hist:   List[float],
    bnpl_exposure: float,
    rng:           np.random.Generator,
) -> int:
    """
    True DGP for int_defaults (0 / 1 / 2).
    Target rates:  ~75 % no default,  ~20 % one default,  ~5 % two defaults.

    Drivers:
      min_saving  – worst running balance (main signal)
      cv_income   – income unpredictability
      bnpl_norm   – monthly BNPL burden relative to €200 cap
    """
    min_saving   = min(saving_hist)
    mean_income  = max(float(np.mean(income_hist)), 1.0)
    cv_income    = float(np.std(income_hist)) / mean_income
    bnpl_norm    = np.clip(bnpl_exposure / 200.0, 0.0, 1.0)

    # Normalise min_saving to [-3, 2] range
    min_sav_norm = float(np.clip(min_saving / 1000.0, -3.0, 2.0))

    logit_p1 = (
        -2.20
        - 1.50 * min_sav_norm   # deeper negative savings → higher risk
        + 1.00 * cv_income       # volatile income → higher risk
        + 1.80 * bnpl_norm       # BNPL burden → higher risk
    )

    p1 = _sigmoid(logit_p1)           # P(int_defaults >= 1)
    p2 = _sigmoid(logit_p1 - 2.8)     # P(int_defaults >= 2)  (much rarer)

    u = rng.random()
    if u < p2:
        return 2
    elif u < p1:
        return 1
    return 0


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    rng  = np.random.default_rng(SEED)
    rows: List[Dict] = []

    print(f"FlowScore synthetic data generator  (N={N_PROFILES}, seed={SEED})")
    print("Simulating 6-month cash flows...")

    for i in range(N_PROFILES):
        params             = sample_profile_params(profile_id=i, rng=rng)
        monthly_var_target = float(rng.uniform(150.0, 500.0))

        sim = CashFlowSimulation(params=params, rng=rng)
        sim.run(monthly_var_target=monthly_var_target)

        income_hist = [round(v, 2) for v in sim.monthly_income]
        var_exp     = [round(v, 2) for v in sim.monthly_var_exp]

        # ── Iterative saving balance ─────────────────────────────────────────
        saving_hist: List[float] = []
        balance = params.initial_saving
        for m in range(N_MONTHS):
            balance = (
                balance
                + income_hist[m]
                - params.fixed_exp
                - var_exp[m]
                - params.bnpl_exposure
            )
            saving_hist.append(round(balance, 2))

        # ── Derived features ─────────────────────────────────────────────────
        # Fraction of months where balance stayed above -€50 (small overdraft tolerated)
        pay_on_time = sum(s > -50.0 for s in saving_hist) / N_MONTHS

        int_defaults = assign_defaults(
            saving_hist=saving_hist,
            income_hist=income_hist,
            bnpl_exposure=params.bnpl_exposure,
            rng=rng,
        )

        rows.append({
            'id':                   i,
            'monthly_income_hist':  json.dumps(income_hist),
            'monthly_fixed_exp':    round(params.fixed_exp, 2),
            'monthly_variable_exp': round(float(np.mean(var_exp)), 2),
            'monthly_saving_hist':  json.dumps(saving_hist),
            'working_category':     params.working_category,
            'pay_on_time_bills':    round(pay_on_time, 4),
            'int_defaults':         int_defaults,
            'bnpl_exposure':        round(params.bnpl_exposure, 2),
        })

        if (i + 1) % 100 == 0:
            print(f"  {i + 1:>4}/{N_PROFILES} profiles done")

    df = pd.DataFrame(rows)
    df.to_csv('data/profiles.csv', index=False)

    # ── Summary statistics ────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print("  FLOWSCORE SYNTHETIC DATASET - SUMMARY STATISTICS")
    print(f"{'='*55}")
    print(f"  Rows: {len(df)}  |  Columns: {df.shape[1]}")

    print("\n  Working category distribution:")
    counts = df['working_category'].value_counts()
    for cat in EMPLOYMENT_TYPES:
        c = counts.get(cat, 0)
        print(f"    {cat:<12}  {c:>4}  ({100*c/len(df):.1f}%)")

    print("\n  Mean monthly income per category:")
    incomes = df['monthly_income_hist'].apply(lambda s: float(np.mean(json.loads(s))))
    for cat in EMPLOYMENT_TYPES:
        mask = df['working_category'] == cat
        mi = incomes[mask].mean()
        print(f"    {cat:<12}  EUR {mi:.0f}/month")

    n_neg = df['monthly_saving_hist'].apply(
        lambda s: any(v < 0 for v in json.loads(s))
    ).sum()
    print(f"\n  Profiles with any negative saving balance: {n_neg} ({100*n_neg/len(df):.1f}%)")

    print("\n  int_defaults distribution (response variable):")
    for v in [0, 1, 2]:
        c = int((df['int_defaults'] == v).sum())
        print(f"    int_defaults={v}:  {c:>4}  ({100*c/len(df):.1f}%)")
    n_def = int((df['int_defaults'] > 0).sum())
    print(f"    --- any default:  {n_def:>4}  ({100*n_def/len(df):.1f}%)")

    print(f"\n  BNPL users: {int((df['bnpl_exposure'] > 0).sum())} / {len(df)}")
    print(f"  Mean BNPL (when active): "
          f"EUR {df.loc[df['bnpl_exposure'] > 0, 'bnpl_exposure'].mean():.0f}/month")

    print(f"\n  Saved -> data/profiles.csv")
    print(f"{'='*55}\n")


if __name__ == '__main__':
    main()
