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
  id                        int
  monthly_income_hist       str  JSON list of 6 monthly income values (€)
  monthly_fixed_exp_hist    str  JSON list of 6 monthly fixed costs (€)
  monthly_variable_exp_hist str  JSON list of 6 monthly variable spending (€)
  monthly_saving_hist       str  JSON list of 6 running savings balances (€)
  working_category          str  gig | part_time | freelance | fixed_term
  pay_on_time_bills         float  [0,1]  fraction of months balance > -50 €
  int_defaults              int   0 / 1 / 2   historical default count
  bnpl_exposure             float  average monthly BNPL instalment cost (€)

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

# Fraction of mean income spent on variable expenses, by category
TYPE_VAR_FRACTION_RANGE: Dict[str, Tuple[float, float]] = {
    'gig':        (0.25, 0.35),
    'freelance':  (0.30, 0.40),
    'part_time':  (0.35, 0.45),
    'fixed_term': (0.40, 0.50),
}

# Month-to-month volatility of variable expenses (sigma = baseline * factor)
TYPE_VAR_VOLATILITY: Dict[str, float] = {
    'gig':        0.40,
    'freelance':  0.30,
    'part_time':  0.20,
    'fixed_term': 0.15,
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
    base_fixed_exp:   float   # baseline fixed costs (€/month) before any step change
    initial_saving:   float   # starting balance (€)
    bnpl_exposure:    float   # monthly BNPL instalment (€, 0 if unused)


# ──────────────────────────────────────────────────────────────────────────────
# SimPy cash-flow simulation  (income only)
# ──────────────────────────────────────────────────────────────────────────────

class CashFlowSimulation:
    """
    Discrete-event simulation of 6 months of personal income.
    Time unit = 1 day.  Income process is type-specific stochastic.
    Fixed and variable expenses are computed analytically after the simulation.
    """

    def __init__(self, params: ProfileParams, rng: np.random.Generator):
        self.p   = params
        self.rng = rng
        self.env = simpy.Environment()
        self.monthly_income: List[float] = [0.0] * N_MONTHS

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

    # ── Run ───────────────────────────────────────────────────────────────────

    def run(self) -> None:
        income_proc = {
            'gig':        self._gig_income,
            'freelance':  self._freelance_income,
            'part_time':  self._fixed_income,
            'fixed_term': self._fixed_income,
        }[self.p.working_category]

        self.env.process(income_proc())
        self.env.run(until=SIM_DAYS)


# ──────────────────────────────────────────────────────────────────────────────
# Expense generators
# ──────────────────────────────────────────────────────────────────────────────

def generate_fixed_exp_hist(base_fixed: float, rng: np.random.Generator) -> List[float]:
    """
    Generate 6-month history of fixed expenses.
    Base value is constant; ~30% of profiles experience a step change
    (e.g. new subscription) that increases costs from a random month onwards.

    Args:
        base_fixed: baseline monthly fixed cost (€250-550)
        rng:        random generator

    Returns:
        List of 6 monthly fixed expense values (€)
    """
    hist = [base_fixed] * N_MONTHS

    # 30% chance of step change
    if rng.random() < 0.30:
        step_month = int(rng.integers(1, N_MONTHS))   # change starts at month 1-5
        step_size  = float(rng.uniform(30.0, 100.0))
        for m in range(step_month, N_MONTHS):
            hist[m] += step_size

    return [round(v, 2) for v in hist]


def generate_variable_exp_hist(
    income_hist:      List[float],
    working_category: str,
    rng:              np.random.Generator,
) -> List[float]:
    """
    Generate 6-month history of variable expenses with:
      - baseline proportional to mean income (fraction depends on category)
      - month-to-month noise (sigma = baseline × volatility_factor)
      - weak income correlation: +15% of deviation from mean income

    Clipped to [€100, €700] per month.

    Args:
        income_hist:      list of 6 monthly income values (€)
        working_category: employment category
        rng:              random generator

    Returns:
        List of 6 monthly variable expense values (€)
    """
    mean_income       = float(np.mean(income_hist))
    fraction_lo, fraction_hi = TYPE_VAR_FRACTION_RANGE[working_category]
    fraction          = float(rng.uniform(fraction_lo, fraction_hi))
    baseline          = fraction * mean_income
    volatility_factor = TYPE_VAR_VOLATILITY[working_category]
    sigma             = baseline * volatility_factor

    hist: List[float] = []
    for t in range(N_MONTHS):
        noise            = float(rng.normal(0.0, sigma))
        correlation_term = 0.15 * (income_hist[t] - mean_income)
        value            = baseline + noise + correlation_term
        value            = float(np.clip(value, 100.0, 700.0))
        hist.append(round(value, 2))

    return hist


# ──────────────────────────────────────────────────────────────────────────────
# Debt history generator  –  purchase-event simulation
# ──────────────────────────────────────────────────────────────────────────────

def generate_debt_hist_via_purchases(
    bnpl_exposure:     float,
    pay_on_time_bills: float,
    working_category:  str,
    rng:               np.random.Generator,
) -> Tuple[List[float], List[float]]:
    """
    Simulate BNPL purchase events and return planned debt schedule + payment dues.

    Returns
    -------
    bnpl_debt_planned : List[float]
        Remaining BNPL debt at each month assuming all instalments paid on time.
        Scaled so bnpl_debt_planned[5] ≈ bnpl_exposure (calibration anchor).
    bnpl_payment_due : List[float]
        Monthly BNPL instalment due at each month (0 if no payment that month).

    Non-users return ([0]*6, [0]*6).

    Design
    ------
    Archetype governs number of purchases and skip probability:
      Accumulator (pay_on_time < 0.5, common in gig): 2–4 purchases
      Moderate    (0.5 ≤ pay_on_time < 0.8):           1–2 purchases
      Light       (pay_on_time ≥ 0.8):                 1 purchase
    The last purchase is forced to month 3 or 4 to guarantee debt at month 5.
    """
    if bnpl_exposure == 0.0:
        return [0.0] * N_MONTHS, [0.0] * N_MONTHS

    # ── Archetype assignment ───────────────────────────────────────────────────
    if pay_on_time_bills < 0.5:
        archetype   = 'accumulator'
        n_purchases = int(rng.integers(2, 5))   # 2–4
    elif pay_on_time_bills >= 0.8:
        archetype   = 'light'
        n_purchases = 1
    else:
        archetype   = 'moderate'
        n_purchases = int(rng.integers(1, 3))   # 1–2

    # Category nudge
    if working_category == 'gig' and archetype != 'accumulator' and rng.random() < 0.25:
        archetype   = 'accumulator'
        n_purchases = max(n_purchases, 2)
    elif working_category == 'fixed_term' and archetype == 'accumulator' and rng.random() < 0.20:
        archetype   = 'moderate'

    # ── Generate purchase events ───────────────────────────────────────────────
    # Last purchase forced to month 3 or 4 → guarantees debt_planned[5] > 0.
    events: List[Tuple[int, float, int]] = []   # (purchase_month, amount, n_inst)
    for idx in range(n_purchases):
        pm     = int(rng.integers(3, 5)) if idx == n_purchases - 1 else int(rng.integers(0, 5))
        amount = float(rng.uniform(80.0, 400.0))
        n_inst = int(rng.choice([3, 4]))
        events.append((pm, amount, n_inst))

    # ── Build planned-debt and payment-due arrays ─────────────────────────────
    debt_planned = [0.0] * N_MONTHS
    payment_due  = [0.0] * N_MONTHS

    for pm, amount, n_inst in events:
        rate      = amount / n_inst
        remaining = amount
        for t in range(N_MONTHS):
            if t < pm:
                continue
            elif t == pm:
                debt_planned[t] += remaining   # full debt at purchase month
                # no instalment due at purchase month itself
            else:
                installment_idx = t - pm       # 1, 2, 3 …
                if installment_idx <= n_inst and remaining > 0.0:
                    payment_due[t] += rate
                    remaining       = max(0.0, remaining - rate)
                debt_planned[t] += remaining

    # ── Scale to bnpl_exposure so payment sizes are in the right ballpark ────
    if debt_planned[5] > 0.0:
        scale        = bnpl_exposure / debt_planned[5]
        debt_planned = [v * scale for v in debt_planned]
        payment_due  = [v * scale for v in payment_due]
    else:
        # Fallback: should not happen with forced late purchase
        debt_planned = [0.0] * (N_MONTHS - 1) + [bnpl_exposure]

    debt_planned[-1] = bnpl_exposure   # floating-point guard

    return (
        [round(v, 2) for v in debt_planned],
        [round(v, 2) for v in payment_due],
    )


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

    base_fixed_exp = float(rng.uniform(250.0, 550.0))

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
        base_fixed_exp=base_fixed_exp,
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
        params = sample_profile_params(profile_id=i, rng=rng)

        # ── Income simulation (SimPy) ────────────────────────────────────────
        sim = CashFlowSimulation(params=params, rng=rng)
        sim.run()
        income_hist = [round(v, 2) for v in sim.monthly_income]

        # ── Expense generation (analytical) ─────────────────────────────────
        fixed_exp_hist    = generate_fixed_exp_hist(params.base_fixed_exp, rng)
        variable_exp_hist = generate_variable_exp_hist(income_hist, params.working_category, rng)

        # ── Stress balance (can go negative) – used only for pay_on_time + defaults ──
        # Approximates the old model so behavioural features remain diverse.
        stress_balance = params.initial_saving
        stress_hist: List[float] = []
        for m in range(N_MONTHS):
            stress_balance += (
                income_hist[m]
                - fixed_exp_hist[m]
                - variable_exp_hist[m]
                - params.bnpl_exposure   # flat monthly BNPL cost as proxy
            )
            stress_hist.append(stress_balance)

        pay_on_time = sum(s > -50.0 for s in stress_hist) / N_MONTHS

        int_defaults = assign_defaults(
            saving_hist=stress_hist,
            income_hist=income_hist,
            bnpl_exposure=params.bnpl_exposure,
            rng=rng,
        )

        # ── BNPL purchase simulation → planned debt + payment schedule ────────
        bnpl_planned, bnpl_due = generate_debt_hist_via_purchases(
            bnpl_exposure=params.bnpl_exposure,
            pay_on_time_bills=pay_on_time,
            working_category=params.working_category,
            rng=rng,
        )

        # ── Cash-flow simulation: saving_hist ≥ 0, debt_hist = BNPL + revolving ─
        # When cash flow turns negative the deficit becomes revolving debt
        # (overdraft / scoperto); saving_hist is always clamped to 0.
        balance  = params.initial_saving
        revolving = 0.0
        saving_hist: List[float] = []
        debt_hist_raw: List[float] = []

        for m in range(N_MONTHS):
            # Gross cash flow (income minus expenses, before BNPL)
            balance += income_hist[m] - fixed_exp_hist[m] - variable_exp_hist[m]
            if balance < 0.0:
                revolving += -balance
                balance    = 0.0

            # Pay BNPL instalment with whatever is available
            due  = bnpl_due[m]
            paid = min(balance, due)
            balance   -= paid
            revolving += (due - paid)   # missed portion accumulates as revolving debt

            saving_hist.append(round(balance, 2))
            debt_hist_raw.append(bnpl_planned[m] + revolving)

        # ── Calibrate debt_hist so debt[5] == bnpl_exposure exactly ──────────
        # (bnpl_planned[5] == bnpl_exposure, revolving pushes raw total above;
        #  scale the whole series back down to the target.)
        if params.bnpl_exposure == 0.0:
            debt_hist: List[float] = [0.0] * N_MONTHS
        elif debt_hist_raw[-1] > 0.0:
            scale     = params.bnpl_exposure / debt_hist_raw[-1]
            debt_hist = [round(v * scale, 2) for v in debt_hist_raw]
            debt_hist[-1] = round(params.bnpl_exposure, 2)
        else:
            debt_hist = [0.0] * (N_MONTHS - 1) + [round(params.bnpl_exposure, 2)]

        # Debito richiesto: chi² con df=3, scale=300, location=200
        # → picco a €500, minimo €200, coda fino a ~€7000 (99.99° perc.)
        debito_richiesto = round(200.0 + float(rng.chisquare(3)) * 300.0, 2)

        rows.append({
            'id':                        i,
            'monthly_income_hist':       json.dumps(income_hist),
            'monthly_fixed_exp_hist':    json.dumps(fixed_exp_hist),
            'monthly_variable_exp_hist': json.dumps(variable_exp_hist),
            'monthly_saving_hist':       json.dumps(saving_hist),
            'debt_hist':                 json.dumps(debt_hist),
            'working_category':          params.working_category,
            'pay_on_time_bills':         round(pay_on_time, 4),
            'int_defaults':              int_defaults,
            'bnpl_exposure':             round(params.bnpl_exposure, 2),
            'debito_richiesto':          debito_richiesto,
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

    print("\n  Mean expenses per category (mean fixed | mean variable):")
    fixed_means    = df['monthly_fixed_exp_hist'].apply(lambda s: float(np.mean(json.loads(s))))
    variable_means = df['monthly_variable_exp_hist'].apply(lambda s: float(np.mean(json.loads(s))))
    for cat in EMPLOYMENT_TYPES:
        mask = df['working_category'] == cat
        mf = fixed_means[mask].mean()
        mv = variable_means[mask].mean()
        mi = incomes[mask].mean()
        print(f"    {cat:<12}  fixed EUR {mf:.0f}  variable EUR {mv:.0f}  "
              f"(fixed={100*mf/mi:.0f}% | var={100*mv/mi:.0f}% of income)")

    n_step = df['monthly_fixed_exp_hist'].apply(
        lambda s: len(set(json.loads(s))) > 1
    ).sum()
    print(f"\n  Profiles with fixed-cost step change: {n_step} ({100*n_step/len(df):.1f}%)")

    n_zero_saving = df['monthly_saving_hist'].apply(
        lambda s: any(v == 0.0 for v in json.loads(s))
    ).sum()
    print(f"  Profiles with at least one saving=0 month (cash depleted): "
          f"{n_zero_saving} ({100*n_zero_saving/len(df):.1f}%)")

    # Sanity check: saving_hist should always be >= 0
    n_neg = df['monthly_saving_hist'].apply(
        lambda s: any(v < 0 for v in json.loads(s))
    ).sum()
    print(f"  Profiles with negative saving (should be 0): {n_neg}")

    print("\n  int_defaults distribution (response variable):")
    for v in [0, 1, 2]:
        c = int((df['int_defaults'] == v).sum())
        print(f"    int_defaults={v}:  {c:>4}  ({100*c/len(df):.1f}%)")
    n_def = int((df['int_defaults'] > 0).sum())
    print(f"    --- any default:  {n_def:>4}  ({100*n_def/len(df):.1f}%)")

    n_bnpl = int((df['bnpl_exposure'] > 0).sum())
    print(f"\n  BNPL users: {n_bnpl} / {len(df)}")
    print(f"  Mean total debt at month 5 (BNPL users): "
          f"EUR {df.loc[df['bnpl_exposure'] > 0, 'bnpl_exposure'].mean():.0f}")

    # Debt breakdown: BNPL component vs revolving/overdraft
    # Approximate BNPL share: months with debt[t] > 0 before any cash stress
    bnpl_mask = df['bnpl_exposure'] > 0
    saving_zeros = df.loc[bnpl_mask, 'monthly_saving_hist'].apply(
        lambda s: sum(1 for v in json.loads(s) if v == 0.0)
    )
    print(f"  Among BNPL users, avg months with saving=0: {saving_zeros.mean():.1f}")

    print("\n  Debt + saving trajectory (BNPL users, first 10):")
    bnpl_df = df[df['bnpl_exposure'] > 0].head(10)
    for _, row in bnpl_df.iterrows():
        debt_vals  = json.loads(row['debt_hist'])
        save_vals  = json.loads(row['monthly_saving_hist'])
        max_debt   = max(debt_vals)
        peak_month = int(np.argmax(debt_vals))
        final_debt = debt_vals[-1]
        trend      = "repaying" if peak_month < 5 else "growing"
        d_str = "[" + ", ".join(f"{v:.0f}" for v in debt_vals) + "]"
        s_str = "[" + ", ".join(f"{v:.0f}" for v in save_vals) + "]"
        print(f"    ID {int(row['id']):<4}: debt={d_str}  saving={s_str}")
        print(f"           max=€{max_debt:.0f} @month_{peak_month}  "
              f"final=€{final_debt:.0f}  target_bnpl=€{row['bnpl_exposure']:.0f}  ({trend})")

    print(f"\n  Saved -> data/profiles.csv")
    print(f"{'='*55}\n")


if __name__ == '__main__':
    main()
