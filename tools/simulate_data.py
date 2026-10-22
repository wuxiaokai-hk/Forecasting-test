"""
simulate_data.py
================
Creates SIMULATED inputs that mimic the real ones, so the pipeline can be run end to end
before the real data is plugged in.

Files written to ./data :
  bank_data.csv            date, loan_balance, deposit_balance            ($ millions, quarter-end)
  bank_events.csv          date, amount, description                      (inorganic events, e.g. acquisitions)
  moodys_baseline.csv      date + ~400 macro variables (history 2000Q1-2025Q4 + forecast to 2030Q4)
  moodys_s1_upside.csv     same columns, upside scenario
  moodys_s3_downside.csv   same columns, moderate recession
  moodys_s4_severe.csv     same columns, severe recession
  macro_metadata.csv       mnemonic, description, geography, theme, transform, expected_sign

The mnemonics imitate the Moody's style (e.g. FGDPR.ION = Ontario real GDP) but are illustrative only.
The "true" loan equation used to generate the data is printed at the end, so you can check
whether the variable-selection pipeline recovers the real drivers.
"""
import os
import numpy as np
import pandas as pd

SEED = 42
rng = np.random.default_rng(SEED)
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
os.makedirs(OUT, exist_ok=True)

Q = pd.period_range("2000Q1", "2030Q4", freq="Q")
N = len(Q)
HIST_END = pd.Period("2025Q4", freq="Q")
NH = int((Q <= HIST_END).sum())          # number of history quarters
NF = N - NH                              # number of scenario quarters
t = np.arange(N)


def qpos(q):
    return int(np.where(Q == pd.Period(q, freq="Q"))[0][0])


def anchors(points, n=NH):
    """Piecewise-linear path through {quarter: value} anchors (history only)."""
    xs = [qpos(k) for k in points]
    return np.interp(np.arange(n), xs, list(points.values()))


def episode(arr, start, end, value):
    a, b = qpos(start), qpos(end)
    arr[a:b + 1] += value


# ---------------------------------------------------------------------------
# 1. CORE MACRO HISTORY (Ontario / Canada), loosely shaped on actual history
# ---------------------------------------------------------------------------
h = {}
# Real GDP growth, Ontario (quarterly log change)
g = 0.005 + rng.normal(0, 0.0035, NH)
for q, v in {"2001Q3": -0.004, "2008Q4": -0.013, "2009Q1": -0.020, "2009Q2": -0.006,
             "2015Q1": -0.003, "2020Q1": -0.020, "2020Q2": -0.115, "2020Q3": 0.090,
             "2020Q4": 0.020, "2021Q1": 0.012, "2023Q2": -0.003, "2025Q2": -0.004}.items():
    g[qpos(q)] += v
h["gdp"] = g

h["policy"] = anchors({"2000Q1": 5.0, "2000Q4": 5.75, "2001Q4": 2.25, "2002Q4": 2.75, "2004Q2": 2.0,
                       "2006Q2": 4.25, "2007Q4": 4.25, "2008Q4": 1.5, "2009Q2": 0.25, "2010Q3": 1.0,
                       "2014Q4": 1.0, "2015Q3": 0.5, "2017Q3": 1.0, "2018Q4": 1.75, "2019Q4": 1.75,
                       "2020Q2": 0.25, "2021Q4": 0.25, "2022Q4": 4.25, "2023Q3": 5.0, "2024Q2": 5.0,
                       "2024Q4": 3.25, "2025Q2": 2.75, "2025Q4": 2.25})
h["bond10"] = anchors({"2000Q1": 6.0, "2003Q1": 4.8, "2007Q2": 4.4, "2008Q4": 3.0, "2011Q1": 3.2,
                       "2012Q3": 1.8, "2014Q1": 2.5, "2016Q2": 1.2, "2018Q3": 2.4, "2019Q3": 1.4,
                       "2020Q2": 0.6, "2021Q4": 1.6, "2022Q4": 3.2, "2023Q3": 3.8, "2024Q3": 3.0,
                       "2025Q4": 3.2}) + rng.normal(0, 0.08, NH)

cpi = 0.005 + rng.normal(0, 0.0025, NH)
episode(cpi, "2008Q4", "2008Q4", -0.009)
episode(cpi, "2021Q2", "2022Q3", 0.010)
h["cpi"] = cpi

pop = np.full(NH, 0.0033) + rng.normal(0, 0.0005, NH)
episode(pop, "2020Q2", "2021Q1", -0.0025)
episode(pop, "2022Q3", "2024Q2", 0.0045)
episode(pop, "2025Q1", "2025Q4", -0.0028)
h["pop"] = pop

hshock = rng.normal(0, 0.008, NH)
for a, b, v in [("2008Q4", "2009Q1", -0.025), ("2016Q1", "2017Q1", 0.030), ("2017Q2", "2017Q4", -0.030),
                ("2020Q3", "2022Q1", 0.035), ("2022Q2", "2023Q1", -0.040), ("2024Q1", "2025Q4", -0.012)]:
    episode(hshock, a, b, v)
h["hpi_shock"] = hshock

uadj = rng.normal(0, 0.12, NH)
episode(uadj, "2025Q1", "2025Q3", 0.30)
h["u_shock"] = uadj

inc_shock = rng.normal(0, 0.004, NH)
episode(inc_shock, "2020Q2", "2020Q2", 0.050)
episode(inc_shock, "2020Q3", "2020Q4", -0.012)
h["inc_shock"] = inc_shock

h["tsx_shock"] = rng.normal(0, 0.045, NH)
for q, v in {"2001Q3": -0.12, "2002Q3": -0.10, "2008Q4": -0.28, "2009Q1": -0.06, "2020Q1": -0.24,
             "2020Q2": 0.15, "2022Q2": -0.12}.items():
    h["tsx_shock"][qpos(q)] += v
h["oil"] = rng.normal(0.004, 0.12, NH)
h["oil"][qpos("2008Q4")] -= 0.6
h["oil"][qpos("2014Q4")] -= 0.35
h["oil"][qpos("2020Q2")] -= 0.5
h["oil"][qpos("2020Q4")] += 0.25
h["fx"] = rng.normal(0, 0.03, NH)
h["us_shock"] = rng.normal(0, 0.003, NH)

# ---------------------------------------------------------------------------
# 2. SCENARIO PATHS for 2026Q1-2030Q4 (deterministic, smooth like Moody's forecasts)
# ---------------------------------------------------------------------------


def scen_path(name):
    f = {}
    k = np.arange(NF)
    if name == "baseline":
        f["gdp"] = np.full(NF, 0.0045)
        f["policy"] = np.interp(k, [0, 8, NF - 1], [2.25, 2.75, 2.75])
        f["bond10"] = np.interp(k, [0, NF - 1], [3.25, 3.5])
        f["hpi_shock"] = np.interp(k, [0, 4, NF - 1], [-0.006, 0.0, 0.0])
        f["tsx_shock"] = np.zeros(NF)
        f["cpi_add"] = 0.0
    elif name == "s1_upside":
        f["gdp"] = np.interp(k, [0, 8, NF - 1], [0.0070, 0.0060, 0.0050])
        f["policy"] = np.interp(k, [0, 6, NF - 1], [2.5, 3.25, 3.25])
        f["bond10"] = np.interp(k, [0, NF - 1], [3.4, 3.8])
        f["hpi_shock"] = np.interp(k, [0, 6, NF - 1], [0.004, 0.003, 0.0])
        f["tsx_shock"] = np.interp(k, [0, 4, NF - 1], [0.03, 0.01, 0.0])
        f["cpi_add"] = 0.0005
    elif name == "s3_downside":
        f["gdp"] = np.array([0.0005, -0.006, -0.008, -0.003, 0.000] + [0.0065] * 6 + [0.0050] * (NF - 11))
        f["policy"] = np.interp(k, [0, 4, 10, NF - 1], [2.0, 1.0, 1.0, 2.25])
        f["bond10"] = np.interp(k, [0, 4, NF - 1], [3.0, 2.6, 3.2])
        f["hpi_shock"] = np.array([-0.010, -0.020, -0.025, -0.020, -0.010] + [0.002] * 6 + [0.0] * (NF - 11))
        f["tsx_shock"] = np.array([-0.06, -0.10, -0.04, 0.02] + [0.02] * 6 + [0.0] * (NF - 10))
        f["cpi_add"] = -0.001
    elif name == "s4_severe":
        f["gdp"] = np.array([-0.004, -0.012, -0.018, -0.010, -0.004, 0.002] + [0.0080] * 6 + [0.0055] * (NF - 12))
        f["policy"] = np.interp(k, [0, 3, 12, NF - 1], [1.75, 0.25, 0.25, 1.75])
        f["bond10"] = np.interp(k, [0, 4, NF - 1], [2.7, 2.0, 2.8])
        f["hpi_shock"] = np.array([-0.020, -0.040, -0.045, -0.040, -0.030, -0.015] + [0.003] * 6 + [0.0] * (NF - 12))
        f["tsx_shock"] = np.array([-0.12, -0.18, -0.08, 0.03] + [0.03] * 6 + [0.0] * (NF - 10))
        f["cpi_add"] = -0.002
    f["cpi"] = np.full(NF, 0.005 + f["cpi_add"])
    f["pop"] = np.full(NF, 0.0018)
    f["u_shock"] = np.zeros(NF)
    f["inc_shock"] = np.zeros(NF)
    f["oil"] = np.zeros(NF) + (f["gdp"] - 0.0045) * 4
    f["fx"] = -(f["gdp"] - 0.0045) * 2
    f["us_shock"] = np.zeros(NF)
    return f


def build_core(name):
    f = scen_path(name)
    c = {k: np.concatenate([h[k], f[k]]) for k in ["gdp", "policy", "bond10", "cpi", "pop", "hpi_shock",
                                                   "u_shock", "inc_shock", "tsx_shock", "oil", "fx", "us_shock"]}
    c["bond5"] = 0.65 * c["bond10"] + 0.35 * c["policy"] + 0.1
    c["mort5"] = c["bond5"] + 1.55
    c["prime"] = c["policy"] + 2.2
    c["dmort"] = np.r_[0, np.diff(c["mort5"])]
    # Unemployment rate, Ontario: Okun's law + mean reversion
    u = np.zeros(N)
    u[0] = 5.8
    for i in range(1, N):
        u[i] = u[i - 1] - 0.42 * (c["gdp"][i] - 0.005) * 100 - 0.06 * (u[i - 1] - 6.6) + c["u_shock"][i]
    c["unemp"] = u
    c["du"] = np.r_[0, np.diff(u)]
    # Nominal disposable income growth
    c["inc"] = 0.55 * c["gdp"] + 0.85 * c["cpi"] + 0.3 * c["pop"] + 0.0005 + c["inc_shock"]
    # House prices, Ontario
    c["hpi"] = 0.011 + 0.45 * (c["gdp"] - 0.005) - 0.015 * c["dmort"] + c["hpi_shock"]
    c["emp"] = c["pop"] - c["du"] / 100 * 1.05
    c["tsx"] = 0.015 + 2.5 * (c["gdp"] - 0.005) + c["tsx_shock"]
    c["us"] = 0.0055 + 0.6 * (c["gdp"] - 0.005) + c["us_shock"]
    c["starts"] = 75 + 900 * pd.Series(c["gdp"] - 0.005).rolling(4, min_periods=1).mean().values \
        - 6 * (c["mort5"] - 5) + 30 * (c["pop"] - 0.0033) * 100
    return c


# ---------------------------------------------------------------------------
# 3. ~400 DERIVED MOODY'S-STYLE VARIABLES
#    growth-type ("level") series: g = mean_parent*(1-beta) + beta*parent + mu + noise -> level
#    rate-type series:            x = base + beta*(parent - mean_parent) + AR(1) noise
# ---------------------------------------------------------------------------
specs = []   # (code, desc, geo, theme, kind, parent, beta, noise, base, mu, transform, expected_sign)


def add(code, desc, geo, theme, kind, parent, beta, noise, base, mu=0.0, sign=0):
    transform = "dlog" if kind == "level" else "diff"
    specs.append(dict(code=code, desc=desc, geo=geo, theme=theme, kind=kind, parent=parent, beta=beta,
                      noise=noise, base=base, mu=mu, transform=transform, expected_sign=sign))


PROV = {"ION": ("Ontario", 1.0), "IQC": ("Quebec", 0.75), "IBC": ("British Columbia", 0.6),
        "IAB": ("Alberta", 0.35), "IMB": ("Manitoba", 0.55), "ISK": ("Saskatchewan", 0.3),
        "INS": ("Nova Scotia", 0.5), "INB": ("New Brunswick", 0.5), "INL": ("Newfoundland", 0.25),
        "IPE": ("Prince Edward Island", 0.45)}
for gc, (gname, b) in PROV.items():
    on = gc == "ION"
    nz = 0.25 if on else 1.0              # Ontario series are close to the core drivers
    add(f"FGDPR.{gc}", f"Real GDP, {gname} (C$ 2017, SAAR)", gname, "Output", "level", "gdp", b, 0.002 * nz, 600 if on else 100, sign=1)
    add(f"FGDP$.{gc}", f"Nominal GDP, {gname} (C$, SAAR)", gname, "Output", "level", "inc", b, 0.003 * nz, 650 if on else 110, sign=1)
    add(f"FET.{gc}", f"Total employment, {gname} (Ths.)", gname, "Labour", "level", "emp", b, 0.002 * nz, 6500 if on else 900, sign=1)
    add(f"FLBR.{gc}", f"Unemployment rate, {gname} (%)", gname, "Labour", "rate", "unemp", b, 0.10 * nz, 6.5 if on else 7.5, sign=-1)
    add(f"FLF.{gc}", f"Labour force, {gname} (Ths.)", gname, "Labour", "level", "pop", b, 0.002 * nz, 7000 if on else 1000, sign=1)
    add(f"FPOPT.{gc}", f"Total population, {gname} (Ths.)", gname, "Demographics", "level", "pop", b, 0.0006 * nz, 12000 if on else 1500, sign=1)
    add(f"FYPD.{gc}", f"Personal disposable income, {gname} (C$)", gname, "Income", "level", "inc", b, 0.0035 * nz, 400 if on else 60, sign=1)
    add(f"FRS.{gc}", f"Retail sales, {gname} (C$)", gname, "Consumption", "level", "inc", b * 0.9, 0.008, 150 if on else 25, sign=1)
    add(f"FHST.{gc}", f"Housing starts, {gname} (Ths., SAAR)", gname, "Housing", "level", "gdp", b * 1.5, 0.08, 75 if on else 10, sign=1)
    add(f"FHPI.{gc}", f"House price index, {gname} (2005=100)", gname, "Housing", "level", "hpi", b, 0.003 * nz, 100, sign=1)
    add(f"FCPI.{gc}", f"Consumer price index, {gname} (2002=100)", gname, "Prices", "level", "cpi", b * 0.9 + 0.1, 0.0015, 100, sign=0)
    add(f"FAWE.{gc}", f"Average weekly earnings, {gname} (C$)", gname, "Income", "level", "inc", b * 0.6, 0.004, 800, sign=1)
    add(f"FBKCPT.{gc}", f"Consumer insolvencies, {gname} (#)", gname, "Credit quality", "level", "gdp", -2.0 * b, 0.04, 20 if on else 3, sign=-1)

CMA = {"ITOR": "Toronto", "IOTT": "Ottawa-Gatineau", "IHAM": "Hamilton", "IKIT": "Kitchener-Cambridge-Waterloo",
       "ILON": "London", "IWIN": "Windsor", "IOSH": "Oshawa", "ISTC": "St. Catharines-Niagara", "IKIN": "Kingston",
       "IGUE": "Guelph", "IBAR": "Barrie", "ISUD": "Greater Sudbury"}
for gc, gname in CMA.items():
    b = float(rng.uniform(0.75, 1.1))
    add(f"FET.{gc}", f"Total employment, {gname} CMA (Ths.)", "Ontario - " + gname, "Labour", "level", "emp", b, 0.004, 500, sign=1)
    add(f"FLBR.{gc}", f"Unemployment rate, {gname} CMA (%)", "Ontario - " + gname, "Labour", "rate", "unemp", b, 0.35, 6.5, sign=-1)
    add(f"FHPI.{gc}", f"House price index, {gname} CMA (2005=100)", "Ontario - " + gname, "Housing", "level", "hpi", b, 0.007, 100, sign=1)
    add(f"FHST.{gc}", f"Housing starts, {gname} CMA (#, SAAR)", "Ontario - " + gname, "Housing", "level", "gdp", b * 1.6, 0.10, 5000, sign=1)
    add(f"FPOPT.{gc}", f"Population, {gname} CMA (Ths.)", "Ontario - " + gname, "Demographics", "level", "pop", b, 0.0012, 500, sign=1)
    add(f"FGDPR.{gc}", f"Real GDP, {gname} CMA (C$ 2017)", "Ontario - " + gname, "Output", "level", "gdp", b, 0.004, 50, sign=1)

# National block
C = "Canada"
nat = [
    ("FGDPR.IC", "Real GDP, Canada", "Output", "level", "gdp", 0.9, 0.0015, 2000, 1),
    ("FGDP$.IC", "Nominal GDP, Canada", "Output", "level", "inc", 0.9, 0.003, 2300, 1),
    ("FCR.IC", "Real personal consumption, Canada", "Consumption", "level", "gdp", 0.7, 0.003, 1100, 1),
    ("FIR.IC", "Real residential investment, Canada", "Housing", "level", "hpi", 0.8, 0.012, 150, 1),
    ("FINR.IC", "Real business non-res. investment, Canada", "Investment", "level", "gdp", 1.8, 0.012, 200, 1),
    ("FGR.IC", "Real government spending, Canada", "Government", "level", "gdp", -0.2, 0.004, 450, 0),
    ("FXR.IC", "Real exports, Canada", "Trade", "level", "us", 1.5, 0.012, 600, 1),
    ("FMR.IC", "Real imports, Canada", "Trade", "level", "gdp", 1.4, 0.012, 620, 0),
    ("FPGDP.IC", "GDP deflator, Canada", "Prices", "level", "cpi", 1.1, 0.003, 100, 0),
    ("FCPI.IC", "CPI all items, Canada", "Prices", "level", "cpi", 1.0, 0.0008, 100, 0),
    ("FCPIXFE.IC", "CPI ex food & energy, Canada", "Prices", "level", "cpi", 0.6, 0.0012, 100, 0),
    ("FCPISHL.IC", "CPI shelter, Canada", "Prices", "level", "cpi", 0.9, 0.002, 100, 0),
    ("FCPIRENT.IC", "CPI rent, Canada", "Prices", "level", "pop", 0.8, 0.003, 100, 0),
    ("FPPI.IC", "Industrial product price index, Canada", "Prices", "level", "oil", 0.15, 0.006, 100, 0),
    ("FRBOC.IC", "Bank of Canada overnight target rate (%)", "Interest rates", "rate", "policy", 1.0, 0.0, 2.5, -1),
    ("FRPRIME.IC", "Chartered bank prime rate (%)", "Interest rates", "rate", "prime", 1.0, 0.0, 4.5, -1),
    ("FRTB3M.IC", "3-month T-bill yield (%)", "Interest rates", "rate", "policy", 0.97, 0.06, 2.4, -1),
    ("FRGOV2Y.IC", "2-year GoC bond yield (%)", "Interest rates", "rate", "bond5", 1.05, 0.10, 2.6, -1),
    ("FRGOV5Y.IC", "5-year GoC bond yield (%)", "Interest rates", "rate", "bond5", 1.0, 0.03, 2.9, -1),
    ("FRGOV10Y.IC", "10-year GoC bond yield (%)", "Interest rates", "rate", "bond10", 1.0, 0.0, 3.2, -1),
    ("FRGOV30Y.IC", "30-year GoC bond yield (%)", "Interest rates", "rate", "bond10", 0.85, 0.08, 3.4, -1),
    ("FRMORT1Y.IC", "1-year conventional mortgage rate (%)", "Interest rates", "rate", "prime", 0.9, 0.12, 5.0, -1),
    ("FRMORT3Y.IC", "3-year conventional mortgage rate (%)", "Interest rates", "rate", "mort5", 0.95, 0.10, 5.3, -1),
    ("FRMORT5Y.IC", "5-year conventional mortgage rate (%)", "Interest rates", "rate", "mort5", 1.0, 0.02, 5.5, -1),
    ("FRCORPBBB.IC", "BBB corporate bond yield (%)", "Interest rates", "rate", "bond10", 0.9, 0.25, 4.8, -1),
    ("FRSPREAD.IC", "Corporate spread over GoC (pp)", "Financial markets", "rate", "unemp", 0.25, 0.15, 1.5, -1),
    ("FSTSX.IC", "S&P/TSX composite index", "Financial markets", "level", "tsx", 1.0, 0.01, 10000, 1),
    ("FVIXC.IC", "Implied volatility index (VIXC)", "Financial markets", "level", "tsx", -1.5, 0.06, 18, -1),
    ("FOILWTI.IC", "WTI crude oil price (US$/bbl)", "Commodities", "level", "oil", 1.0, 0.02, 60, 0),
    ("FOILWCS.IC", "Western Canada Select (US$/bbl)", "Commodities", "level", "oil", 1.1, 0.05, 45, 0),
    ("FNATGAS.IC", "Natural gas price, AECO (C$/GJ)", "Commodities", "level", "oil", 0.6, 0.10, 3, 0),
    ("FBCPI.IC", "BoC commodity price index", "Commodities", "level", "oil", 0.5, 0.03, 500, 0),
    ("FXUSCAD.IC", "Exchange rate, C$ per US$", "Trade", "level", "fx", 1.0, 0.0, 1.3, 0),
    ("FXEURCAD.IC", "Exchange rate, C$ per Euro", "Trade", "level", "fx", 0.7, 0.025, 1.45, 0),
    ("FM1.IC", "Money supply M1+, Canada", "Money & credit", "level", "inc", 1.2, 0.008, 900, 1),
    ("FM2.IC", "Money supply M2, Canada", "Money & credit", "level", "inc", 1.0, 0.005, 1500, 1),
    ("FM2P.IC", "Money supply M2++, Canada", "Money & credit", "level", "inc", 0.9, 0.005, 2200, 1),
    ("FCRHH.IC", "Household credit outstanding, Canada", "Money & credit", "level", "hpi", 0.35, 0.004, 2500, 1),
    ("FCRMORT.IC", "Residential mortgage credit, Canada", "Money & credit", "level", "hpi", 0.4, 0.004, 1900, 1),
    ("FCRCONS.IC", "Consumer credit, Canada", "Money & credit", "level", "inc", 0.6, 0.005, 650, 1),
    ("FCRBUS.IC", "Business credit, Canada", "Money & credit", "level", "gdp", 1.2, 0.007, 2000, 1),
    ("FDTI.IC", "Household debt-to-disposable income (%)", "Money & credit", "rate", "mort5", -2.0, 1.2, 170, 0),
    ("FSAVR.IC", "Household saving rate (%)", "Income", "rate", "unemp", 0.6, 0.6, 5.0, 0),
    ("FCAPU.IC", "Capacity utilisation, non-farm (%)", "Output", "rate", "unemp", -1.4, 0.6, 81, 1),
    ("FBKBUS.IC", "Business insolvencies, Canada (#)", "Credit quality", "level", "gdp", -1.6, 0.05, 900, -1),
    ("FDELMORT.IC", "Mortgage arrears rate, 90+ days (%)", "Credit quality", "rate", "unemp", 0.04, 0.02, 0.3, -1),
    ("FDELCC.IC", "Credit card delinquency rate (%)", "Credit quality", "rate", "unemp", 0.12, 0.08, 1.0, -1),
    ("FHSEXIST.IC", "Existing home sales, Canada (Ths., SAAR)", "Housing", "level", "hpi", 1.8, 0.05, 480, 1),
    ("FHSLR.IC", "Sales-to-new-listings ratio (%)", "Housing", "rate", "unemp", -3.0, 3.0, 55, 1),
    ("FBPRES.IC", "Residential building permits (C$)", "Housing", "level", "hpi", 1.4, 0.06, 3000, 1),
    ("FBPNRES.IC", "Non-residential building permits (C$)", "Investment", "level", "gdp", 2.0, 0.08, 2000, 1),
    ("FAUTO.IC", "New motor vehicle sales (units)", "Consumption", "level", "gdp", 2.2, 0.04, 450, 1),
    ("FCONF.IC", "Consumer confidence index", "Sentiment", "level", "tsx", 0.3, 0.03, 100, 1),
    ("FBOS.IC", "BoC business outlook indicator", "Sentiment", "rate", "unemp", -0.6, 0.5, 0.0, 1),
    ("FPOPIMM.IC", "Net international migration, Canada (Ths.)", "Demographics", "level", "pop", 5.0, 0.02, 300, 1),
    ("FWAGE.IC", "Average hourly wages, Canada", "Income", "level", "inc", 0.5, 0.003, 30, 1),
    ("FPRODL.IC", "Labour productivity, business sector", "Output", "level", "gdp", 0.4, 0.004, 100, 0),
    ("FULC.IC", "Unit labour costs, business sector", "Prices", "level", "cpi", 0.8, 0.004, 100, 0),
    ("FRENTVAC.IC", "Rental vacancy rate (%)", "Housing", "rate", "pop", -1.5, 0.3, 2.5, 0),
]
for code, desc, theme, kind, parent, beta, noise, base, sign in nat:
    add(code, desc, C, theme, kind, parent, beta, noise, base, sign=sign)

SECT = ["Agriculture", "Mining & oil/gas", "Utilities", "Construction", "Manufacturing", "Wholesale trade",
        "Retail trade", "Transportation", "Information & culture", "Finance & insurance", "Real estate",
        "Professional services", "Management of companies", "Administrative support", "Educational services",
        "Health care", "Arts & recreation", "Accommodation & food", "Other services", "Public administration"]
SB = [0.3, 0.6, 0.2, 1.6, 1.8, 1.3, 1.1, 1.4, 0.8, 0.7, 0.9, 0.9, 0.8, 1.2, 0.2, 0.2, 2.5, 3.0, 1.0, 0.1]
for i, (s, b) in enumerate(zip(SECT, SB)):
    tag = f"N{i + 1:02d}"
    for geo, gname in [("ION", "Ontario"), ("IC", "Canada")]:
        par = "oil" if (s.startswith("Mining") and geo == "IC") else "gdp"
        bb = 0.25 if par == "oil" else b
        add(f"FGDPR{tag}.{geo}", f"Real GDP: {s}, {gname}", gname, "Output (by industry)", "level", par, bb, 0.006, 30, sign=1)
        add(f"FE{tag}.{geo}", f"Employment: {s}, {gname} (Ths.)", gname, "Labour (by industry)", "level", "emp", b * 0.8, 0.006, 300, sign=1)

usb = [("FGDPR.IUSA", "Real GDP, United States", "US economy", "level", "us", 1.0, 0.001, 20000, 1),
       ("FLBR.IUSA", "Unemployment rate, United States (%)", "US economy", "rate", "unemp", 0.8, 0.3, 5.5, -1),
       ("FRFED.IUSA", "Federal funds rate (%)", "US economy", "rate", "policy", 1.05, 0.25, 2.2, -1),
       ("FRGOV10Y.IUSA", "US 10-year Treasury yield (%)", "US economy", "rate", "bond10", 0.9, 0.2, 3.3, -1),
       ("FRMORT30.IUSA", "US 30-year mortgage rate (%)", "US economy", "rate", "bond10", 0.9, 0.25, 5.0, -1),
       ("FCPI.IUSA", "CPI-U, United States", "US economy", "level", "cpi", 1.0, 0.002, 100, 0),
       ("FHPI.IUSA", "Case-Shiller home price index, US", "US economy", "level", "hpi", 0.5, 0.01, 150, 1),
       ("FIP.IUSA", "Industrial production, US", "US economy", "level", "us", 1.5, 0.006, 100, 1),
       ("FSP500.IUSA", "S&P 500 index", "US economy", "level", "tsx", 1.0, 0.02, 2000, 1),
       ("FCSENT.IUSA", "U. Michigan consumer sentiment", "US economy", "level", "tsx", 0.3, 0.04, 85, 1),
       ("FHST.IUSA", "Housing starts, US (Mil., SAAR)", "US economy", "level", "us", 3.0, 0.06, 1.4, 1),
       ("FRETAIL.IUSA", "Retail sales, US", "US economy", "level", "us", 1.1, 0.008, 400, 1),
       ("FET.IUSA", "Nonfarm payrolls, US", "US economy", "level", "us", 0.7, 0.002, 140000, 1),
       ("FBAA.IUSA", "Moody's Baa corporate yield (%)", "US economy", "rate", "bond10", 0.8, 0.3, 5.5, -1),
       ("FDJIA.IUSA", "Dow Jones industrial average", "US economy", "level", "tsx", 0.95, 0.025, 15000, 1)]
for code, desc, theme, kind, parent, beta, noise, base, sign in usb:
    add(code, desc, "United States", theme, kind, parent, beta, noise, base, sign=sign)

# Ontario extras (fiscal, housing detail, etc.)
oe = [("FHSEXIST.ION", "Existing home sales, Ontario (Ths.)", "Housing", "level", "hpi", 1.8, 0.05, 200, 1),
      ("FHPICONDO.ITOR", "Condo apartment price index, Toronto", "Housing", "level", "hpi", 1.1, 0.012, 100, 1),
      ("FHPISFD.ITOR", "Single-detached price index, Toronto", "Housing", "level", "hpi", 1.05, 0.006, 100, 1),
      ("FRENT.ITOR", "Average rent, 2-bedroom, Toronto (C$)", "Housing", "level", "pop", 1.2, 0.006, 1500, 0),
      ("FBPRES.ION", "Residential building permits, Ontario (C$)", "Housing", "level", "hpi", 1.3, 0.07, 1200, 1),
      ("FGOVBAL.ION", "Ontario budget balance (% of GDP)", "Government", "rate", "unemp", -0.6, 0.4, -1.5, 0),
      ("FGOVDEBT.ION", "Ontario net debt (% of GDP)", "Government", "rate", "unemp", 1.5, 1.0, 38, 0),
      ("FWAGE.ION", "Average hourly wages, Ontario", "Income", "level", "inc", 0.5, 0.003, 30, 1),
      ("FMFGSHIP.ION", "Manufacturing shipments, Ontario (C$)", "Output", "level", "us", 1.8, 0.015, 25000, 1),
      ("FAUTOPROD.ION", "Motor vehicle production, Ontario (units)", "Output", "level", "us", 2.5, 0.04, 600, 1),
      ("FEXP.ION", "International exports, Ontario (C$)", "Trade", "level", "us", 1.6, 0.012, 200, 1),
      ("FCONF.ION", "Consumer confidence, Ontario", "Sentiment", "level", "tsx", 0.3, 0.03, 100, 1),
      ("FPOPIMM.ION", "Net international migration, Ontario (Ths.)", "Demographics", "level", "pop", 5.0, 0.03, 130, 1),
      ("FPOP2544.ION", "Population aged 25-44, Ontario (Ths.)", "Demographics", "level", "pop", 1.3, 0.0015, 3500, 1),
      ("FPOP65.ION", "Population aged 65+, Ontario (Ths.)", "Demographics", "level", "pop", 0.6, 0.002, 2000, 0),
      ("FHHFORM.ION", "Household formation, Ontario (Ths.)", "Demographics", "level", "pop", 1.0, 0.002, 5000, 1),
      ("FLBRPART.ION", "Participation rate, Ontario (%)", "Labour", "rate", "unemp", -0.3, 0.25, 65, 1),
      ("FLBRYOUTH.ION", "Youth unemployment rate, Ontario (%)", "Labour", "rate", "unemp", 1.8, 0.6, 13, -1),
      ("FEFT.ION", "Full-time employment, Ontario (Ths.)", "Labour", "level", "emp", 1.1, 0.003, 5200, 1),
      ("FEPT.ION", "Part-time employment, Ontario (Ths.)", "Labour", "level", "emp", 0.6, 0.006, 1300, 0),
      ("FHOURS.ION", "Average weekly hours, Ontario", "Labour", "level", "gdp", 0.3, 0.003, 33, 1),
      ("FCPISHL.ION", "CPI shelter, Ontario", "Prices", "level", "cpi", 0.9, 0.002, 100, 0),
      ("FCPIENE.ION", "CPI energy, Ontario", "Prices", "level", "oil", 0.2, 0.01, 100, 0),
      ("FCPIFOOD.ION", "CPI food, Ontario", "Prices", "level", "cpi", 1.0, 0.003, 100, 0),
      ("FAFFORD.ION", "Housing affordability index, Ontario", "Housing", "rate", "mort5", 4.0, 1.5, 50, 0)]
for code, desc, theme, kind, parent, beta, noise, base, sign in oe:
    add(code, desc, "Ontario", theme, kind, parent, beta, noise, base, sign=sign)

# Pad with miscellaneous low-relevance series (noise-dominated) to reach 400
i = 0
misc_parents = ["gdp", "cpi", "oil", "fx", "us", "tsx", "pop"]
while len(specs) < 400:
    par = misc_parents[i % len(misc_parents)]
    add(f"FMISC{i + 1:03d}.IC", f"Miscellaneous indicator {i + 1} (survey/diffusion index)", C, "Other", "level",
        par, float(rng.uniform(-0.3, 0.3)), float(rng.uniform(0.01, 0.03)), 100, sign=0)
    i += 1

# Map parent -> (series, is_growth)
GROWTH_PARENTS = {"gdp", "inc", "hpi", "cpi", "pop", "emp", "tsx", "oil", "fx", "us"}


def build_macro(core, noise_store):
    out = {}
    for sp in specs:
        p = core[sp["parent"]]
        pm = p[:NH].mean()
        e = noise_store[sp["code"]]
        if sp["kind"] == "level":
            if sp["parent"] in GROWTH_PARENTS:
                gr = pm * (1 - sp["beta"]) + sp["beta"] * p + sp["mu"] + e
            else:   # level-type series driven by a rate/level parent (e.g. starts) -> use its change
                gr = 0.004 + sp["beta"] * 0.01 * np.r_[0, np.diff(p)] + e
            if sp["parent"] == "gdp" and sp["theme"] == "Housing":   # housing starts: cyclical level
                lvl = sp["base"] * np.exp(np.clip(sp["beta"] * 8 * pd.Series(p - pm).rolling(4, min_periods=1).mean().values
                                                  + np.cumsum(e) * 0.3 - 0.08 * (core["mort5"] - core["mort5"][:NH].mean()), -2, 2))
                out[sp["code"]] = lvl
                continue
            out[sp["code"]] = sp["base"] * np.exp(np.cumsum(gr) - gr[0])
        else:
            if sp["parent"] in GROWTH_PARENTS:
                x = sp["base"] + sp["beta"] * (p - pm) * 100 + e
            else:
                x = sp["base"] + sp["beta"] * (p - pm) + e
            out[sp["code"]] = x
    return pd.DataFrame(out, index=Q.astype(str))


# Idiosyncratic noise: same in history for all scenarios, zero in forecast period (smooth Moody's paths)
noise_store = {}
for sp in specs:
    if sp["kind"] == "level":
        e = rng.normal(0, sp["noise"], N)
    else:   # AR(1) noise for rate-type variables
        e = np.zeros(N)
        z = rng.normal(0, sp["noise"], N)
        for k in range(1, N):
            e[k] = 0.7 * e[k - 1] + z[k]
    e[NH:] = e[NH - 1] * (0.7 ** np.arange(1, NF + 1)) if sp["kind"] == "rate" else 0.0
    noise_store[sp["code"]] = e

SCEN = {"baseline": "moodys_baseline.csv", "s1_upside": "moodys_s1_upside.csv",
        "s3_downside": "moodys_s3_downside.csv", "s4_severe": "moodys_s4_severe.csv"}
cores = {}
for s, fn in SCEN.items():
    cores[s] = build_core(s)
    m = build_macro(cores[s], noise_store)
    m.index.name = "date"
    m.round(5).to_csv(os.path.join(OUT, fn))

meta = pd.DataFrame([{k: sp[k] for k in ["code", "desc", "geo", "theme", "transform", "expected_sign"]} for sp in specs])
meta.columns = ["mnemonic", "description", "geography", "theme", "transform", "expected_sign"]
meta.to_csv(os.path.join(OUT, "macro_metadata.csv"), index=False)

# ---------------------------------------------------------------------------
# 4. BANK DATA: Ontario loan and deposit balances (history only)
#    TRUE loan equation (unknown to the pipeline):
#    dlog L_t = 0.0040 + 0.30 dlog L_{t-1} + 0.15 dlog HPI_ON_{t-1} + 0.20 dlog YPD_ON_{t-1}
#               - 0.0025 d(Mort5Y)_{t-2} - 0.0020 d(Unemp_ON)_{t-1}  + COVID effects + e
# ---------------------------------------------------------------------------
c = cores["baseline"]
mac = build_macro(c, noise_store)
hpi_on = np.r_[0, np.diff(np.log(mac["FHPI.ION"].values))]
inc_on = np.r_[0, np.diff(np.log(mac["FYPD.ION"].values))]
dmort = np.r_[0, np.diff(mac["FRMORT5Y.IC"].values)]
du = np.r_[0, np.diff(mac["FLBR.ION"].values)]
dpol = np.r_[0, np.diff(mac["FRBOC.IC"].values)]

y = np.zeros(NH)
d = np.zeros(NH)
y[:3] = 0.012
d[:3] = 0.011
eps = rng.normal(0, 0.0030, NH)
epsd = rng.normal(0, 0.0040, NH)
for k in range(3, NH):
    y[k] = (0.0040 + 0.30 * y[k - 1] + 0.15 * hpi_on[k - 1] + 0.20 * inc_on[k - 1]
            - 0.0025 * dmort[k - 2] - 0.0020 * du[k - 1] + eps[k])
    d[k] = (0.0030 + 0.25 * d[k - 1] + 0.35 * inc_on[k] + 0.20 * y[k] - 0.0015 * dpol[k] + epsd[k])
# COVID effects (payment deferrals / paydowns; deposit surge)
y[qpos("2020Q2")] -= 0.008
y[qpos("2020Q3")] += 0.004
d[qpos("2020Q2")] += 0.040
d[qpos("2020Q3")] += 0.015

L = 52000 * np.exp(np.cumsum(y))
D = 47000 * np.exp(np.cumsum(d))
# Acquisition of a loan portfolio in 2014Q3 (inorganic jump)
acq_q = qpos("2014Q3")
acq_amt = round(0.06 * L[acq_q], 0)
L[acq_q:] = L[acq_q:] * (1 + acq_amt / L[acq_q])
D[acq_q:] = D[acq_q:] * 1.03

bank = pd.DataFrame({"date": Q[:NH].astype(str), "loan_balance": L.round(1), "deposit_balance": D.round(1)})
bank.to_csv(os.path.join(OUT, "bank_data.csv"), index=False)
dep_amt = round(D[acq_q] - D[acq_q] / 1.03, 0)
pd.DataFrame([{"date": "2014Q3", "series": "loan_balance", "amount": acq_amt,
               "description": "Acquired loan portfolio (inorganic)"},
              {"date": "2014Q3", "series": "deposit_balance", "amount": dep_amt,
               "description": "Deposits acquired with the portfolio (inorganic)"}]
             ).to_csv(os.path.join(OUT, "bank_events.csv"), index=False)

print(f"Macro variables: {len(specs)} | history {Q[0]}-{HIST_END} | scenarios to {Q[-1]}")
print(f"Loan balance: {L[0]:,.0f} -> {L[-1]:,.0f}  (avg qoq dlog {y.mean():.4f})")
print("TRUE drivers: FHPI.ION dlog lag1 (+0.15), FYPD.ION dlog lag1 (+0.20), FRMORT5Y.IC diff lag2 (-0.0025), "
      "FLBR.ION diff lag1 (-0.0020), AR(1)=0.30")
print("Files written to", OUT)
