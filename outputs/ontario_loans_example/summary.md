# Loan forecasting run: ontario_loans_example

Config: `/home/claude/loan_model/config/example_config.yaml`  
Target: `loan_balance` (organic quarterly log change). Estimation 2002Q2–2025Q4; backtest origins from 2010Q4, 1–9 quarters ahead.

## Screen first

| Step | Count |
|---|---|
| Candidate variables | 142 |
| With a stationary form | 142 |
| Features tested (form x lag) | 2304 |
| Variables passing the screen | 98 |
| Clusters | 43 |
| Shortlist | 25 |
| Final drivers | 2 |

Final drivers: `FHPISFD.ITOR|dlog|L1`, `FYPD.ION|dlog|L1`

## Elastic Net first

| Step | Count |
|---|---|
| Candidate variables | 142 |
| Features into the Elastic Net | 2304 |
| Selected by Elastic Net (stable) | 6 |
| Groups after correlation check | 2 |
| Final drivers | 2 |

Final drivers: `FHPI.ION|dlog|L1`, `FYPD.ION|dlog|L1`

## Comparison (average balance error over 1–H quarters, dummy quarters excluded)

| model             |   avg_MAPE_exCOVID_pct |   avg_MAPE_all_pct |   avg_bias_exCOVID_pct |   MAPE_h1_exCOVID_pct |   MAPE_h4_exCOVID_pct |   MAPE_h9_exCOVID_pct |   improvement_vs_benchmark_pct | drivers                                |
|:------------------|-----------------------:|-------------------:|-----------------------:|----------------------:|----------------------:|----------------------:|-------------------------------:|:---------------------------------------|
| AR(1) benchmark   |                  1.501 |              1.586 |                 -0.368 |                 0.344 |                 1.299 |                 2.504 |                          0     | (none)                                 |
| Screen first      |                  0.612 |              0.55  |                 -0.429 |                 0.233 |                 0.499 |                 1.039 |                         59.22  | FHPISFD.ITOR|dlog|L1, FYPD.ION|dlog|L1 |
| Elastic Net first |                  0.72  |              0.635 |                 -0.515 |                 0.226 |                 0.616 |                 1.229 |                         51.998 | FHPI.ION|dlog|L1, FYPD.ION|dlog|L1     |

Drivers chosen by both approaches: FYPD.ION|dlog|L1

## Diagnostics

| model             |   n_obs |   adj_R2 |    BIC |   max p-value of drivers |   max VIF |   Ljung-Box p (4) |   Ljung-Box p (8) |   Breusch-Pagan p |   Jarque-Bera p |   Durbin-Watson |   CUSUM p | wrong-sign drivers   |
|:------------------|--------:|---------:|-------:|-------------------------:|----------:|------------------:|------------------:|------------------:|----------------:|----------------:|----------:|:---------------------|
| AR(1) benchmark   |      95 |    0.381 | -735.8 |                      nan |      1    |             0.234 |             0.375 |             0.666 |           0.694 |            2.12 |     0.679 | none                 |
| Screen first      |      95 |    0.675 | -790.1 |                        0 |      1.2  |             0.23  |             0.546 |             0.57  |           0.961 |            2.23 |     0.275 | none                 |
| Elastic Net first |      95 |    0.69  | -794.6 |                        0 |      1.24 |             0.618 |             0.78  |             0.601 |           0.662 |            2.14 |     0.135 | none                 |
