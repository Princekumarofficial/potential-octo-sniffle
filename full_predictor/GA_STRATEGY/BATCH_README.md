# Monthly evaluation across stocks

The objective defaults to `blend`, the original Sharpe/Sortino/drawdown score. Set `--objective calmar` to use CAGR divided by absolute maximum drawdown for grid optimization, GP fitness, and monthly winner selection. CAGR uses actual elapsed calendar time. Undefined Calmar candidates, including zero-drawdown candidates, are skipped. GP retains its 0.01-per-node complexity penalty during evolution. The shared default is `DEFAULT_OBJECTIVE` in `engine/objectives.py`. CSVs record the objective.

Use `--years 5` to evaluate the latest 60 calendar months, `--year 2022` for January through December 2022, or `--start-month 2020-01 --end-month 2024-12` for an explicit range. Earlier bars remain available for the six-month training window. The latest month may be partial. For a dataset ending August 2025, `--years 5` covers September 2020 through August 2025. Add `--end-month 2025-07` to anchor the 60-month window to July instead.

```powershell
python -m GA_STRATEGY.batch --workers 4 --years 5 --objective calmar --output reports/calmar_5years
python -m GA_STRATEGY.batch --workers 4 --year 2022 --output reports/blend_2022
```

From `full_predictor`, run:

```powershell
python -m GA_STRATEGY.batch --output reports/batch_validation
```

The default validates the first five alphabetically sorted Parquet stocks. Specify names with `--stocks RELIANCE ABB ADANIENT ADANIPORTS ADANIGREEN`. Use `--limit 0` to process all available stocks. Processing defaults to one worker. Add `--workers 4` to process up to four stocks concurrently. Each worker loads one stock into memory. Monthly refits within a stock remain chronological, and the parent process writes the combined CSVs.

Each evaluation month uses the preceding six calendar months for fitting. The four grid strategies and the GP winner compete using the same training Sharpe/Sortino/drawdown score. GP uses its existing complexity penalty to evolve a rule, then that winner is compared using the common score. All five candidates are evaluated, and the Selected series follows the candidate chosen from training data. A failed candidate holds cash. If every training score is undefined, Selected holds cash.

By default, every stock uses its entire available history. The first six calendar months supply training data, the seventh month is evaluated, and the window moves forward one month at a time. Each refit uses the immediately preceding six months. All evaluation months are labeled test. For a stock starting in February 2015, the first fit covers February through July and the first test is August 2015. The next fit covers March through August and the next test is September.

Each stock has its own start and end dates. First and last calendar months can be partial. Actual dates are recorded, and comparisons across stocks should account for different coverage. Use `--end-month 2025-07` to cap the endpoint. Use `--test-months 12` to restrict evaluation to the latest 12 months, or also set `--valid-months` to label an initial validation portion of that recent window. Earlier test months become eligible for later fits once those months have ended.

To run the full history of all stocks:

```powershell
python -m GA_STRATEGY.batch --limit 0 --train-months 6 --output reports/batch_full_history
```


Equity compounds continuously across evaluation months. The prior close and position carry into the next month, including the intervening price move. Switching from long to short incurs two units of turnover. The new target decided at the first close earns subsequent returns. Each new rule calculates its signal from that month's start. At the end of the final period, positions remain marked to market with no forced closing fee.

The output directory contains:

- `monthly_results.csv` with each stock, month, candidate, parameters or GP tree, training score, selection flag, actual dates, return, drawdown, costs, exposure, Sharpe, and Sortino. Selected is an additional row tracking the chosen candidate.
- `stock_summary.csv` with compounded results for each stock and strategy, including Selected. CAGR uses actual elapsed calendar time with 365.25 days per year. Calmar is CAGR divided by absolute drawdown. The realized_best column marks the highest realized cumulative return within a stock and is descriptive.
- `stock_rankings.csv` with one Selected result per stock, ranked by CAGR. This represents the approach that made its choices before each evaluation month.
- The ranking label is `Monthly selected strategy`. Its `selection_by_month` column lists the actual winning strategy for each month.
- `plots/STOCK.png` plots all four fixed strategy families, GP, the monthly-selected approach, and buy and hold. The test boundary is marked. Equity compounds across validation and test. Each fixed strategy family also refits its parameters monthly. Corresponding `STOCK_daily_returns.csv` files contain the plotted cumulative-return series.
- `errors.csv` listing stocks that could not complete. Their partial results are discarded.

Returns and exposures are decimal fractions in the CSVs. For example, 0.05 represents 5%. CAGR annualizes the observed growth and can be extreme for short periods. Cross-stock comparisons should inspect the actual dates and coverage. GP settings are available through `--gp-population`, `--gp-generations`, and `--seed`.
