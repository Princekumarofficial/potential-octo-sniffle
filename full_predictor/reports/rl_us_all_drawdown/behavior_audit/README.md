# Earlier checkpoint audit

The saved run made 520 test round trips. All entries were at 09:31; median
holding duration was 384 minutes. Net profit was $4,016.35, gross realized
trading P&L $4,073.38, and fees $57.03. Win rate was 53.85%. September net P&L
was -$17.815, October +$2,324.045, and November +$1,710.12. These are accounting
facts from the saved trades, not a new simulation.

The original prepared.npz is missing. Replaying its saved checkpoint using
rl_us_all_run2's cache and the current simulator produces +3.743565%, not
+4.01635%. Of 520 morning stock entries, 450 match the saved run. This audit
cannot isolate the mismatch to data, strategy preprocessing, or numerical
execution differences. Exact reproducibility remains unresolved.

All feature-removal and random-selection comparisons use that later cache.
Zeroing strategy observations yields +7.16273%; zeroing relative OHLC inputs
yields +1.39036%. Five random daily ten-stock selectors return approximately
+1.68%, +1.43%, +0.20%, -0.53% and -0.83%. Random selectors retain positions
until session liquidation and use the same entry allocation and fees. Five
seeds and one already-inspected test period cannot establish significance.

These observations suggest the earlier policy's low-turnover daily selection
behavior was useful in this period; they do not establish that its strategy
signals improved performance, or that the 4% result generalizes. Do not select
an ablated policy on these test returns and describe it as untouched testing.

opening_decisions.csv contains available stock observations and probabilities
at 09:30. summary.json records comparison results and cache provenance. The
original_equity.csv filename denotes the current baseline replay (3.74%), not
the original saved 4.02% equity series in the parent directory.
