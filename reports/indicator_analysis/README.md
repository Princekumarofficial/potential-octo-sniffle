# NIFTY 50 indicator analysis

209,888 bars; 61 indicators; 32 compact candidates.
114 feature pairs have |training Spearman correlation| >= 0.95.

Targets: future close-to-close returns over 1, 3 and 12 five-minute bars.
Only targets with the exact elapsed horizon are included; overnight and missing-bar jumps are excluded.
Features are available at the current bar close. Rolling history may span sessions.
Three expanding training windows end at 50%, 65%, and 80% of the data; labels are purged at boundaries.
Redundancy selection uses the first 50% only. Imputation, clipping and scaling use each training fold only.
Fixed ridge penalty: 0.01 on mean squared loss. No hyperparameter tuning.
Skill = 1 - model MSE / zero-return MSE. Positive is better than predicting unchanged price.

```
 horizon_bars                          model  mean_skill  worst_skill  mean_direction_accuracy
            1                 all_indicators   -0.004056    -0.007217                 0.500881
            1                        compact   -0.000667    -0.001070                 0.495553
            1   compact_without_candle_shape   -0.000312    -0.000615                 0.496014
            1    compact_without_oscillators   -0.000720    -0.001204                 0.496682
            1 compact_without_relative_trend   -0.000385    -0.000706                 0.495811
            1        compact_without_returns   -0.000731    -0.001290                 0.493742
            1     compact_without_volatility   -0.000439    -0.000755                 0.495092
            1                 recent_returns    0.000062    -0.000316                 0.498496
            3                 all_indicators   -0.008125    -0.014851                 0.497688
            3                        compact   -0.001770    -0.002735                 0.494306
            3   compact_without_candle_shape   -0.001498    -0.002485                 0.495150
            3    compact_without_oscillators   -0.001913    -0.003072                 0.494174
            3 compact_without_relative_trend   -0.001360    -0.002193                 0.495864
            3        compact_without_returns   -0.001305    -0.002213                 0.493796
            3     compact_without_volatility   -0.001486    -0.002135                 0.495972
            3                 recent_returns   -0.000922    -0.001575                 0.497982
           12                 all_indicators   -0.015936    -0.030545                 0.486757
           12                        compact   -0.001849    -0.008902                 0.498882
           12   compact_without_candle_shape   -0.001474    -0.008556                 0.498222
           12    compact_without_oscillators   -0.001940    -0.009220                 0.498374
           12 compact_without_relative_trend   -0.001815    -0.007336                 0.496070
           12        compact_without_returns   -0.001096    -0.007645                 0.500625
           12     compact_without_volatility   -0.002140    -0.007210                 0.498194
           12                 recent_returns   -0.000607    -0.002021                 0.497187
```

## Interpretation limits
This is exploratory linear-model screening, not proof of usefulness for every model.
Holdouts have now been inspected: confirm any selected subset on new untouched data.
Overlapping multibar targets are dependent; no significance claims or p-values are made.
Rank correlations are univariate associations, not conditional feature importance.
Ablation refits the compact model without an entire family; correlated families can substitute for one another.
No trading costs, trading rules, cross-market inputs, or time-of-day features are evaluated.
Keep original columns until a final model and forecast horizon are selected.