# Post hoc decision audit

Read summary.json first: replay_matches_original must be true before treating this as an explanation of the original run. A mismatch means the current simulator or inputs differ; do not interpret it as an exact historical replay.

Each decisions.jsonl record links the observation time, stock, action probabilities, action mask, entry priority and resulting fills. The observation precedes execution. Risk and session exits are environmental rules, not necessarily agent SELL decisions. A BUY request can fail due to funding, capacity, availability or session constraints; zero fills alone does not identify which constraint blocked it.

feature_effects.csv measures changes in the chosen action probability when one feature is set to zero, keeping the action mask fixed. Positive probability_drop means the observed value supported that action relative to zero. Gradients measure local sensitivity and depend on feature units. Removing all strategy signals is a separate intervention in the decision records. These are model-behavior diagnostics, not causal explanations of market returns, SHAP values or proofs of why a profitable trade occurred. Setting account or price inputs to zero may produce unrealistic states.

Feature summary covers the first capped explanations in chronological order, including sampled HOLD decisions. It is not an unbiased global importance estimate. The complete replay continues after that cap. Action counts include all stock-timestamps, including unavailable stocks forced to HOLD. Probability expresses model preference, not calibrated likelihood of profit. Entry priority uses total BUY probability, while the chosen action is the largest individual category.

Old three-action checkpoints are supported. New checkpoints must match the installed risk-action mapping. Analysis never updates model weights or searches parameters. Do not use test explanations to tune a model and then call that same period untouched test data.
