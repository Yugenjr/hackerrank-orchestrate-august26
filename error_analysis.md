# Error Analysis (Mode C)

## sample_msg_042
- Ground Truth: notify (urgent)
- Predicted: digest (personal)
- Oracle Primary Cause: Retrieval

## sample_msg_043
- Ground Truth: mute (spam)
- Predicted: digest (personal)
- Oracle Primary Cause: Retrieval

## sample_msg_045
- Ground Truth: mute (promotion)
- Predicted: digest (promotion)
- Oracle Primary Cause: Retrieval

## sample_msg_047
- Ground Truth: mute (promotion)
- Predicted: digest (promotion)
- Oracle Primary Cause: Retrieval

## sample_msg_048
- Ground Truth: digest (business_update)
- Predicted: notify (payment_reminder)
- Oracle Primary Cause: Retrieval



## Pareto Chart of Failure Causes

| Cause                    |   Count |   Cumulative % |
|:-------------------------|--------:|---------------:|
| Retrieval Fallback       |       4 |        44.4444 |
| Deterministic Rules Miss |       3 |        77.7778 |
| LLM Generalization       |       1 |        88.8889 |
| Safety False Positive    |       1 |       100      |