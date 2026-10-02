# Remaining useful life (RUL)

AeroMind reports RUL as three flight-hour quantiles: **p10 < p50 < p90**.

- p50: the median estimate. Half of the estimated failure-time distribution lies before it.
- p10: a conservative estimate. The model says about a 10% chance of failure before this time.
- p90: an optimistic estimate.

The band p10–p90 is a nominal 80% interval. On the simulator the gradient-boosting intervals
under-cover unless conformally calibrated; on NASA C-MAPSS the coverage is roughly 0.70–0.82
depending on subset and model. An RUL value is an **estimate with uncertainty, not a guarantee**.

RUL is estimated only while the persistence gate is open. With no open gate there is no prediction.

Models: quantile gradient-boosted trees (default) or an optional PyTorch LSTM, with optional
conformal calibration of the interval. The decision engine turns the three quantiles into a
piecewise-linear failure-time distribution to compute the probability of failure before the next
check; this is a modelling convenience, not a validated reliability model.
