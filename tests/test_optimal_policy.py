import optimalpolicy._core as rust_helpers
res = rust_helpers.optimal_policy_par(
    2,
    1,
    1,
    1,
    1,
    9,
    0,
    1,
    p=0.8,
    distribution='P',
    max_wh=10,
    max_sa=5,
    max_sb=5
)

