import optimalpolicy._core as rust_helpers
opt_pol = rust_helpers.optimal_policy_par(
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

pol_eval = rust_helpers.policy_evaluation_par_opt(
    2,
    1,
    1,
    1,
    1,
    9,
    0,
    1,
    opt_pol[0],
    p=0.8,
    max_wh=10,
    max_sa=5,
    max_sb=5
)

assert(3.83 == round(min(pol_eval.values()),2))