RISK_PAIRS = [(sl, round(sl * rr, 12)) for sl in (.005, .0075, .01) for rr in (1., 2., 3.)]
BUY_ACTIONS = [1, *range(3, 11)]
ACTION_RISK = dict(zip(BUY_ACTIONS, RISK_PAIRS))
ACTION_COUNT = 11
