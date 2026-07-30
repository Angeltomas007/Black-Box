from __future__ import annotations

from blackbox.backtest.backtester import run_backtest, walk_forward_windows


def test_transaction_costs_reduce_net_return(random_walk_df):
    zero_cost = run_backtest(random_walk_df, commission_bps=0.0, slippage_bps=0.0)
    with_cost = run_backtest(random_walk_df, commission_bps=5.0, slippage_bps=5.0)
    assert with_cost.equity_curve.iloc[-1] <= zero_cost.equity_curve.iloc[-1]


def test_backtest_equity_curve_starts_at_capital(random_walk_df):
    result = run_backtest(random_walk_df, starting_capital=500_000.0)
    assert result.equity_curve.iloc[0] == 500_000.0


def test_backtest_reports_expected_stat_keys(random_walk_df):
    result = run_backtest(random_walk_df)
    for value in (result.sharpe, result.sortino, result.max_drawdown, result.cagr, result.turnover):
        assert value == value  # not NaN (NaN != NaN)


def test_walk_forward_windows_never_overlap_train_into_test():
    windows = walk_forward_windows(n_obs=1000, train_size=200, test_size=100)
    assert len(windows) > 0
    for train_slice, test_slice in windows:
        assert train_slice.stop <= test_slice.start


def test_walk_forward_windows_expand_over_time():
    windows = walk_forward_windows(n_obs=1000, train_size=200, test_size=100)
    train_sizes = [w[0].stop - w[0].start for w in windows]
    assert train_sizes == sorted(train_sizes)
