from src.train import train_and_log
def test_train_runs():
    res = train_and_log(n_estimators=5, max_depth=5)
    assert "run_id" in res
    assert "rmse" in res
