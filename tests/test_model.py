from churnlens.data import FEATURES, make_synthetic_telecom_data
from churnlens.train import build_pipeline


def test_model_scores_probability_for_expected_schema():
    data = make_synthetic_telecom_data(100, 7)
    model = build_pipeline().fit(data[FEATURES], data["churn_label"])
    probability = model.predict_proba(data[FEATURES].head(1))[:, 1][0]
    assert 0 <= probability <= 1


def test_synthetic_data_is_reproducible():
    first = make_synthetic_telecom_data(10, 11)
    second = make_synthetic_telecom_data(10, 11)
    assert first.equals(second)