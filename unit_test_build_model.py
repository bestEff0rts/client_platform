#unit_test_build_model.py
from myfunctions import build_model

def test_build_model_returns_correct_type():
    model = build_model("svm")
    from sklearn.svm import SVR
    assert isinstance(model, SVR), "Model is not SVR as expected"

