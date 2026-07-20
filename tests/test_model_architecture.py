from sklearn.base import clone
from sklearn.ensemble import StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression

from services.model_registry import MODEL_IDS, build_model_candidates


EXPECTED_NAMES = (
    "Random Forest",
    "Gradient Boosting",
    "AdaBoost",
    "K-Nearest Neighbors",
    "Naive Bayes",
    "Soft Voting Ensemble",
    "Stacking Ensemble",
)


def test_registry_contains_exactly_seven_models():
    assert tuple(MODEL_IDS.values()) == EXPECTED_NAMES


def test_all_registered_estimators_are_sklearn_cloneable():
    candidates = build_model_candidates(stacking_cv=3)

    for specification in candidates.values():
        assert clone(specification["estimator"]) is not specification["estimator"]


def test_soft_voting_uses_all_five_fresh_estimators():
    candidates = build_model_candidates(stacking_cv=3)
    voting = candidates["soft_voting"]["estimator"]
    assert isinstance(voting, VotingClassifier)
    assert voting.voting == "soft"
    assert [name for name, _ in voting.estimators] == list(MODEL_IDS.keys())[:5]
    for name, estimator in voting.estimators:
        assert estimator is not candidates[name]["estimator"]


def test_stacking_uses_five_models_and_logistic_regression():
    candidates = build_model_candidates(stacking_cv=3)
    stacking = candidates["stacking"]["estimator"]
    assert isinstance(stacking, StackingClassifier)
    assert [name for name, _ in stacking.estimators] == list(MODEL_IDS.keys())[:5]
    assert isinstance(stacking.final_estimator, LogisticRegression)
    assert stacking.stack_method == "predict_proba"
    assert stacking.passthrough is False


def test_all_five_individual_models_train_successfully(trained_bundle):
    individual_results = trained_bundle["result"]["model_results"][:5]
    assert len(individual_results) == 5
    assert all(result["status"] == "completed" for result in individual_results)


def test_successful_run_produces_seven_results(trained_bundle):
    from services.database_service import list_model_results

    results = trained_bundle["result"]["model_results"]
    assert len(results) == 7
    assert len(list_model_results(trained_bundle["run_id"])) == 7
    assert all(result["status"] == "completed" for result in results)


def test_training_reports_start_and_completion_for_each_model(trained_bundle):
    events = trained_bundle["progress_events"]
    started = [event for event in events if event[3] == "started"]
    completed = [event for event in events if event[3] == "completed"]

    assert len(started) == 7
    assert len(completed) == 7
    assert [event[0] for event in completed] == list(range(1, 8))
    assert all(event[1] == 7 for event in events)
