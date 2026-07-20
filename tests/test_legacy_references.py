from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ACTIVE_PATHS = [ROOT / "app.py", ROOT / "services", ROOT / "templates"]


def active_source_text():
    chunks = []
    for path in ACTIVE_PATHS:
        files = [path] if path.is_file() else path.rglob("*")
        for file_path in files:
            if file_path.suffix in {".py", ".html", ".js"}:
                chunks.append(file_path.read_text(encoding="utf-8").lower())
    return "\n".join(chunks)


def test_legacy_three_model_and_old_stacking_references_are_absent():
    text = active_source_text()
    forbidden = (
        "stacking_top3_lr",
        'voting="hard"',
        "voting='hard'",
        "random forest + adaboost",
        "gradient boosting meta-learner",
        "only three models",
        "three-model comparison",
        "services.simulation.",
        "soft_voting",
        "soft voting",
    )
    assert all(value not in text for value in forbidden)
