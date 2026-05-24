"""Wine 품종 분류 — 학습, 평가, 하이퍼파라미터 실험."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

MODEL_PATH = Path(__file__).resolve().parent / "models" / "wine_rf.pkl"
RANDOM_STATE = 42
TEST_SIZE = 0.2

# 베이스라인(1차 커밋)과 동일한 설정
BASELINE_PARAMS = {"n_estimators": 100, "max_depth": None}

# 2차: 하이퍼파라미터 비교 실험 후보
PARAM_CANDIDATES = [
    {"n_estimators": 50, "max_depth": None, "label": "rf_50_trees"},
    {"n_estimators": 100, "max_depth": None, "label": "rf_100_trees_baseline"},
    {"n_estimators": 100, "max_depth": 5, "label": "rf_100_depth5"},
    {"n_estimators": 200, "max_depth": 10, "label": "rf_200_depth10"},
]


def load_split_data():
    wine = load_wine()
    X_train, X_test, y_train, y_test = train_test_split(
        wine.data,
        wine.target,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=wine.target,
    )
    return X_train, X_test, y_train, y_test, wine.target_names


def build_pipeline(n_estimators: int, max_depth: int | None) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def evaluate(
    pipeline: Pipeline,
    X_train,
    X_test,
    y_train,
    y_test,
    target_names,
    title: str,
) -> dict[str, float]:
    y_pred_train = pipeline.predict(X_train)
    y_pred_test = pipeline.predict(X_test)

    metrics = {
        "train_acc": accuracy_score(y_train, y_pred_train),
        "test_acc": accuracy_score(y_test, y_pred_test),
        "test_f1_macro": f1_score(y_test, y_pred_test, average="macro"),
    }

    print(f"\n{'=' * 60}")
    print(title)
    print(f"{'=' * 60}")
    print(
        f"Train accuracy: {metrics['train_acc']:.4f} | "
        f"Test accuracy: {metrics['test_acc']:.4f} | "
        f"Test F1 (macro): {metrics['test_f1_macro']:.4f}"
    )
    print("\n[Confusion matrix — test set]")
    print(confusion_matrix(y_test, y_pred_test))
    print("\n[Classification report — test set]")
    print(classification_report(y_test, y_pred_test, target_names=target_names))

    return metrics


def run_hyperparameter_experiments(
    X_train, X_test, y_train, y_test, target_names
) -> tuple[dict, Pipeline]:
    print("\n" + "#" * 60)
    print("# Hyperparameter comparison (RandomForest)")
    print("#" * 60)

    rows: list[dict] = []
    best_row: dict | None = None
    best_pipeline: Pipeline | None = None

    for spec in PARAM_CANDIDATES:
        pipeline = build_pipeline(spec["n_estimators"], spec["max_depth"])
        pipeline.fit(X_train, y_train)
        metrics = evaluate(
            pipeline,
            X_train,
            X_test,
            y_train,
            y_test,
            target_names,
            title=(
                f"{spec['label']} | n_estimators={spec['n_estimators']}, "
                f"max_depth={spec['max_depth']}"
            ),
        )
        row = {
            "label": spec["label"],
            "n_estimators": spec["n_estimators"],
            "max_depth": spec["max_depth"],
            **metrics,
        }
        rows.append(row)

        if best_row is None or row["test_f1_macro"] > best_row["test_f1_macro"]:
            best_row = row
            best_pipeline = pipeline

    print("\n" + "-" * 60)
    print("Experiment summary (sorted by test F1 macro)")
    print("-" * 60)
    print(
        f"{'label':<28} {'n_est':>6} {'depth':>6} "
        f"{'train_acc':>10} {'test_acc':>10} {'test_f1':>10}"
    )
    for row in sorted(rows, key=lambda r: r["test_f1_macro"], reverse=True):
        depth = "None" if row["max_depth"] is None else str(row["max_depth"])
        print(
            f"{row['label']:<28} {row['n_estimators']:>6} {depth:>6} "
            f"{row['train_acc']:>10.4f} {row['test_acc']:>10.4f} "
            f"{row['test_f1_macro']:>10.4f}"
        )

    assert best_row is not None and best_pipeline is not None
    print(
        f"\nBest model: {best_row['label']} "
        f"(test F1 macro={best_row['test_f1_macro']:.4f})"
    )
    return best_row, best_pipeline


def train_baseline(X_train, X_test, y_train, y_test, target_names) -> Pipeline:
    pipeline = build_pipeline(
        BASELINE_PARAMS["n_estimators"],
        BASELINE_PARAMS["max_depth"],
    )
    pipeline.fit(X_train, y_train)
    evaluate(
        pipeline,
        X_train,
        X_test,
        y_train,
        y_test,
        target_names,
        title="Baseline | n_estimators=100, max_depth=None",
    )
    return pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Wine classification training")
    parser.add_argument(
        "--mode",
        choices=("baseline", "experiment", "full"),
        default="full",
        help="baseline: 1차 설정만 | experiment: 파라미터 비교 | full: 둘 다",
    )
    args = parser.parse_args()

    X_train, X_test, y_train, y_test, target_names = load_split_data()
    pipeline_to_save: Pipeline | None = None

    if args.mode in ("baseline", "full"):
        pipeline_to_save = train_baseline(
            X_train, X_test, y_train, y_test, target_names
        )

    if args.mode in ("experiment", "full"):
        _, best_pipeline = run_hyperparameter_experiments(
            X_train, X_test, y_train, y_test, target_names
        )
        pipeline_to_save = best_pipeline

    if pipeline_to_save is None:
        raise RuntimeError("No model was trained.")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline_to_save, MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
