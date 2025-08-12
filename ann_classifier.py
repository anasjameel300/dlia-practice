import time
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from tqdm import tqdm
import matplotlib.pyplot as plt
DEFAULT_DATASET_FILENAME = "airlines_flights_data.csv"  # Change this one line to use a different CSV



def load_dataset(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found at: {csv_path}")
    df = pd.read_csv(csv_path)
    # Drop auto index column if present
    if "index" in df.columns:
        df = df.drop(columns=["index"])  # created by some Kaggle exports
    return df


def identify_feature_types(df: pd.DataFrame, target_column: str) -> Tuple[List[str], List[str]]:
    feature_cols = [c for c in df.columns if c != target_column]
    categorical_cols: List[str] = []
    numeric_cols: List[str] = []
    for col in feature_cols:
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols.append(col)
        else:
            categorical_cols.append(col)
    return categorical_cols, numeric_cols


def choose_targets(df: pd.DataFrame) -> Tuple[str, str]:
    # Prefer explicit binary -> "class" if exactly two unique values
    candidate_binary_cols = [
        col for col in df.columns if df[col].nunique(dropna=True) == 2
    ]
    binary_target: Optional[str] = None
    if "class" in df.columns and df["class"].nunique(dropna=True) == 2:
        binary_target = "class"
    elif len(candidate_binary_cols) > 0:
        # pick a stable option
        binary_target = sorted(candidate_binary_cols)[0]

    # If no suitable binary label, derive one from price median
    if binary_target is None:
        if "price" in df.columns and pd.api.types.is_numeric_dtype(df["price"]):
            median_price = df["price"].median()
            derived = (df["price"] >= median_price).astype(int)
            df["high_price"] = derived
            binary_target = "high_price"
        else:
            # Fallback: use the least frequent nominal column and binarize top-1 vs rest
            nominal_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
            if not nominal_cols:
                raise ValueError("Could not find suitable column to derive binary target.")
            col = nominal_cols[0]
            top_class = df[col].value_counts().idxmax()
            df["top_vs_rest"] = (df[col] == top_class).astype(int)
            binary_target = "top_vs_rest"

    # Prefer explicit multiclass -> "airline" if > 2 classes, else any with > 2
    multiclass_target: Optional[str] = None
    if "airline" in df.columns and df["airline"].nunique(dropna=True) > 2:
        multiclass_target = "airline"
    else:
        candidates_multi = [c for c in df.columns if df[c].nunique(dropna=True) > 2]
        # Avoid using price (regression) if possible
        candidates_multi = [c for c in candidates_multi if c != "price"]
        if len(candidates_multi) == 0:
            # If still none, discretize price into quantiles
            if "price" in df.columns and pd.api.types.is_numeric_dtype(df["price"]):
                df["price_bucket"] = pd.qcut(df["price"], q=4, labels=["Q1", "Q2", "Q3", "Q4"])
                multiclass_target = "price_bucket"
            else:
                # Last resort: reuse binary column but warn via name
                multiclass_target = binary_target
        else:
            multiclass_target = sorted(candidates_multi)[0]

    return binary_target, multiclass_target


def build_preprocessor(categorical_cols: List[str], numeric_cols: List[str]) -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )
    return preprocessor


def run_experiments(
    df: pd.DataFrame,
    target_column: str,
    hidden_sizes: List[int],
    learning_rates: List[float],
    random_state: int = 42,
    task_name: str = "",
    show_progress: bool = True,
    epoch_verbose: bool = False,
) -> pd.DataFrame:
    categorical_cols, numeric_cols = identify_feature_types(df, target_column)

    X = df.drop(columns=[target_column])
    y = df[target_column].astype("category").cat.codes  # robust encoding for labels

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    preprocessor = build_preprocessor(categorical_cols, numeric_cols)

    results: List[Dict] = []

    total_configs = len(hidden_sizes) * len(learning_rates)
    progress = tqdm(total=total_configs, desc=f"{task_name} training", disable=not show_progress)

    for hidden in hidden_sizes:
        for lr in learning_rates:
            model = MLPClassifier(
                hidden_layer_sizes=(hidden,),  # 3-layer network: input -> hidden -> output
                activation="relu",
                solver="adam",
                learning_rate_init=lr,
                max_iter=200,
                early_stopping=True,
                n_iter_no_change=10,
                random_state=random_state,
                verbose=epoch_verbose,
            )

            pipeline = Pipeline(steps=[("prep", preprocessor), ("clf", model)])

            start = time.perf_counter()
            pipeline.fit(X_train, y_train)
            train_time_s = time.perf_counter() - start

            y_pred = pipeline.predict(X_test)
            acc = accuracy_score(y_test, y_pred)

            # Estimate number of parameters for reference (not exact after one-hot until fit)
            # We approximate input size after preprocessing using the fitted onehot categories
            prep = pipeline.named_steps["prep"]  # already fitted in pipeline.fit above
            num_features_numeric = len(prep.transformers_[0][2])
            onehot = prep.named_transformers_["cat"].named_steps["onehot"]
            onehot_feature_count = int(sum(len(cats) for cats in onehot.categories_)) if hasattr(onehot, "categories_") else 0
            approx_input_dim = num_features_numeric + onehot_feature_count
            n_classes = int(len(np.unique(y)))
            hidden_units = int(hidden)
            approx_params = (
                approx_input_dim * hidden_units
                + hidden_units  # hidden biases
                + hidden_units * n_classes
                + n_classes  # output biases
            )

            # Epochs actually used and an approximate training FLOPs estimate
            model_fitted: MLPClassifier = pipeline.named_steps["clf"]
            epochs = int(getattr(model_fitted, "n_iter_", 0))
            num_train_samples = int(len(y_train))
            # Rough FLOPs estimate (forward + backward) per epoch ~ 2 * (input*hidden + hidden*classes) per sample
            approx_flops = (
                2.0 * (approx_input_dim * hidden_units + hidden_units * n_classes)
            ) * float(num_train_samples) * float(max(epochs, 1))

            if show_progress:
                progress.set_postfix(
                    hidden=hidden,
                    lr=f"{lr:g}",
                    acc=f"{acc:.3f}",
                    secs=f"{train_time_s:.1f}",
                    epochs=epochs,
                )
                progress.update(1)

            results.append(
                {
                    "target": target_column,
                    "hidden": hidden,
                    "learning_rate": lr,
                    "accuracy": acc,
                    "train_time_s": train_time_s,
                    "epochs": epochs,
                    "approx_params": approx_params,
                    "approx_flops": int(approx_flops),
                    "num_classes": n_classes,
                    "approx_input_dim": approx_input_dim,
                }
            )

    progress.close()
    return pd.DataFrame(results)


def main() -> None:
    np.random.seed(42)

    parser = argparse.ArgumentParser(description="3-layer ANN experiments on airline flights data")
    parser.add_argument("--csv", type=str, default=DEFAULT_DATASET_FILENAME, help="Path to CSV dataset")
    parser.add_argument("--sample", type=int, default=0, help="Optional number of rows to sample for quick runs (0 = use all)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling and splitting")
    parser.add_argument("--epoch-verbose", action="store_true", help="Show per-iteration loss logs from the MLP training")
    parser.add_argument("--no-plots", action="store_true", help="Disable charts at the end")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    csv_path = Path(args.csv)
    if not csv_path.is_absolute():
        csv_path = project_root / csv_path
    df = load_dataset(csv_path)

    if args.sample and args.sample > 0 and args.sample < len(df):
        df = df.sample(n=args.sample, random_state=args.seed).reset_index(drop=True)

    # Pick targets for binary and multiclass tasks
    binary_target, multiclass_target = choose_targets(df)

    # Required experiments
    hidden_small = [32]  # < 50
    hidden_large = [128]  # > 100
    learning_rates = [0.0001, 0.001, 0.01]
        
    print("Running Binary Classification (3-layer ANN)...")
    results_binary_small = run_experiments(
        df.copy(), binary_target, hidden_small, learning_rates, random_state=args.seed, task_name="Binary (small)", show_progress=True, epoch_verbose=args.epoch_verbose
    )
    results_binary_large = run_experiments(
        df.copy(), binary_target, hidden_large, learning_rates, random_state=args.seed, task_name="Binary (large)", show_progress=True, epoch_verbose=args.epoch_verbose
    )
    results_binary = pd.concat([results_binary_small, results_binary_large], ignore_index=True)

    print("Running Multiclass Classification (3-layer ANN)...")
    results_multi_small = run_experiments(
        df.copy(), multiclass_target, hidden_small, learning_rates, random_state=args.seed, task_name="Multiclass (small)", show_progress=True, epoch_verbose=args.epoch_verbose
    )
    results_multi_large = run_experiments(
        df.copy(), multiclass_target, hidden_large, learning_rates, random_state=args.seed, task_name="Multiclass (large)", show_progress=True, epoch_verbose=args.epoch_verbose
    )
    results_multi = pd.concat([results_multi_small, results_multi_large], ignore_index=True)

    all_results = pd.concat([results_binary, results_multi], ignore_index=True)

    # Save results
    out_csv = project_root / "ann_results.csv"
    all_results.to_csv(out_csv, index=False)

    def fmt_rows(df_res: pd.DataFrame) -> str:
        # Compact table string
        df_print = df_res.copy()
        df_print["learning_rate"] = df_print["learning_rate"].map(lambda x: f"{x:g}")
        df_print["accuracy"] = df_print["accuracy"].map(lambda x: f"{x:.4f}")
        df_print["train_time_s"] = df_print["train_time_s"].map(lambda x: f"{x:.2f}")
        df_print["approx_flops"] = df_print["approx_flops"].map(lambda x: f"{int(x):,}")
        df_print = df_print[
            [
                "target",
                "hidden",
                "learning_rate",
                "accuracy",
                "train_time_s",
                "epochs",
                "approx_params",
                "approx_flops",
                "num_classes",
                "approx_input_dim",
            ]
        ]
        return df_print.to_string(index=False)

    print("\nBinary task results (saved to ann_results.csv):")
    print(fmt_rows(results_binary))
    print("\nMulticlass task results (saved to ann_results.csv):")
    print(fmt_rows(results_multi))
    print(f"\nAll results saved to: {out_csv}")

    if not args.no_plots:
        # Charts: accuracy and training time per config for each task
        def make_plots(df_task: pd.DataFrame, task_label: str) -> None:
            # Sort for consistent plotting
            df_task = df_task.copy().sort_values(["hidden", "learning_rate"]) 

            # Accuracy plot
            plt.figure(figsize=(8, 4))
            labels = [f"h{h}\nlr={lr:g}" for h, lr in zip(df_task["hidden"], df_task["learning_rate"]) ]
            plt.bar(range(len(df_task)), df_task["accuracy"], color="#4C72B0")
            plt.xticks(range(len(df_task)), labels, rotation=0)
            plt.ylabel("Accuracy")
            plt.title(f"{task_label}: Accuracy by Config")
            plt.tight_layout()
            acc_path = project_root / f"ann_{task_label.lower().replace(' ', '_')}_accuracy.png"
            plt.savefig(acc_path, dpi=200)

            # Training time plot
            plt.figure(figsize=(8, 4))
            plt.bar(range(len(df_task)), df_task["train_time_s"], color="#55A868")
            plt.xticks(range(len(df_task)), labels, rotation=0)
            plt.ylabel("Train Time (s)")
            plt.title(f"{task_label}: Train Time by Config")
            plt.tight_layout()
            time_path = project_root / f"ann_{task_label.lower().replace(' ', '_')}_time.png"
            plt.savefig(time_path, dpi=200)

        make_plots(results_binary, "Binary")
        make_plots(results_multi, "Multiclass")
        # Show all figures at the end
        plt.show()


if __name__ == "__main__":
    main()


