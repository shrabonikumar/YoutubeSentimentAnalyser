import joblib
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
warnings.filterwarnings("ignore")

from sklearn.model_selection   import train_test_split, StratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model      import LogisticRegression
from sklearn.svm               import LinearSVC
from sklearn.ensemble          import RandomForestClassifier
from sklearn.naive_bayes       import MultinomialNB
from sklearn.pipeline          import Pipeline
from sklearn.preprocessing     import LabelEncoder
from sklearn.metrics           import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score, ConfusionMatrixDisplay,
)

INPUT_CSV        = "iphone16_sentiment.csv"
OUTPUT_MODEL     = "best_model.pkl"
OUTPUT_VECTORIZER = "tfidf_vectorizer.pkl"
REPORT_PNG       = "training_report.png"

LABEL_COL   = "final_label"
TEXT_COL    = "clean_text"
TEST_SIZE   = 0.2
RANDOM_SEED = 42
CV_FOLDS    = 5
LABEL_ORDER = ["positive", "neutral", "negative"]

COLORS = {
    "positive": "#3B8BD4",
    "neutral":  "#888780",
    "negative": "#E24B4A",
}

# ─────────────────────────────────────────────────────────────────────────────
#  DATA LOADING & VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

def load_data(path: str) -> pd.DataFrame:
    print(f"Loading {path}...")
    df = pd.read_csv(path)

    required = [TEXT_COL, LABEL_COL]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing column: '{col}'. Run sentiment_labeling.py first.")

    df = df.dropna(subset=required)
    df = df[df[TEXT_COL].str.strip().ne("")]
    df[LABEL_COL] = df[LABEL_COL].str.lower().str.strip()
    df = df[df[LABEL_COL].isin(LABEL_ORDER)]
    df.reset_index(drop=True, inplace=True)

    print(f"  {len(df):,} usable samples.")
    print("\n  Label distribution:")
    for label, count in df[LABEL_COL].value_counts().items():
        pct = count / len(df) * 100
        bar = "█" * int(pct / 2)
        print(f"    {label:<12} {count:>5}  ({pct:.1f}%)  {bar}")

    # Warn if dataset is small
    if len(df) < 300:
        print("\n  WARNING: fewer than 300 samples — results may be unreliable.")
        print("  Collect more data or lower TEST_SIZE.")

    return df


# ─────────────────────────────────────────────────────────────────────────────
#  TF-IDF VECTORIZER
#  Converts token strings into numeric feature matrices.
#  Uses unigrams + bigrams (e.g. "not good", "battery life" are single features)
# ─────────────────────────────────────────────────────────────────────────────

def build_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        ngram_range    = (1, 2),      # unigrams + bigrams
        max_features   = 15_000,      # top N features by TF-IDF score
        min_df         = 2,           # ignore terms appearing in < 2 docs
        max_df         = 0.95,        # ignore terms in > 95% of docs (too common)
        sublinear_tf   = True,        # apply log(tf) scaling
        strip_accents  = "unicode",
        token_pattern  = r"\b[a-z][a-z0-9]*\b",
    )


# ─────────────────────────────────────────────────────────────────────────────
#  MODEL DEFINITIONS
# ─────────────────────────────────────────────────────────────────────────────

def get_models() -> dict:
    return {
        "Logistic Regression": LogisticRegression(
            C=1.0,
            max_iter=1000,
            class_weight="balanced",   # handles class imbalance automatically
            random_state=RANDOM_SEED,
            solver="lbfgs",
        ),
        "SVM (LinearSVC)": LinearSVC(
            C=1.0,
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_SEED,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            class_weight="balanced",
            random_state=RANDOM_SEED,
            n_jobs=-1,
        ),
        "Naive Bayes": MultinomialNB(
            alpha=0.1,    # Laplace smoothing
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
#  TRAINING & EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def train_and_evaluate(
    X_train, X_test, y_train, y_test, X_raw_train, X_raw
) -> tuple[dict, dict, object, str]:
    """
    Train all models. Return:
        results     : per-model metrics dict
        reports     : per-model classification reports
        best_model  : fitted Pipeline of the best model
        best_name   : name of the best model
    """
    models    = get_models()
    vectorizer = build_vectorizer()

    X_train_tfidf = vectorizer.fit_transform(X_raw_train)
    X_test_tfidf  = vectorizer.transform(X_test)

    results   = {}
    reports   = {}
    pipelines = {}

    print("\n── Cross-validation (5-fold) ────────────────────────────────────")
    for name, model in models.items():
        # Skip Naive Bayes for LinearSVC (NB needs non-negative; LinearSVC is fine)
        if name == "Naive Bayes":
            # NB needs TF-IDF without sublinear; build a simple inner pipeline
            nb_vec = TfidfVectorizer(
                ngram_range=(1, 2), max_features=15_000,
                min_df=2, max_df=0.95, strip_accents="unicode",
                token_pattern=r"\b[a-z][a-z0-9]*\b",
            )
            pipe = Pipeline([("tfidf", nb_vec), ("clf", model)])
            cv_scores = cross_val_score(
                pipe, X_raw_train, y_train,
                cv=StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED),
                scoring="f1_macro", n_jobs=-1,
            )
            pipe.fit(X_raw_train, y_train)
            y_pred = pipe.predict(X_test)
        else:
            cv_scores = cross_val_score(
                model, X_train_tfidf, y_train,
                cv=StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED),
                scoring="f1_macro", n_jobs=-1,
            )
            model.fit(X_train_tfidf, y_train)
            y_pred = model.predict(X_test_tfidf)
            pipe = None

        acc  = accuracy_score(y_test, y_pred)
        f1   = f1_score(y_test, y_pred, average="macro")
        report = classification_report(
            y_test, y_pred, target_names=LABEL_ORDER, zero_division=0
        )

        results[name]   = {
            "accuracy":  acc,
            "f1_macro":  f1,
            "cv_mean":   cv_scores.mean(),
            "cv_std":    cv_scores.std(),
            "y_pred":    y_pred,
        }
        reports[name]   = report
        pipelines[name] = (pipe, model, vectorizer)

        print(f"  {name:<22}  CV f1={cv_scores.mean():.3f} ± {cv_scores.std():.3f}  "
              f"Test acc={acc:.3f}  Test f1={f1:.3f}")

    # Best model = highest test F1-macro
    best_name = max(results, key=lambda k: results[k]["f1_macro"])
    pipe_obj, model_obj, vec_obj = pipelines[best_name]

    if pipe_obj is not None:
        best_pipeline = pipe_obj
    else:
        # Wrap model + vectorizer into a reusable pipeline for saving
        best_pipeline = Pipeline([
            ("tfidf", vec_obj),
            ("clf",   model_obj),
        ])
        # vectorizer is already fitted; just assigning
        best_pipeline.named_steps["tfidf"] = vec_obj
        best_pipeline.named_steps["clf"]   = model_obj

    return results, reports, best_pipeline, best_name, vec_obj


# ─────────────────────────────────────────────────────────────────────────────
#  FEATURE IMPORTANCE  (top words per class for Logistic Regression)
# ─────────────────────────────────────────────────────────────────────────────

def get_top_features(model, vectorizer, top_n: int = 15) -> dict:
    """
    For Logistic Regression: extract the features (words/bigrams) that
    most strongly push toward each sentiment class.
    """
    if not hasattr(model, "coef_"):
        return {}

    feature_names = vectorizer.get_feature_names_out()
    classes       = model.classes_
    top_features  = {}

    for i, cls in enumerate(classes):
        coefs    = model.coef_[i] if len(model.coef_) > 1 else model.coef_[0]
        top_idx  = np.argsort(coefs)[-top_n:][::-1]
        top_features[cls] = [(feature_names[j], round(coefs[j], 3)) for j in top_idx]

    return top_features


# ─────────────────────────────────────────────────────────────────────────────
#  VISUALIZATIONS
# ─────────────────────────────────────────────────────────────────────────────

def plot_results(results: dict, reports: dict,
                 best_name: str, y_test, best_model,
                 vectorizer, df: pd.DataFrame) -> None:

    fig = plt.figure(figsize=(18, 14))
    fig.suptitle("iPhone 16 Sentiment — Model Training Report",
                 fontsize=15, fontweight="bold", y=0.98)
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    model_names  = list(results.keys())
    accuracies   = [results[m]["accuracy"]  for m in model_names]
    f1_scores    = [results[m]["f1_macro"]  for m in model_names]
    cv_means     = [results[m]["cv_mean"]   for m in model_names]
    cv_stds      = [results[m]["cv_std"]    for m in model_names]
    bar_colors   = ["#185FA5" if m == best_name else "#B5D4F4" for m in model_names]

    # ── 1. Accuracy comparison ───────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    bars = ax1.barh(model_names, accuracies, color=bar_colors, edgecolor="white")
    for bar, val in zip(bars, accuracies):
        ax1.text(val + 0.005, bar.get_y() + bar.get_height()/2,
                 f"{val:.3f}", va="center", fontsize=9)
    ax1.set_xlim(0, 1.1)
    ax1.set_xlabel("Accuracy")
    ax1.set_title("Test accuracy", fontweight="bold")
    ax1.axvline(0.8, color="gray", linestyle="--", linewidth=0.7, alpha=0.5)

    # ── 2. F1-macro comparison ───────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    bars = ax2.barh(model_names, f1_scores, color=bar_colors, edgecolor="white")
    for bar, val in zip(bars, f1_scores):
        ax2.text(val + 0.005, bar.get_y() + bar.get_height()/2,
                 f"{val:.3f}", va="center", fontsize=9)
    ax2.set_xlim(0, 1.1)
    ax2.set_xlabel("F1-macro")
    ax2.set_title("Test F1-macro", fontweight="bold")

    # ── 3. Cross-validation scores ───────────────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.barh(model_names, cv_means, xerr=cv_stds,
             color=bar_colors, edgecolor="white",
             error_kw={"ecolor": "gray", "capsize": 4})
    ax3.set_xlim(0, 1.1)
    ax3.set_xlabel("CV F1-macro (mean ± std)")
    ax3.set_title(f"{CV_FOLDS}-fold CV score", fontweight="bold")

    # ── 4. Confusion matrix for best model ──────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 0:2])
    y_pred_best = results[best_name]["y_pred"]
    cm = confusion_matrix(y_test, y_pred_best, labels=LABEL_ORDER)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=LABEL_ORDER)
    disp.plot(ax=ax4, cmap="Blues", colorbar=False)
    ax4.set_title(f"Confusion matrix — {best_name}  (best model)", fontweight="bold")

    # ── 5. Class-level F1 for best model ────────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 2])
    report_dict = classification_report(
        y_test, y_pred_best, target_names=LABEL_ORDER,
        output_dict=True, zero_division=0,
    )
    per_class_f1  = [report_dict[l]["f1-score"]  for l in LABEL_ORDER]
    per_class_prec= [report_dict[l]["precision"] for l in LABEL_ORDER]
    per_class_rec = [report_dict[l]["recall"]    for l in LABEL_ORDER]
    x = np.arange(len(LABEL_ORDER))
    w = 0.25
    ax5.bar(x - w, per_class_prec, w, label="Precision", color="#185FA5")
    ax5.bar(x,     per_class_f1,   w, label="F1",        color="#3B8BD4")
    ax5.bar(x + w, per_class_rec,  w, label="Recall",    color="#85B7EB")
    ax5.set_xticks(x)
    ax5.set_xticklabels(LABEL_ORDER)
    ax5.set_ylim(0, 1.15)
    ax5.legend(fontsize=8)
    ax5.set_title("Per-class metrics (best model)", fontweight="bold")

    # ── 6. Top words per class (Logistic Regression) ─────────────────────────
    lr_model = None
    for name in model_names:
        if "Logistic" in name:
            lr_model = results[name]
            break

    top_feats = {}
    if hasattr(best_model.named_steps.get("clf", None), "coef_"):
        top_feats = get_top_features(
            best_model.named_steps["clf"],
            best_model.named_steps["tfidf"],
        )

    for col_i, label in enumerate(LABEL_ORDER):
        ax = fig.add_subplot(gs[2, col_i])
        if label in top_feats and top_feats[label]:
            feats  = top_feats[label][:12]
            words  = [f[0] for f in feats]
            scores = [f[1] for f in feats]
            ax.barh(words[::-1], scores[::-1], color=COLORS[label], edgecolor="white")
            ax.set_xlabel("Coefficient")
        else:
            ax.text(0.5, 0.5, "N/A for this model",
                    ha="center", va="center", transform=ax.transAxes, color="gray")
        ax.set_title(f"Top '{label}' features", fontweight="bold")

    plt.savefig(REPORT_PNG, dpi=150, bbox_inches="tight")
    print(f"\n  Report saved → {REPORT_PNG}")
    plt.show()


def predict_new(texts: list[str], model_path: str = OUTPUT_MODEL) -> pd.DataFrame:
    """
    Quick helper to classify new comments with the saved model.
    Usage:
        from model_training import predict_new
        results = predict_new(["Battery drains so fast", "Camera is absolutely stunning"])
    """
    model = joblib.load(model_path)
    preds = model.predict(texts)
    try:
        probs  = model.predict_proba(texts)
        confs  = probs.max(axis=1).round(3)
    except AttributeError:
        confs = [None] * len(preds)     # LinearSVC has no predict_proba

    return pd.DataFrame({
        "text":       texts,
        "prediction": preds,
        "confidence": confs,
    })


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    df = load_data(INPUT_CSV)

    X = df[TEXT_COL].fillna("").astype(str)
    y = df[LABEL_COL]

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        stratify=y,                   # keep class balance in both splits
        random_state=RANDOM_SEED,
    )

    print(f"\n  Train : {len(X_train_raw):,} samples")
    print(f"  Test  : {len(X_test_raw):,} samples")

    results, reports, best_pipeline, best_name, vectorizer = train_and_evaluate(
        X_train_raw, X_test_raw, y_train, y_test,
        X_train_raw, X,
    )

    # ── Print full classification reports ────────────────────────────────────
    print("\n── Classification reports ───────────────────────────────────────")
    for name, report in reports.items():
        marker = " ← BEST" if name == best_name else ""
        print(f"\n  {name}{marker}")
        for line in report.strip().split("\n"):
            print(f"    {line}")

    # ── Save best model ───────────────────────────────────────────────────────
    joblib.dump(best_pipeline, OUTPUT_MODEL)
    joblib.dump(vectorizer,    OUTPUT_VECTORIZER)
    print(f"\n  Best model : {best_name}")
    print(f"  Saved      → {OUTPUT_MODEL}")
    print(f"  Vectorizer → {OUTPUT_VECTORIZER}")

    # ── Add predictions to the full dataset and save ──────────────────────────
    df["predicted_label"]      = best_pipeline.predict(X)
    try:
        probs                  = best_pipeline.predict_proba(X)
        df["pred_confidence"]  = probs.max(axis=1).round(3)
    except AttributeError:
        df["pred_confidence"]  = None

    df.to_csv("iphone16_modeled.csv", index=False, encoding="utf-8-sig")
    print(f"  Full predictions saved → iphone16_modeled.csv")

    # ── Visualize ─────────────────────────────────────────────────────────────
    plot_results(results, reports, best_name, y_test,
                 best_pipeline, vectorizer, df)

if __name__ == "__main__":
    main()