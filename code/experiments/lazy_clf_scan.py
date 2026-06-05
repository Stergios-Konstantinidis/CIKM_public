"""Manual classifier scan — all sklearn classifiers, no XGBoost/LightGBM."""
import sys
import warnings
import numpy as np
from pathlib import Path
from sklearn.model_selection import cross_val_predict, KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import RidgeCV
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# All sklearn classifiers worth trying
from sklearn.ensemble import (
    AdaBoostClassifier, RandomForestClassifier, ExtraTreesClassifier,
    GradientBoostingClassifier, HistGradientBoostingClassifier,
    BaggingClassifier, StackingClassifier,
)
from sklearn.svm import SVC, NuSVC, LinearSVC
from sklearn.linear_model import (
    LogisticRegression, RidgeClassifier, SGDClassifier, Perceptron,
    PassiveAggressiveClassifier,
)
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis,
)
from sklearn.dummy import DummyClassifier
from sklearn.calibration import CalibratedClassifierCV

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments.experiment_gbt_classifier import (
    load_tesseract_records, load_llm_corrections, build_features
)

warnings.filterwarnings('ignore')

print("Loading data...")
records = load_tesseract_records()
target_file = "corrections/tesseract/tesseract_Full_Advanced_5_google__gemini-3-flash-preview.json"
corrections = load_llm_corrections(target_file)
X = build_features(records)

# Compute ground-truth deltas
base_wer = np.array([float(r["wer"]) for r in records])
corr_wer = np.array([corrections.get(r["filename"], {}).get("wer", float(r["wer"])) for r in records])
delta_wer = base_wer - corr_wer
base_cer = np.array([float(r["cer"]) for r in records])
corr_cer = np.array([corrections.get(r["filename"], {}).get("cer", float(r["cer"])) for r in records])
delta_cer = base_cer - corr_cer

# Train delta regressions (10-fold CV)
print("Training delta regressions...")
cv = KFold(n_splits=10, shuffle=True, random_state=42)
ridge_pipe = Pipeline([('scaler', StandardScaler()), ('ridge', RidgeCV(alphas=np.logspace(-3, 3, 50)))])
pred_ridge_delta_wer = cross_val_predict(ridge_pipe, X, delta_wer, cv=cv)
pred_ridge_delta_cer = cross_val_predict(ridge_pipe, X, delta_cer, cv=cv)
mlp_pipe = Pipeline([('scaler', StandardScaler()), ('mlp', MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=500, early_stopping=True, random_state=42, verbose=False))])
pred_mlp_delta_wer = cross_val_predict(mlp_pipe, X, delta_wer, cv=cv)
pred_mlp_delta_cer = cross_val_predict(mlp_pipe, X, delta_cer, cv=cv)

# Stack features
X_stacked = np.column_stack([X, pred_ridge_delta_wer, pred_ridge_delta_cer, pred_mlp_delta_wer, pred_mlp_delta_cer])
print(f"Feature matrix: {X_stacked.shape}")

# Binary labels
y = (delta_wer > 0.0).astype(int)
print(f"Class distribution: neg={np.sum(y==0)}, pos={np.sum(y==1)} ({np.mean(y==1):.1%} positive)")

# Evaluate using actual routing metric (average WER after routing)
corr_vals = corr_wer
base_vals = base_wer

all_models = {
    # Ensemble
    "HistGBT": HistGradientBoostingClassifier(max_iter=500, max_depth=4, learning_rate=0.03, min_samples_leaf=5, random_state=42, class_weight='balanced'),
    "GBT": GradientBoostingClassifier(n_estimators=500, max_depth=3, learning_rate=0.02, subsample=0.8, min_samples_leaf=4, random_state=42),
    "AdaBoost": AdaBoostClassifier(n_estimators=300, learning_rate=0.03, random_state=42),
    "ExtraTrees": ExtraTreesClassifier(n_estimators=500, max_depth=10, min_samples_leaf=3, class_weight='balanced', random_state=42),
    "RandomForest": RandomForestClassifier(n_estimators=500, max_depth=8, min_samples_leaf=3, class_weight='balanced', random_state=42),
    "Bagging": BaggingClassifier(n_estimators=200, random_state=42),
    # SVM
    "NuSVC": NuSVC(probability=True, nu=0.4, kernel='rbf', random_state=42),
    "SVC-RBF": SVC(kernel='rbf', C=1.0, class_weight='balanced', random_state=42),
    "LinearSVC": CalibratedClassifierCV(LinearSVC(C=1.0, class_weight='balanced', max_iter=5000, random_state=42)),
    # Linear
    "LogReg": LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000, random_state=42),
    "RidgeClf": RidgeClassifier(class_weight='balanced', alpha=1.0),
    "SGD": SGDClassifier(class_weight='balanced', max_iter=1000, random_state=42),
    "PassiveAggressive": PassiveAggressiveClassifier(class_weight='balanced', max_iter=1000, random_state=42),
    # Neural
    "MLP-Clf": MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, early_stopping=True, random_state=42),
    "MLP-Large": MLPClassifier(hidden_layer_sizes=(256, 128, 64), max_iter=500, early_stopping=True, random_state=42),
    # Neighbors
    "KNN-5": KNeighborsClassifier(n_neighbors=5),
    "KNN-10": KNeighborsClassifier(n_neighbors=10),
    "KNN-20": KNeighborsClassifier(n_neighbors=20),
    # Trees
    "DecisionTree": DecisionTreeClassifier(max_depth=8, class_weight='balanced', random_state=42),
    "ExtraTree": ExtraTreeClassifier(max_depth=8, class_weight='balanced', random_state=42),
    # Probabilistic
    "NaiveBayes": GaussianNB(),
    "BernoulliNB": BernoulliNB(),
    # Discriminant
    "LDA": LinearDiscriminantAnalysis(),
    "QDA": QuadraticDiscriminantAnalysis(),
    # Dummy
    "Random": DummyClassifier(strategy='stratified', random_state=42),
}

print(f"\nEvaluating {len(all_models)} classifiers via 10-fold CV...")
print(f"{'Model':<22} {'Acc':>6} {'F1':>6} {'%Routed':>8} {'Avg WER':>8}  {'ΔvsBase':>8}")
print("-" * 68)

results = []
for name, clf in all_models.items():
    try:
        pipeline = Pipeline([('scaler', StandardScaler()), ('clf', clf)])
        y_pred = cross_val_predict(pipeline, X_stacked, y, cv=cv, n_jobs=-1)
        mask = y_pred == 1
        pct_routed = mask.mean() * 100
        final_vals = np.where(mask, corr_vals, base_vals)
        final_avg = np.mean(final_vals)
        acc = accuracy_score(y, y_pred)
        f1 = f1_score(y, y_pred)
        delta_vs_base = final_avg - np.mean(base_vals)
        results.append((name, acc, f1, pct_routed, final_avg, delta_vs_base))
        print(f"{name:<22} {acc:>6.3f} {f1:>6.3f} {pct_routed:>7.1f}% {final_avg:>8.4f}  {delta_vs_base:>+8.4f}")
    except Exception as e:
        print(f"{name:<22} FAILED: {e}")

# Sort by final WER (lower is better)
print("\n" + "="*68)
print("RANKED BY ROUTING WER (lower = better):")
print("="*68)
results.sort(key=lambda x: x[4])
for i, (name, acc, f1, pct, wer, d) in enumerate(results, 1):
    marker = " ★" if i <= 8 else ""
    print(f"{i:>2}. {name:<22} WER={wer:.4f}  routed={pct:5.1f}%  acc={acc:.3f}  f1={f1:.3f}{marker}")
