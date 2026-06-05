import sys
from pathlib import Path
from sklearn.model_selection import train_test_split
from lazypredict.Supervised import LazyClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments.experiment_gbt_classifier import (
    load_tesseract_records,
    load_llm_corrections,
    build_features,
    build_labels
)

def main():
    print("\n" + "=" * 65)
    print("LazyPredict — Finding the Best Model for Selective Routing")
    print("=" * 65)

    records = load_tesseract_records()
    target_file = "corrections/tesseract/tesseract_Full_Advanced_5_google__gemini-3-flash-preview.json"
    corrections = load_llm_corrections(target_file)

    print("\n  Building features …")
    X = build_features(records)
    
    # We will use the WER delta > 0 as our standard test case
    metric = "wer"
    min_delta = 0.00
    print(f"\n  Generating labels for {metric.upper()} with delta > {min_delta} …")
    y, _, _ = build_labels(records, corrections, min_delta, metric)

    # Lazypredict requires train/test split
    # Since we have ~600 records, we'll use a 75/25 split (447 train, 150 test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    
    print(f"  Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    print("\n  Running LazyClassifier … (this might take a minute)")
    
    # Initialize and fit
    clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
    models, predictions = clf.fit(X_train, X_test, y_train, y_test)
    
    print("\n" + "=" * 80)
    print("  LAZYPREDICT RESULTS RANKING (WER > 0)")
    print("=" * 80)
    print(models)

if __name__ == "__main__":
    main()
