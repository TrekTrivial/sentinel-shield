import joblib
import sys
import os
import argparse
import pandas as pd
import numpy as np
import time

print("=" * 80)
print("URL MODEL INFERENCE - TEST NEW URLs")
print("=" * 80)

start_time = time.time()

# Get script directory and construct absolute path to models
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))  # Go up to Impelement/
model_path = os.path.join(project_root, 'models', 'url_model_xgboost', 'xgboost_url_model.pkl')
metadata_path = os.path.join(project_root, 'results', 'url_model', 'model_metadata.json')

# Check if model exists
if not os.path.exists(model_path):
    print(f"ERROR: Model not found at {model_path}")
    print(f"DEBUG: Script dir: {script_dir}")
    print(f"DEBUG: Project root: {project_root}")
    print("Please train the model first using: python scripts/stream_B_url/2_train_xgboost.py")
    sys.exit(1)

# Load model
print(f"\n[STEP 1] Loading XGBoost Model...")
load_start = time.time()
try:
    model = joblib.load(model_path)
    print(f"  ✓ Model loaded successfully in {time.time() - load_start:.2f}s")
except Exception as e:
    print(f"ERROR loading model: {e}")
    sys.exit(1)

# Load metadata for optimal threshold
import json
optimal_threshold = 0.5  # Default
num_features = None

if os.path.exists(metadata_path):
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
        optimal_threshold = metadata.get('optimal_threshold', 0.5)
        num_features = metadata.get('num_features', None)

print(f"  Optimal Threshold: {optimal_threshold:.4f}")
print(f"  Expected Features: {num_features}")

# Parse arguments
parser = argparse.ArgumentParser(description='Test XGBoost URL Classification Model')
parser.add_argument('--file', type=str, required=True, help='CSV file with URL features (36+ numerical columns)')
args = parser.parse_args()

# Smart file path resolution - try multiple locations
file_path = args.file

# Try 1: Exact path as provided
if not os.path.exists(file_path):
    # Try 2: Look in Test/input_data/
    alternative_path = os.path.join(project_root, 'Test', 'input_data', os.path.basename(args.file))
    if os.path.exists(alternative_path):
        file_path = alternative_path
        print(f"  📁 Found file in: {file_path}")
    else:
        # Try 3: Look in current directory
        if os.path.exists(os.path.basename(args.file)):
            file_path = os.path.basename(args.file)
            print(f"  📁 Found file in current directory")
        else:
            # File not found - show helpful error
            print(f"ERROR: File not found: {args.file}")
            print(f"\nTried locations:")
            print(f"  1. {args.file}")
            print(f"  2. {alternative_path}")
            print(f"  3. {os.path.basename(args.file)} (current directory)")
            print(f"\n✓ SOLUTION: Use one of these:")
            print(f"  - python scripts/inference/test_url_model.py --file url_features_correct.csv")
            print(f"    (automatically searches Test/input_data/)")
            print(f"  - python scripts/inference/test_url_model.py --file Test/input_data/url_features_correct.csv")
            print(f"    (full relative path from Impelement/)")
            print(f"  - python scripts/inference/test_url_model.py --file /absolute/path/to/url_features.csv")
            print(f"    (absolute path)")
            print(f"\nCSV Format (requires 36 URL feature columns):")
            print(f"  DomainLength, IsDomainIP, CharContinuationRate, ... NoOfExternalRef, [optional_label]")
            test_input_dir = os.path.join(project_root, 'Test', 'input_data')
            if os.path.exists(test_input_dir):
                print(f"\n📂 Available files in Test/input_data/:")
                for file in os.listdir(test_input_dir):
                    if file.endswith('.csv'):
                        print(f"     - {file}")
            sys.exit(1)

# Load CSV
print(f"\n[STEP 2] Loading URL Features from CSV...")
df = pd.read_csv(file_path)
print(f"  Loaded {len(df)} rows")
print(f"  Columns: {len(df.columns)}")

# Expected features that the model was trained on
expected_features = [
    'DomainLength', 'IsDomainIP', 'CharContinuationRate', 'TLDLegitimateProb', 
    'TLDLength', 'NoOfSubDomain', 'HasObfuscation', 'NoOfObfuscatedChar', 
    'ObfuscationRatio', 'IsHTTPS', 'LineOfCode', 'LargestLineLength', 'HasTitle',
    'DomainTitleMatchScore', 'HasFavicon', 'Robots', 'IsResponsive', 
    'NoOfSelfRedirect', 'HasDescription', 'NoOfPopup', 'NoOfiFrame', 
    'HasExternalFormSubmit', 'HasSocialNet', 'HasSubmitButton', 'HasHiddenFields',
    'HasPasswordField', 'Bank', 'Pay', 'Crypto', 'HasCopyrightInfo', 'NoOfImage',
    'NoOfCSS', 'NoOfJS', 'NoOfSelfRef', 'NoOfEmptyRef', 'NoOfExternalRef'
]

# Check if last column might be label
has_label = False
y_true = None

# Try to find label column
label_candidates = ['label', 'target', 'y', 'class']
for col in label_candidates:
    if col in df.columns:
        has_label = True
        y_true = df[col].values
        print(f"  💡 Detected label column: {col}")
        break

# Filter to only the expected features
missing_features = [f for f in expected_features if f not in df.columns]
extra_features = [f for f in df.columns if f not in expected_features and f not in label_candidates]

if missing_features:
    print(f"\n  ❌ ERROR: Missing features: {missing_features}")
    print(f"  Model expects {len(expected_features)} features, but {len(missing_features)} are missing")
    sys.exit(1)

if extra_features:
    print(f"  ⚠️  WARNING: Extra features found (will be ignored): {extra_features}")
    print(f"  Filtering to {len(expected_features)} expected features only...")

# Filter dataframe to only expected features
X = df[expected_features].copy()
print(f"  ✓ Using {X.shape[1]} features for prediction")

# Predict
print(f"\n[STEP 3] Making Predictions...")
print(f"  Processing {len(X)} samples...")

inference_start = time.time()
try:
    y_pred_proba = model.predict_proba(X)[:, 1]
    y_pred = (y_pred_proba >= optimal_threshold).astype(int)
    print(f"  ✓ Predictions completed in {time.time() - inference_start:.2f}s")
except Exception as e:
    print(f"ERROR during prediction: {e}")
    print(f"Model expects {num_features} features, got {X.shape[1]}")
    sys.exit(1)

# Prepare results (optimized - no slow iteration)
results = []
for idx in range(len(y_pred)):
    risk_level = 'HIGH' if y_pred[idx] == 1 and y_pred_proba[idx] > 0.7 else 'MEDIUM' if y_pred_proba[idx] >= 0.5 else 'LOW'
    
    results.append({
        'sample_id': idx + 1,
        'prediction': 'PHISHING' if y_pred[idx] == 1 else 'LEGITIMATE',
        'confidence': f"{y_pred_proba[idx]*100:.2f}%",
        'risk_level': risk_level
    })

results_df = pd.DataFrame(results)

# Calculate metrics if labels provided
if has_label:
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_true, y_pred_proba)
    
    print(f"\n[PREDICTION PERFORMANCE METRICS]")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  ROC-AUC:   {roc_auc:.4f}")

# Save and display results
output_dir = os.path.join(project_root, 'Test', 'output_predictions')
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, 'url_predictions.csv')
results_df.to_csv(output_file, index=False)

print(f"\n[BATCH PREDICTION SUMMARY]")
print(f"  Total URLs: {len(results)}")
print(f"  Phishing: {(results_df['prediction'] == 'PHISHING').sum()}")
print(f"  Legitimate: {(results_df['prediction'] == 'LEGITIMATE').sum()}")
print(f"  High Risk: {(results_df['risk_level'] == 'HIGH').sum()}")
print(f"  Results saved to: {output_file}")
print(f"\n  Total execution time: {time.time() - start_time:.2f}s")

print(f"\nFirst 10 predictions:")
print(results_df.head(10).to_string(index=False))

print("\n" + "=" * 80)
