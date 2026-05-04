import torch
import sys
import os
import argparse
import pandas as pd
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from pathlib import Path
import time

print("=" * 80)
print("TEXT MODEL INFERENCE - TEST NEW EMAILS")
print("=" * 80)

start_time = time.time()

# Get script directory and construct absolute path to models
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))  # Go up to Impelement/
model_path = os.path.join(project_root, 'models', 'text_model_distilbert')

# Check if model exists
if not os.path.exists(model_path):
    print(f"ERROR: Model not found at {model_path}")
    print(f"DEBUG: Script dir: {script_dir}")
    print(f"DEBUG: Project root: {project_root}")
    print("Please train the model first using: python scripts/stream_A_text/2_train_distilbert.py")
    sys.exit(1)

# Load model and tokenizer
print(f"\n[STEP 1] Loading Model and Tokenizer...")
load_start = time.time()
try:
    tokenizer = DistilBertTokenizer.from_pretrained(model_path)
    model = DistilBertForSequenceClassification.from_pretrained(model_path)
    print(f"  ✓ Model loaded successfully in {time.time() - load_start:.1f}s")
except Exception as e:
    print(f"ERROR loading model: {e}")
    sys.exit(1)

# Move model to device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()
print(f"  Device: {device}")
print(f"  Total startup time: {time.time() - start_time:.1f}s")

# Parse arguments
parser = argparse.ArgumentParser(description='Test DistilBERT Email Classification Model')
parser.add_argument('--text', type=str, help='Single email text to classify')
parser.add_argument('--file', type=str, help='CSV file with emails (columns: text, optional_label)')
args = parser.parse_args()

def predict_email(text):
    """Predict if an email is phishing or legitimate (single)"""
    # Tokenize
    encoding = tokenizer(
        text,
        max_length=384,
        padding=True,
        truncation=True,
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    # Predict
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)
        pred_class = torch.argmax(probs, dim=-1)
        confidence = probs[0, pred_class].item()
    
    return pred_class.item(), confidence

def predict_emails_batch(texts):
    """Predict multiple emails at once (FASTER - batch processing)"""
    # Tokenize all texts at once
    encodings = tokenizer(
        texts,
        max_length=384,
        padding=True,
        truncation=True,
        return_tensors='pt'
    )
    
    input_ids = encodings['input_ids'].to(device)
    attention_mask = encodings['attention_mask'].to(device)
    
    # Predict all at once
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)
        pred_classes = torch.argmax(probs, dim=-1)
        confidences = probs[range(len(texts)), pred_classes]
    
    return pred_classes.cpu().numpy(), confidences.cpu().numpy()

# Get class names (assuming binary: 0=Legitimate, 1=Phishing)
class_names = {0: 'LEGITIMATE', 1: 'PHISHING'}

# Process input
if args.text:
    print(f"\n[STEP 2] Testing Single Email...")
    print(f"\nEmail Text:")
    print(f"  {args.text[:100]}..." if len(args.text) > 100 else f"  {args.text}")
    
    pred_class, confidence = predict_email(args.text)
    
    print(f"\n[PREDICTION RESULT]")
    print(f"  Classification: {class_names[pred_class]}")
    print(f"  Confidence: {confidence*100:.2f}%")
    print(f"  Risk Level: {'HIGH' if pred_class == 1 and confidence > 0.7 else 'MEDIUM' if confidence >= 0.5 else 'LOW'}")

elif args.file:
    print(f"\n[STEP 2] Testing Batch from CSV...")
    
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
                print(f"  - python scripts/inference/test_text_model.py --file emails_test.csv")
                print(f"    (automatically searches Test/input_data/)")
                print(f"  - python scripts/inference/test_text_model.py --file Test/input_data/emails_test.csv")
                print(f"    (full relative path from Impelement/)")
                print(f"  - python scripts/inference/test_text_model.py --file /absolute/path/to/emails.csv")
                print(f"    (absolute path)")
                test_input_dir = os.path.join(project_root, 'Test', 'input_data')
                if os.path.exists(test_input_dir):
                    print(f"\n📂 Available files in Test/input_data/:")
                    for file in os.listdir(test_input_dir):
                        if file.endswith('.csv'):
                            print(f"     - {file}")
                sys.exit(1)
    
    
    # Load CSV
    df = pd.read_csv(file_path)
    print(f"  Loaded {len(df)} rows from {args.file}")
    
    # Infer text column
    text_col = None
    for col in ['text', 'email', 'message', 'content']:
        if col in df.columns:
            text_col = col
            break
    
    if text_col is None:
        text_col = df.columns[0]
    
    print(f"  Using text column: {text_col}")
    
    # Predict using BATCH PROCESSING (much faster)
    print(f"\n[BATCH PROCESSING {len(df)} EMAILS]")
    email_texts = df[text_col].astype(str).tolist()
    
    # Process in batches for better performance
    batch_size = 32  # Process 32 emails at a time
    all_pred_classes = []
    all_confidences = []
    
    inference_start = time.time()
    for batch_start in range(0, len(email_texts), batch_size):
        batch_end = min(batch_start + batch_size, len(email_texts))
        batch_texts = email_texts[batch_start:batch_end]
        
        pred_classes, confidences = predict_emails_batch(batch_texts)
        all_pred_classes.extend(pred_classes)
        all_confidences.extend(confidences)
        
        print(f"  ✓ Processed {batch_end}/{len(email_texts)} emails ({time.time() - inference_start:.1f}s)")
    
    print(f"  Total inference time: {time.time() - inference_start:.1f}s")
    
    # Prepare results
    results = []
    for idx, email_text in enumerate(email_texts):
        pred_class = all_pred_classes[idx]
        confidence = all_confidences[idx]
        
        results.append({
            'email': email_text[:50] + '...' if len(email_text) > 50 else email_text,
            'prediction': class_names[pred_class],
            'confidence': f"{confidence*100:.2f}%",
            'risk_level': 'HIGH' if pred_class == 1 and confidence > 0.7 else 'MEDIUM' if confidence >= 0.5 else 'LOW'
        })
    
    # Save results
    results_df = pd.DataFrame(results)
    output_dir = os.path.join(project_root, 'Test', 'output_predictions')
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'email_predictions.csv')
    results_df.to_csv(output_file, index=False)
    
    print(f"\n[BATCH PREDICTION SUMMARY]")
    print(f"  Total Emails: {len(results)}")
    print(f"  Phishing: {(results_df['prediction'] == 'PHISHING').sum()}")
    print(f"  Legitimate: {(results_df['prediction'] == 'LEGITIMATE').sum()}")
    print(f"  Results saved to: {output_file}")
    print(f"\n  Total execution time: {time.time() - start_time:.1f}s")
    print(f"\nFirst 5 results:")
    print(results_df.head())

else:
    print("\nUSAGE:")
    print("  Single email: python test_text_model.py --text \"Email content here\"")
    print("  Batch CSV:    python test_text_model.py --file emails.csv")
    print("\nCSV Format (text column required):")
    print("  text,label (optional)")
    print("  \"Your email content here\",0")

print("\n" + "=" * 80)
