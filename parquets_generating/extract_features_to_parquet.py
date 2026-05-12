import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from aggregation import aggregation_and_feature_extraction
from model import MAX_LENGTH, get_model_and_tokenizer

# ============================================================
# CONFIG
# ============================================================

DATA_FILE = "./data/dataset.csv"
TEST_FILE = "./data/test.csv"

OUTPUT_DIR = "./artifacts"

EXPORT_TEST = True

BATCH_SIZE = 4
USE_GEOMETRIC = False

# ============================================================

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)


def extract_features(texts, model, tokenizer, device):
    all_features = []

    for start in tqdm(
        range(0, len(texts), BATCH_SIZE),
        desc="Extracting & aggregating",
        unit="batch",
    ):
        batch_texts = texts[start : start + BATCH_SIZE]

        encoding = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
        )

        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        hidden = torch.stack(outputs.hidden_states, dim=1).float()
        mask = attention_mask.cpu()

        for i in range(hidden.size(0)):
            feat = aggregation_and_feature_extraction(
                hidden[i],
                mask[i],
                use_geometric=USE_GEOMETRIC,
            )

            all_features.append(feat.cpu())

    return np.vstack([f.numpy() for f in all_features])


if __name__ == "__main__":

    # ========================================================
    # DEVICE
    # ========================================================

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Device: {device}")

    # ========================================================
    # LOAD MODEL
    # ========================================================

    model, tokenizer = get_model_and_tokenizer()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.to(device)

    # ========================================================
    # TRAIN DATASET
    # ========================================================

    print("\nLoading dataset.csv ...")

    df = pd.read_csv(DATA_FILE)

    texts = [
        f"{row['prompt']}{row['response']}"
        for _, row in df.iterrows()
    ]

    labels = np.array([int(float(h)) for h in df["label"]])

    t0 = time.time()

    X = extract_features(
        texts=texts,
        model=model,
        tokenizer=tokenizer,
        device=device,
    )

    elapsed = time.time() - t0

    print(f"\nFeature extraction completed in {elapsed:.1f} seconds")
    print(f"Feature matrix shape: {X.shape}")

    feature_cols = [f"f_{i}" for i in range(X.shape[1])]

    df_features = pd.DataFrame(X, columns=feature_cols)

    df_features["label"] = labels
    df_features["prompt"] = df["prompt"]
    df_features["response"] = df["response"]

    dataset_output_path = (
        Path(OUTPUT_DIR) / "features_dataset.parquet"
    )

    df_features.to_parquet(dataset_output_path, index=False)

    print(f"\nSaved dataset features to:")
    print(dataset_output_path)

    # ========================================================
    # TEST DATASET
    # ========================================================

    if EXPORT_TEST:

        print("\nLoading test.csv ...")

        df_test = pd.read_csv(TEST_FILE)

        test_texts = [
            f"{row['prompt']}{row['response']}"
            for _, row in df_test.iterrows()
        ]

        X_test = extract_features(
            texts=test_texts,
            model=model,
            tokenizer=tokenizer,
            device=device,
        )

        df_test_features = pd.DataFrame(
            X_test,
            columns=feature_cols,
        )

        df_test_features["prompt"] = df_test["prompt"]
        df_test_features["response"] = df_test["response"]

        test_output_path = (
            Path(OUTPUT_DIR) / "features_test.parquet"
        )

        df_test_features.to_parquet(
            test_output_path,
            index=False,
        )

        print(f"\nSaved test features to:")
        print(test_output_path)

    print("\nDone.")
