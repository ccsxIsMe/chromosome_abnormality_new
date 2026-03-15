import pandas as pd
from sklearn.utils.class_weight import compute_class_weight
import numpy as np


def main(csv_path="/data5/chensx/MyProject/chromosome_abnormality_new/data/train.csv"):
    df = pd.read_csv(csv_path)
    labels = df["label"].values

    classes = np.array([0, 1])
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=labels
    )

    print("Classes:", classes.tolist())
    print("Weights:", weights.tolist())


if __name__ == "__main__":
    main()