from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class RCNNClassifierBundle:
    pipeline: Pipeline
    classes: list[str]
    backbone: str

    def save(self, path: str | Path) -> None:
        joblib.dump(
            {
                "pipeline": self.pipeline,
                "classes": self.classes,
                "backbone": self.backbone,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "RCNNClassifierBundle":
        payload = joblib.load(path)
        return cls(
            pipeline=payload["pipeline"],
            classes=payload["classes"],
            backbone=payload["backbone"],
        )


def train_classifier_from_chunks(
    chunk_paths: Iterable[str | Path],
    classes: list[str],
    backbone: str,
) -> RCNNClassifierBundle:
    scaler = StandardScaler()
    clf = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=1e-4,
        max_iter=1,
        tol=None,
        random_state=42,
    )
    class_indices = np.arange(len(classes), dtype=np.int64)

    first_batch = True
    for chunk_path in chunk_paths:
        payload = np.load(chunk_path)
        features = payload["features"]
        labels = payload["labels"]

        if features.size == 0 or labels.size == 0:
            continue

        scaler.partial_fit(features)
        scaled_features = scaler.transform(features)

        if first_batch:
            clf.partial_fit(scaled_features, labels, classes=class_indices)
            first_batch = False
        else:
            clf.partial_fit(scaled_features, labels)

    if first_batch:
        raise RuntimeError("No training chunks were available for classifier training.")

    pipeline = Pipeline(
        [
            ("scaler", scaler),
            ("clf", clf),
        ]
    )
    return RCNNClassifierBundle(pipeline=pipeline, classes=classes, backbone=backbone)
