# Loomora V1.0

Loomora V1.0 is an in-context model for tabular classification with independently trained weights. It uses two rounds of column and row interaction followed by 12 context-learning blocks. The labeled rows supplied to `fit` provide context for prediction.

## Installation

Loomora V1.0 is distributed as the Python package `loomora` version `1.0.0`. It requires Python 3.11 or later.

Install the exact release using Git tag `v1.0.0`. After the tag is published on GitHub, replace `OWNER` with the GitHub account that hosts the repository.

```sh
python -m pip install "loomora @ git+https://github.com/OWNER/loomora.git@v1.0.0"
```

If you already have a local Git clone, use its absolute path to install the same tag.

```sh
python -m pip install "loomora @ git+file:///opt/workspace/loomora@v1.0.0"
```

Verify that the installed package reports `1.0.0`.

```sh
python -c 'from importlib.metadata import version; print(version("loomora"))'
```

## Prediction

Get the published Loomora V1.0 weights and pass their local path to the classifier. Use the same feature columns in the labeled context and in the rows to predict.

```python
from loomora import LoomoraClassifier

classifier = LoomoraClassifier("/opt/workspace/loomora/loomora-v1-classifier.safetensors")
classifier.fit(train_features, train_labels)
probabilities = classifier.predict_proba(test_features)
predictions = classifier.predict(test_features)
```

The columns of `probabilities` follow `classifier.classes_`. Binary tasks use the model's native binary head. For multiclass tasks, the classifier uses the same weights with a separate one-vs-rest context per class and normalizes the positive scores for each sample. The number of classes affects prediction time and context memory.

## Code attribution

The [Apache-2.0 license text](LICENSE), [upstream source attribution](NOTICE), and [third-party notices and license terms](THIRD-PARTY-NOTICES.md) accompany the inference source and its distribution packages. The Hugging Face model card states the terms for the released weights.
