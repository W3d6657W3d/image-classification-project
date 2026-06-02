# Dataset Report

## Dataset Choice

This project uses a 12-class garbage classification dataset as the primary
dataset for a lightweight image classification training and deployment system.

Classes:

- battery
- biological
- brown-glass
- cardboard
- clothes
- green-glass
- metal
- paper
- plastic
- shoes
- trash
- white-glass

The dataset is suitable for this portfolio project because it has a practical
business scenario, clear class labels, a moderate download size, and enough
images for transfer learning with a lightweight model.

## Raw Dataset Summary

Raw dataset path:

```text
data/raw/garbage_classification/
```

Raw image format:

```text
.jpg
```

Raw class distribution:

| Class | Images |
| --- | ---: |
| battery | 945 |
| biological | 985 |
| brown-glass | 607 |
| cardboard | 891 |
| clothes | 5325 |
| green-glass | 629 |
| metal | 769 |
| paper | 1050 |
| plastic | 865 |
| shoes | 1977 |
| trash | 697 |
| white-glass | 775 |

Total images:

```text
15515
```

Largest class:

```text
clothes: 5325
```

Smallest class:

```text
brown-glass: 607
```

Raw imbalance ratio:

```text
8.77
```

No unreadable images were found during the basic image validation step.

## Processing Strategy

The raw dataset has a clear class imbalance, especially for `clothes` and
`shoes`. To keep the project suitable for low-end local hardware and to make the
first model evaluation easier to interpret, the training dataset uses a balanced
subset.

Processing rule:

```text
max 600 images per class
```

Split rule:

```text
train: 70%
validation: 15%
test: 15%
random seed: 42
```

Processed dataset path:

```text
data/processed/garbage_classification/
```

Processed split summary:

| Split | Images per class | Total images |
| --- | ---: | ---: |
| train | 420 | 5040 |
| validation | 90 | 1080 |
| test | 90 | 1080 |

Total processed images:

```text
7200
```

## Generated Artifacts

Data scripts:

```text
src/data/analyze_dataset.py
src/data/prepare_dataset.py
```

Reports:

```text
reports/class_distribution.csv
reports/split_summary.csv
```

Figures:

```text
reports/figures/class_distribution.png
reports/figures/sample_grid.png
reports/figures/split_distribution.png
```

## Notes for Modeling

Recommended baseline:

- PyTorch `ImageFolder`
- image size: 224 x 224
- pretrained MobileNetV2 or ResNet18
- batch size: 8 or 16 depending on local memory
- start with 5 to 10 epochs

The first goal is to produce a reliable end-to-end baseline before optimizing
model accuracy.
