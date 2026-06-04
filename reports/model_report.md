# Model Evaluation Report

## Checkpoint

`models\best_mobilenet_v2.pth`

## Overall Metrics

- Accuracy: 0.9537
- Macro precision: 0.9547
- Macro recall: 0.9537
- Macro F1: 0.9538

## Per-Class Metrics

| Class | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| battery | 0.9565 | 0.9778 | 0.9670 | 90 |
| biological | 0.9885 | 0.9556 | 0.9718 | 90 |
| brown-glass | 0.9770 | 0.9444 | 0.9605 | 90 |
| cardboard | 1.0000 | 0.9556 | 0.9773 | 90 |
| clothes | 0.9778 | 0.9778 | 0.9778 | 90 |
| green-glass | 0.9468 | 0.9889 | 0.9674 | 90 |
| metal | 0.8700 | 0.9667 | 0.9158 | 90 |
| paper | 0.9457 | 0.9667 | 0.9560 | 90 |
| plastic | 0.9070 | 0.8667 | 0.8864 | 90 |
| shoes | 0.9663 | 0.9556 | 0.9609 | 90 |
| trash | 1.0000 | 0.9889 | 0.9944 | 90 |
| white-glass | 0.9205 | 0.9000 | 0.9101 | 90 |

## Notes

This report is generated from the held-out test split. The test set is
kept separate from training and validation during model selection.
