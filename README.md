# MNIST Handwritten Digit Classifier

**Notebook:** `day1\_2.ipynb`

Image classification with a fully connected neural network in PyTorch on the **MNIST** dataset.

## Pipeline

1. **Data**: `torchvision.datasets.MNIST` (60,000 train / 10,000 test) with `ToTensor()` (pixels scaled to 0-1).
2. **Visualize**: first 5 images, using `.squeeze()` to go from `\[1, 28, 28]` to `\[28, 28]`.
3. **DataLoaders**: batch size 64 (batches are `\[64, 1, 28, 28]`).
4. **Model** `DigitClassifier`: flatten (784) → 128 → 64 → 10, ReLU between layers.
5. **Train** with `CrossEntropyLoss` and Adam (lr = 0.001) for 5 epochs.
6. **Evaluate**: accuracy via `torch.max` on the 10 output scores.
7. **Plot**: loss curves, accuracy curve, and 5 test predictions (green = correct, red = wrong).

## Results

|Epoch|Val Loss|Accuracy|
|-|-|-|
|1|0.3169|90.82%|
|3|0.1782|94.72%|
|5|0.1435|**95.69%**|

## Notes

* The model is built as `DigitClassifier(INPUT\_DIM, HIDDEN\_DIM\_1, HIDDEN\_DIM\_2, OUTPUT\_DIM)`, but the class signature is `(input\_dim, output\_dim, hidden\_dim1, hidden\_dim2)`. The arguments are in a different order, so the layers get the wrong sizes. Pass them by keyword (or reorder) to fix it.
* The emoji in prediction titles may not render in the default Matplotlib font.

## Requirements

```bash
pip install torch torchvision matplotlib tqdm
```

