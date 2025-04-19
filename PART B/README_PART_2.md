# Image Classification using Fine-Tuned GoogLeNet (Transfer Learning)

This project demonstrates transfer learning by fine-tuning the GoogLeNet model on a balanced version of the iNaturalist 12K dataset. The technique of freezing early layers is applied to retain low-level feature extraction while adapting higher layers for classification.

---

## Dataset

- Dataset: Subset of iNaturalist 2021 with 10 well-balanced classes:
  - Amphibia, Animalia, Arachnida, Aves, Fungi, Insecta, Mammalia, Mollusca, Plantae, Reptilia
- Directory structure:
  ```
  inaturalist_12K/
  ├── train/
  │   ├── Amphibia/
  │   ├── Animalia/
  │   └── ...
  └── test/
      ├── Amphibia/
      ├── Animalia/
      └── ...
  ```

---

## Model: GoogLeNet (Transfer Learning)

- Base architecture: GoogLeNet (Inception v1) from torchvision.models.
- Transfer learning applied by:
  - Freezing all convolutional layers (feature extractor).
  - Replacing the final classifier (fully connected layer) to suit 10 classes.
  - Only the classifier (and optionally the last few inception blocks) is fine-tuned.
- Benefits:
  - Faster training
  - Better performance on limited data
  - Preserves learned visual features from large datasets (ImageNet)

---

## Training Pipeline

1. Environment Setup:
   - Python ≥ 3.8
   - Install dependencies:  
     pip install torch torchvision matplotlib wandb

2. Data Preprocessing:
   - Resize and center crop to 224×224 (GoogLeNet input size).
   - Normalize using ImageNet mean and std.
   - DataLoaders prepared for train and test sets.

3. Training:
   - Model initialized with pretrained=True.
   - All parameters except classifier are frozen.
   - Optimizer: Adam on classifier parameters.
   - Loss: CrossEntropyLoss
   - Training progress logged with accuracy/loss using wandb or matplotlib.

---

## Evaluation

- Model tested on the test split of the dataset.
- Metrics:
  - Accuracy (overall)
  - Visual predictions on a sample grid of test images
- Performance achieved:
  - Accuracy > 75% on test data (varies depending on learning rate and training epochs)

---

## Observations

- Transfer learning significantly improves model accuracy vs scratch model (from ~10% to >70%).
- Freezing layers reduces training time while retaining strong performance.
- GoogLeNet architecture is effective for moderate-sized datasets due to inception blocks.
