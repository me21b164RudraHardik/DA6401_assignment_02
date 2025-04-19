# DA6401 - Image Classification Assignment

This repository contains the complete implementation and evaluation for the DA6401 image classification assignment. The objective of the assignment is to explore two different deep learning strategies for image classification:

1. Building a CNN model from scratch.
2. Applying transfer learning by fine-tuning a pretrained GoogLeNet model using layer freezing.

The models are trained and evaluated on a well-balanced subset of the iNaturalist 12K dataset.

---

## Dataset

- Dataset: iNaturalist 12K (subset with 10 balanced classes).
- Each class contains approximately the same number of images to ensure fairness in evaluation.
- Classes:
  - Amphibia, Animalia, Arachnida, Aves, Fungi, Insecta, Mammalia, Mollusca, Plantae, Reptilia
- Structure:
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

## Assignment Structure

### Part A - Scratch CNN Model

- Implemented a custom convolutional neural network from scratch using PyTorch.
- Contains 2-3 convolutional layers with ReLU activations and max-pooling.
- Followed by fully connected layers for classification into 10 classes.
- Model trained on the training set and evaluated on the test set.
- Achieved a test accuracy of approximately 10%, which is expected due to the simplicity of the architecture and limited training epochs.

Notebook: `DA6401_Assignment_02_PART_A_01.ipynb`

### Part B - Fine-Tuned GoogLeNet (Transfer Learning)

- Used GoogLeNet from torchvision.models as a base model.
- Feature extraction layers are frozen to preserve learned representations from ImageNet.
- The final classification head is modified and fine-tuned for our 10-class task.
- Achieved significantly better performance (>85% accuracy) compared to the scratch model.

Notebook: `DA6401_Assignment_02_PART_B.ipynb`

---

## Setup Instructions
Install requirements:
   ```
   pip install torch torchvision matplotlib wandb
   ```


## How to Run

- Run the Part A notebook to train the scratch CNN model and observe the performance.
- Run the Part B notebook to train and evaluate the fine-tuned GoogLeNet model using transfer learning.
- Predictions and performance metrics are visualized within each notebook.

---

## Results Summary

| Approach             | Test Accuracy |
|----------------------|----------------|
| CNN from Scratch     | ~10%           |
| Fine-tuned GoogLeNet | >75%           |

---
