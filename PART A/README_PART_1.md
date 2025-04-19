# Image Classification using CNN from Scratch

This project implements a basic Convolutional Neural Network (CNN) model from scratch for multi-class image classification on the balanced iNaturalist 12K dataset. The model is built and trained without using any pretrained weights or advanced architectures, emphasizing fundamental deep learning principles.

---

## Dataset

- The dataset used is a subset of the iNaturalist 2021 dataset with 10 balanced classes:
  - Amphibia, Animalia, Arachnida, Aves, Fungi, Insecta, Mammalia, Mollusca, Plantae, Reptilia
- Folder structure:
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
- Each class folder contains corresponding image files.

---

## Model Architecture

- A simple CNN model was built from scratch using PyTorch.
- Architecture includes:
  - 3 convolutional layers with ReLU activation and max-pooling
  - 2 fully connected (linear) layers
  - Output layer with 10 units (for 10 classes)
- No pretrained weights or transfer learning techniques were used.

---

## Training Procedure

1. Set up the environment:
   - Install requirements:  
     pip install torch torchvision matplotlib wandb

2. Configure training parameters:
   - Learning rate, batch size, number of epochs are configurable in the notebook.

3. Data loading and preprocessing:
   - Data is normalized and resized to 256×256.
   - Custom PyTorch Dataset and DataLoader used for batching.

4. Training:
   - Run all cells in the notebook DA6401_Assignment_02_PART_A_01.ipynb.
   - The model is trained using CrossEntropyLoss and the Adam optimizer.
   - Accuracy and loss are logged for each epoch.

---

## Evaluation

- The final model is evaluated on the test dataset using accuracy.
- Visualizations include:
  - Sample test predictions with predicted vs actual class labels.
  - Model performance metrics such as accuracy and loss curve.

---

## Observations

- The model achieved ~10% accuracy on the test set.
- Despite the dataset being balanced, performance was poor due to:
  - The limited capacity of the CNN architecture.
  - No use of regularization, or pretrained features.

---

## Files

- DA6401_Assignment_02_PART_A_01.ipynb – Main notebook
- README_PART_1.md – Instructions and documentation (this file)
