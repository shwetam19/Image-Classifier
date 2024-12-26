# **Image Classifier with PyTorch**

This project demonstrates how to build a state-of-the-art **image classification application** using PyTorch. The application allows users to train a neural network on a dataset of flower images, save the trained model, and predict flower categories for new images. The project also features a command-line interface for ease of use.

---

## **Dataset**

The project uses the [Flowers 102 dataset](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html), which contains images of 102 flower categories. The dataset is ideal for demonstrating the power of transfer learning and deep neural networks.

## **Features**

1. **Train a Model**:
   - Train a neural network on the flower dataset using a pre-trained model (e.g., VGG16, ResNet18).
   - Customize hyperparameters like learning rate, hidden units, and epochs.
   - Save the trained model checkpoint for future use.

2. **Predict Classes**:
   - Predict the class of a flower image using the trained model.
   - Display the top \( K \) most probable classes with associated probabilities.
   - Optionally, map class indices to human-readable names using a JSON mapping file.

3. **Command-Line Interface**:
   - Two scripts: `train.py` for training and `predict.py` for predictions.
   - Easily run training and predictions directly from the command line with configurable options.

---

# Usage

## 1. Training the Model
Run the `train.py` script to train the model:

```bash
python train.py flowers --arch vgg16 --learning_rate 0.001 --hidden_units 512 --epochs 5 --gpu
```

**Options:**
- `flowers`: Path to the dataset directory.
- `--arch`: Specify the model architecture (e.g., vgg16, resnet18).
- `--learning_rate`: Set the learning rate (default: 0.001).
- `--hidden_units`: Define the number of hidden units for the classifier (default: 512).
- `--epochs`: Set the number of training epochs (default: 5).
- `--gpu`: Use GPU for training.

## 2. Predicting with the Model
Run the `predict.py` script to predict the class of a flower image:

```bash
python predict.py /path/to/image checkpoint.pth --top_k 5 --category_names cat_to_name.json --gpu
```

**Options:**
- `/path/to/image`: Path to the input image.
- `checkpoint.pth`: Path to the saved model checkpoint.
- `--top_k`: Specify the number of top predictions to display (default: 5).
- `--category_names`: Path to a JSON file mapping class indices to category names (optional).
- `--gpu`: Use GPU for inference.
