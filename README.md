# Neural Network


## Overview
This repository contains the implementation of a neural network developed from scratch in Python without using any deep learning libraries (e.g., PyTorch, TensorFlow). The project showcases the process of building, training, and evaluating a custom neural network.

## Features
- Implementation of feedforward and backpropagation algorithms.
- Modular architecture with customizable activation functions, layers, and optimizers.
- Training accuracy of up to 92.63% over 2500 iterations.

## Prerequisites
Ensure you have the following installed on your system:
- Python 3.8+
- MNIST dataset: https://www.kaggle.com/datasets/oddrationale/mnist-in-csv
- Required libraries (install via `requirements.txt`):
  ```bash
  pip install pandas
  ```

## Usage

### 1. Clone the Repository
```bash
git clone https://github.com/IAmFarrokhnejad/Neural-Network
cd your-repo-name
```

### 2. Prepare the Data
Place your dataset files in the `data/` directory. Ensure that the data is preprocessed as required by the training script.

### 3. Train the Neural Network
Run the training script to start the training process:
```bash
python mnist.py
```
Training logs, including iteration-wise accuracy, will be displayed in the console.

## Sample Output
Example of training accuracy progression:
```
Iteration 0, Accuracy: 0.1285
Iteration 1000, Accuracy: 0.9147
Iteration 2500, Accuracy: 0.9263
Iteration 7000, Accuracy: 0.9378
Iteration 9990, Accuracy: 0.9429

```

## Contributions
Contributions are welcome! Feel free to open an issue or submit a pull request.

## License
This project is licensed under the MIT License. See `LICENSE` for details.