
 Quantum Support Vector Machine (QSVM) with Pulsar Dataset

This project implements a Quantum Support Vector Machine (QSVM) to classify pulsar stars using quantum kernel methods. It uses PennyLane, Scikit-Learn, and Matplotlib to process data, compute quantum kernels, train the QSVM, and evaluate its performance.

 Features
- Quantum kernel calculation using PennyLane's QNode framework.
- Preprocessing that scales the input data between 0 and π for compatibility with quantum circuits.
- Classical Support Vector Classifier (SVC) training using the quantum kernel.
- Performance evaluation with metrics like accuracy, precision, recall, specificity, and F1 score.
- Visualizations, including kernel matrix heatmaps, a confusion matrix, and decision boundaries using PCA.

 Requirements
- Python 3.8 or newer
- Libraries:
  - pandas
  - numpy
  - pennylane
  - scikit-learn
  - matplotlib
  - seaborn

Install the dependencies using:

```
pip install pandas numpy pennylane scikit-learn matplotlib seaborn
```

Dataset
The pulsar.csv file is expected in the project directory. Ensure the file contains the following:
- Eight feature columns.
- A ninth column indicating the label (0 or 1).

 Usage

1. Fetch and preprocess the data using the fetch_data_random_seed_val function. This step reads the dataset, scales the features, and splits it into training, validation, and test sets.
2. Compute the quantum kernel using the quantum_kernel function.
3. Train and evaluate the QSVM using the quantum kernel matrix.
4. Visualize the results, including heatmaps of the kernel matrices, the confusion matrix, and decision boundaries.

 Example Output

Performance metrics:
```
Accuracy: 0.XX
Precision: 0.XX
Recall: 0.XX
Specificity: 0.XX
F1 Score: 0.XX
```

Heatmaps and decision boundaries are visualized using Matplotlib.

 Code Overview

Quantum kernel function:
```python
@qml.qnode(dev)
def quantum_kernel(x1, x2):
    qml.templates.AngleEmbedding(x1, wires=range(8))
    qml.adjoint(qml.templates.AngleEmbedding)(x2, wires=range(8))
    return qml.expval(qml.PauliZ(0))
```

Kernel matrix calculation:
```python
def quantum_kernel_matrix(X1, X2):
    return np.array([[quantum_kernel(x1, x2) for x2 in X2] for x1 in X1])
```

Data preprocessing:
```python
def fetch_data_random_seed_val(n_samples, seed):
    # Reads data, scales features, and splits into train, validation, and test sets.
    ...
```

 Results

Training kernel matrix heatmap  
Test kernel matrix heatmap  
Decision boundaries  

 Future Work

- Experiment with different quantum circuit architectures.
- Explore alternative quantum kernels.
- Extend the model to multi-class classification problems.

 License

This project is licensed under the MIT License. Feel free to use and modify it as needed.  
