# Feedforward Neural Network in PyTorch

A clean and modular implementation of a **Feedforward Neural Network (FFN)** for multi-class classification on the **Iris dataset** using **PyTorch**.

This project demonstrates a complete deep learning training pipeline, including reproducibility, early stopping, learning rate scheduling, checkpointing, and evaluation across multiple random seeds.

---

## 🚀 Features

* Feedforward Neural Network implemented in PyTorch
* Modular project structure
* Reproducible training using fixed random seeds
* AdamW optimizer with weight decay
* Learning Rate Scheduler (`ReduceLROnPlateau`)
* Early Stopping
* Model checkpointing
* Training, validation, and test split
* Evaluation across multiple random seeds
* Performance metrics:

  * Accuracy
  * ROC-AUC
  * Confusion Matrix
  * Classification Report (Precision, Recall, F1-score)

---

## 🛠 Technologies

* Python
* PyTorch
* NumPy
* scikit-learn

---

## 📂 Project Structure

```text
src/
│
├── data/              # Data loading and preprocessing
├── evaluation/        # Evaluation metrics
├── experiments/       # Multi-seed experiment runner
├── models/            # Neural network architecture
├── training/          # Training pipeline
├── utils/             # Utility functions (seed, helpers)
├── config.py          # Hyperparameters
├── inference.py       # Model inference
└── main.py            # Project entry point
```

---

## ⚙️ How to Run

Clone this repository, Install the required packages:

```bash
pip install -r requirements.txt
```

Run the project:

```bash
python src/main.py
```

---

## 📈 Results

The model was trained and evaluated using **5 different random seeds** to measure training stability and reproducibility.

| Metric                  | Result                     |
| ----------------------- | -------------------------- |
| Dataset                 | Iris                       |
| Model                   | Feedforward Neural Network |
| Optimizer               | AdamW                      |
| Learning Rate Scheduler | ReduceLROnPlateau          |
| Early Stopping          | ✅                          |
| Evaluation Seeds        | 5                          |
| Mean Accuracy           | **94.67%**                 |
| Standard Deviation      | **4.99%**                  |
| Best Accuracy           | **100.00%**                |
| Lowest Accuracy         | **86.67%**                 |
| Mean ROC-AUC            | **1.00**                   |

For each seed, the project reports:

* Accuracy
* ROC-AUC
* Confusion Matrix
* Classification Report (Precision, Recall, F1-score)

Example output:

```text
Seed 42

Accuracy : 1.0000
ROC-AUC  : 1.0000

Confusion Matrix
[[5 0 0]
 [0 6 0]
 [0 0 4]]
```

---

## 📚 What I Learned

Through this project I gained experience with:

* Building modular PyTorch projects
* Designing reproducible machine learning pipelines
* Implementing early stopping and model checkpointing
* Using learning rate scheduling
* Evaluating classification models using Accuracy, ROC-AUC, Precision, Recall and F1-score
* Running experiments across multiple random seeds to measure model stability
* Organizing a deep learning project using a clean folder structure

---

## 🚀 Future Improvements

* Compare the FFN against XGBoost on the same dataset
* Extend the pipeline to larger real-world biomedical datasets
* Add hyperparameter tuning
* Integrate experiment tracking (MLflow or Weights & Biases)

---

## 👨‍💻 Author

This project was developed as part of my machine learning portfolio while preparing for a master's thesis in AI and bioinformatics. 

💡 **The Backpropagation Evolution:** Before building this production-ready PyTorch pipeline, I built the entire neural network and backpropagation engine completely from scratch using only NumPy to master the underlying calculus and gradient math. You can check out that foundational project here: [Link ]
