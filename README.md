# CNN Optimizer Comparison on CIFAR-10

This project compares the performance of various optimizers (SGD, Momentum, NAG, Adagrad, Adadelta, Adam, RMSprop)
on a custom CNN architecture designed using the reversed digits of a student ID.

## 🧑‍💻 Authors
- **Art1Mal**

## 🧠 Architecture Design
A systematic approach was applied to define the number of filters in convolutional layers, using a reversed digit-based scheme to ensure reproducibility and architectural diversity without disclosing personal identifiers. All models use:

- L2 Regularization = 0.001
- Dropout = 0.3
- Batch size = 64
- CIFAR-10 dataset
- Training duration: 45 epochs
- Evaluation metrics: Accuracy, Precision, Recall, F1-score

## ⚙️ Optimizers & Parameters

| Optimizer | Learning Rate | Other Parameters |
|-----------|---------------|------------------|
| SGD       | 0.01          | momentum=0.9, nesterov=False |
| Momentum  | 0.01          | momentum=0.9, nesterov=False |
| NAG       | 0.01          | momentum=0.9, nesterov=True  |
| Adagrad   | 0.01          | default config                |
| Adadelta  | 1.0           | default config                |
| Adam      | 0.001         | default config                |
| RMSprop   | 0.001         | default config                |

## 📊 Results Summary

| Optimizer | Accuracy | Precision | Recall | F1-score | Time (min) |
|-----------|----------|-----------|--------|----------|------------|
| SGD       | 0.6620   | 0.6899    | 0.6620 | 0.6611   | 33.92      |
| NAG       | 0.6451   | 0.6510    | 0.6451 | 0.6476   | XX.XX      |
| Adagrad   | 0.6585   | 0.6670    | 0.6585 | 0.6557   | 32.01      |
| Adadelta  | **0.7002** | **0.7068** | **0.7002** | **0.7001** | 33.44    |
| Adam      | 0.6813   | 0.6841    | 0.6813 | 0.6744   | 32.41      |
| RMSprop   | 0.6357   | 0.6743    | 0.6357 | 0.6372   | 33.51      |

📌 **Best Optimizer: Adadelta**

## 🚀 How to Run

### Notebook (Recommended for Colab)
Open `optimizer_experiment.ipynb` in Google Colab.

### Local Python Script

```bash
pip install -r requirements.txt
python main.py
```

## 📁 File Structure

- `main.py` — the training script
- `optimizer_experiment.ipynb` — Jupyter version of the project
- `README.md` — this documentation
- `requirements.txt` — list of required packages
- `results/` — folder for metrics or screenshots

## 📜 License
MIT License (optional)

