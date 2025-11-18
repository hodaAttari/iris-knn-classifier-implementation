## Iris KNN Classifier

This project implements the **K‑Nearest Neighbors (KNN)** algorithm on the classic **Iris dataset**, both **from scratch (NumPy-based)** and using **Scikit‑Learn**.  
It covers all steps from data exploration and profiling to model evaluation and visualization.

---

### Project Overview
- **Dataset:** Iris (150 samples, 4 features, 3 classes)
- **Goal:** Classify flower species based on sepal and petal dimensions
- **Approach:**
  1. Data Profiling using [YData‑Profiling](https://github.com/ydataai/ydata-profiling)
  2. Data Preprocessing (Scaling via `MinMaxScaler`)
  3. Model Implementation (`MyKNNClassifier` from scratch)
  4. Model Evaluation (`scikit-learn` KNN + GridSearch)
  5. Performance Reporting (`classification_report`)

---

### Implementation Details
- **Custom Class:** `MyKNNClassifier` in [`knn.py`](./knn.py)
  - Supports metrics: `euclidean`, `manhattan`
  - Weighted and unweighted voting
  - Compatible with Scikit‑Learn pipelines

- **Experiments:**  
  - [`iris.ipynb`](./iris.ipynb): Simple demo of custom KNN  
  - [`knn.ipynb`](./knn.ipynb): Full implementation with preprocessing and grid search  
  - [`report.html`](./report.html): Exploratory data profiling report

---

### Example Output
Typical model accuracy (on test split):

accuracy: ~0.90

weighted avg f1-score: ~0.90


### Run Locally
```bash
git clone https://github.com/yourusername/iris-knn-classifier
cd iris-knn-classifier
pip install -r requirements.txt
jupyter notebook iris.ipynb
```
### requirements
- numpy
- pandas
- scikit-learn
- ydata-profiling
- matplotlib
- seaborn

### Files

├── knn.py           # Custom KNN implementation

├── iris.ipynb       # Basic KNN on Iris example

├── knn.ipynb        # Full experiment (scikit-learn + pipeline)

├── report.html      # YData profiling report
