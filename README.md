# ML-Case-Study-1
Machine Learning Case Study 2 - Data Cleaning, L2 Regularization, ROC-AUC and False Negative

# Logistic Regression – Purchase Prediction

A beginner-friendly Machine Learning project that uses **Logistic Regression** to predict whether a customer will purchase a product based on their **Age** and **Salary**.

The project also demonstrates important Machine Learning preprocessing and evaluation techniques such as:

* Handling missing values
* Train-test split
* Feature scaling
* Logistic Regression
* L2 Regularization
* Accuracy Score
* ROC-AUC Score
* ROC Curve
* Confusion Matrix

---

## 📌 Project Overview

The goal of this project is to predict the `Purchased` class using two input features:

* `Age`
* `Salary`

The target variable is:

* `0` → Not Purchased
* `1` → Purchased

The dataset contains **20 observations and 3 columns**: `Age`, `Salary`, and `Purchased`.

---

## 🧠 Machine Learning Workflow

The project follows this workflow:

```text
Dataset
   ↓
Data Inspection
   ↓
Missing Value Detection
   ↓
Missing Value Handling
   ↓
Feature & Target Separation
   ↓
Train-Test Split
   ↓
Feature Scaling
   ↓
Logistic Regression
   ↓
L2 Regularization
   ↓
Prediction
   ↓
Model Evaluation
```

---

## 📊 Dataset

The dataset is created manually using Python dictionaries and converted into a Pandas DataFrame.

### Features

| Feature | Description       |
| ------- | ----------------- |
| Age     | Customer's age    |
| Salary  | Customer's salary |

### Target

| Value | Meaning                   |
| ----- | ------------------------- |
| `0`   | Customer did not purchase |
| `1`   | Customer purchased        |

The notebook initially checks the dataset using `head()` and `isnull().sum()`.

---

## 🧹 Data Preprocessing

### 1. Missing Value Creation

A missing value is intentionally introduced into the `Salary` column:

```python
df.loc[2, 'Salary'] = np.nan
```

This demonstrates how missing data can be handled in a Machine Learning workflow.

### 2. Missing Value Handling

The missing salary value is replaced using the mean salary:

```python
df['Salary'] = df['Salary'].fillna(df['Salary'].mean())
```

After this step, there are no missing values in the dataset.

---

## 🎯 Feature and Target Selection

The input features are:

```python
X = df[['Age', 'Salary']]
```

The target variable is:

```python
y = df['Purchased']
```

Therefore:

```text
X → Age + Salary
y → Purchased
```

---

## ✂️ Train-Test Split

The dataset is divided into training and testing sets using:

```python
train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

This means:

* **80%** of the data → Training
* **20%** of the data → Testing
* `random_state=42` → Reproducible split

The notebook produces 16 training observations and 4 testing observations.

---

## 📏 Feature Scaling

`StandardScaler` is used to standardize the features:

```python
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

The scaler is fitted only on the training data and then applied to the test data.

---

## 🤖 Logistic Regression

Two Logistic Regression models are trained.

### Model 1 – Without L2 Regularization

```python
model_without_l2 = LogisticRegression(
    penalty=None,
    max_iter=1000
)
```

The model achieved:

```text
Accuracy without L2: 1.0
```

### Model 2 – With L2 Regularization

```python
model_with_l2 = LogisticRegression(
    penalty='l2',
    max_iter=1000
)
```

The model achieved:

```text
Accuracy with L2: 1.0
```

---

## 📈 Model Evaluation

### Accuracy

Both models achieved:

```text
Accuracy = 1.00
```

For this particular test split, the predictions were completely correct.

> Note: The dataset is very small, containing only 20 observations, so this accuracy should not be interpreted as evidence that the model will achieve 100% accuracy on new real-world data.

---

## 📊 ROC-AUC

The probability of the positive class is obtained using:

```python
y_probability = model_with_l2.predict_proba(X_test_scaled)[:, 1]
```

The ROC-AUC score is then calculated:

```python
auc = roc_auc_score(y_test, y_probability)
```

Result:

```text
ROC-AUC Score: 1.0
```

---

## 📉 ROC Curve

The project also generates an ROC Curve using:

```python
fpr, tpr, thresholds = roc_curve(
    y_test,
    y_probability
)
```

The graph plots:

* **False Positive Rate**
* **True Positive Rate**

---

## 🔢 Confusion Matrix

The confusion matrix is calculated using:

```python
cm = confusion_matrix(
    y_test,
    y_pred_with_l2
)
```

Output:

```text
[[2 0]
 [0 2]]
```

This gives:

| Metric              | Value |
| ------------------- | ----: |
| True Negative (TN)  |     2 |
| False Positive (FP) |     0 |
| False Negative (FN) |     0 |
| True Positive (TP)  |     2 |

---

## 🛠️ Technologies Used

* **Python 3**
* **Pandas**
* **NumPy**
* **Scikit-learn**
* **Matplotlib**
* **Google Colab / Jupyter Notebook**

The notebook uses Pandas, NumPy, Scikit-learn's Logistic Regression and evaluation metrics, and Matplotlib.

---

## 📦 Installation

Install the required libraries:

```bash
pip install pandas numpy scikit-learn matplotlib
```

---

## ▶️ How to Run

### 1. Clone the repository

```bash
git clone <your-repository-url>
```

### 2. Open the notebook

Open:

```text
934aad88-bf6b-4ee4-8f4c-77de70e6e491.ipynb
```

using:

* Jupyter Notebook
* JupyterLab
* Google Colab
* VS Code

### 3. Run all cells

Execute the notebook from top to bottom.

---

## 📁 Project Structure

```text
Logistic-Regression-Purchase-Prediction/
│
├── README.md
├── logistic_regression.ipynb
└── requirements.txt
```

---

## 🎓 Concepts Learned

This project demonstrates the following Machine Learning concepts:

* Pandas DataFrames
* Missing-value detection
* Mean imputation
* Feature and target selection
* Train-test splitting
* `random_state`
* Feature standardization
* Logistic Regression
* L2 Regularization
* Classification accuracy
* Probability prediction
* ROC Curve
* ROC-AUC
* Confusion Matrix
* TP, TN, FP and FN

---

## 🚀 Future Improvements

Possible improvements for this project include:

* Use a larger real-world dataset
* Add more customer features
* Perform exploratory data analysis
* Visualize the decision boundary
* Compare Logistic Regression with other classification algorithms
* Perform cross-validation
* Tune hyperparameters
* Add precision, recall and F1-score
* Deploy the model as a web application

---

## 📌 Results

| Evaluation Metric   | Result |
| ------------------- | -----: |
| Accuracy without L2 |   1.00 |
| Accuracy with L2    |   1.00 |
| ROC-AUC             |   1.00 |
| Test Samples        |      4 |
| Total Samples       |     20 |

The reported results are specific to the dataset and train-test split used in the notebook.

---

## 👨‍💻 Author

**Tanmay Kushwah**

Machine Learning / Python Practice Project

---

## ⭐ Acknowledgement

This project was created as a practical exercise to understand the complete workflow of a basic binary classification problem using Logistic Regression.
