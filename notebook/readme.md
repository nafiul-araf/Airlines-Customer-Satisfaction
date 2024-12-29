# Airlines Customer Satisfaction

### Steps Followed:

1. **Data Understanding and Exploration**  
   - Analyzed the structure of the dataset.
   - Explored the relationship between features and the target variable.

2. **Data Preprocessing**  
   - Handled missing values and outliers.
   - Encoded categorical variables using appropriate encoding techniques.
   - Standardized/normalized numerical features for consistency.
   - Split the data into training and testing sets.

3. **Feature Engineering**  
   - Analyzed correlations and relationships among features.  
   - Selected key features relevant to predicting customer satisfaction.  

4. **Baseline Modeling**  
   - Built and evaluated initial models to understand baseline performance.

5. **Model Development and Tuning**  
   - Implemented Logistic Regression, Random Forest, and LightGBM models.
   - Performed hyperparameter tuning for each model using Optuna:
     - **Logistic Regression**: Tuned `C`, `solver`, and `penalty`.
     - **Random Forest**: Tuned `n_estimators`, `max_depth`, `min_samples_split`, and `min_samples_leaf`.
     - **LightGBM**: Tuned `learning_rate`, `num_leaves`, `max_depth`, `min_child_samples`, `subsample`, and `colsample_bytree`.

6. **Model Evaluation**  
   - Compared the performance of all models using metrics such as accuracy, precision, recall, F1-score, and ROC-AUC.
   - Visualized the ROC curve for each model.

7. **Model Selection**  
   - Selected the LightGBM model with optimized hyperparameters as the best-performing model.

8. **Model Saving**  
   - Saved the trained model using `joblib` for future use.

---

## **Data Loading and Exploration**

This section outlines the process of loading the dataset, performing data cleaning, and conducting exploratory data analysis (EDA) to better understand the structure and distribution of the data.

### **1. Loading the Dataset**
The dataset was loaded using `pandas`. Here's a preview of the initial dataset:

```python
df = pd.read_csv('path/to/dataset/airlines.csv')
df.head()
```

---

### **2. Data Cleaning**

#### **2.1 Handling Missing Values**
To ensure data integrity, missing values were handled by dropping rows with null entries. The dataset's size was reduced accordingly:

```python
df.dropna(inplace=True)
df.reset_index(drop=True)
```

#### **2.2 Removing Duplicates**
Duplicate records were checked and removed to avoid redundancy:

```python
df.duplicated().sum()
```

#### **2.3 Verifying Data Types**
The data types of each column were inspected to ensure compatibility with analysis and modeling:

```python
df.dtypes
```

---

### **3. Exploratory Data Analysis (EDA)**

#### **3.1 Summary Statistics**
The statistical summary of numerical columns provided insights into the central tendency and variability of key features:

```python
df.describe(include=np.number).transpose().round(2)
```

#### **3.2 Visualizing Data Distribution**
Key numerical attributes such as `Age`, `Flight Distance`, `Departure Delay in Minutes`, and `Arrival Delay in Minutes` were visualized using histograms to understand their distributions:

```python
cols_to_viz = ['Age', 'Flight Distance', 'Departure Delay in Minutes', 'Arrival Delay in Minutes']
# Code to create histograms...
```

**Observation:**
- Age and flight distances showed right-skewed distributions.
- Departure and arrival delays exhibited significant right tails, indicating frequent delays.

![Histogram Visuals](path/to/histogram.png)

---

### **4. Addressing Skewness**

#### **4.1 Identifying Skewed Features**
Skewness in numerical features was quantified, and features with skewness > 0.50 were identified for transformation:

```python
skew_df = pd.DataFrame(cols_to_viz, columns=['feature'])
skew_df['skew'] = skew_df['feature'].apply(lambda x: scipy.stats.skew(df[x]))
```

#### **4.2 Applying Log Transformation**
Logarithmic transformation was applied to reduce skewness, making the data more suitable for modeling:

```python
for column in skew_df.query("skewed == True")['feature']:
    df[column] = np.log1p(df[column])
```

---

### **5. Distribution of Categorical Data**
The distribution of categorical features, such as customer type and travel class, was examined to understand the proportion of each category:

```python
cat_cols = df.select_dtypes(include=['object']).columns
for col in cat_cols:
    print(df[col].value_counts(normalize=True))
```

---

### **Key Insights**
- The dataset contains right-skewed numerical features that were normalized using log transformation.
- Categorical variables are well-balanced, with a clear differentiation between customer and travel types.

---

## **Feature Engineering**

Feature engineering plays a crucial role in preparing the dataset for machine learning models. This section details the steps involved in handling the features, reducing multicollinearity, scaling numerical features, and encoding categorical variables.

---

### **1. Splitting Predictors (X) and Target (y)**

The dataset is divided into independent variables (`X`) and the target variable (`y`):

```python
X = df.drop('satisfaction', axis=1)
y = df['satisfaction']
```

---

### **2. Label Encoding**

Since the `Class` column contains ordinal data (`Eco`, `Eco Plus`, `Business`), it is label-encoded to represent the order numerically:

```python
X['Class'] = X['Class'].map({'Eco': 1, 'Eco Plus': 2, 'Business': 3})
```

---

### **3. Splitting the Dataset**

The dataset is split into training (70%) and testing (30%) sets:

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
```

---

### **4. Feature Selection Using Variance Inflation Factor (VIF)**

**Why VIF?**
Variance Inflation Factor helps identify multicollinearity between features. Features with high VIF values (>5) are likely correlated with other features, which can affect model stability.

**Steps:**
1. Calculate the VIF for numerical features.
2. Iteratively remove features with high VIF values.
3. Scale numerical features to address variance.

```python
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Function to calculate VIF
def calculate_vif(data):
    vif_data = pd.DataFrame()
    vif_data['Feature'] = data.columns
    vif_data['VIF'] = [variance_inflation_factor(data.values, i) for i in range(data.shape[1])]
    return vif_data

# Initial VIF calculation
numerical_cols = X_train.select_dtypes(np.number).columns
vif_unscaled = calculate_vif(X_train[numerical_cols])
```

**Action:**
Features such as `Online support`, `Ease of Online booking`, and `Cleanliness` were removed to reduce VIF.

---

### **5. Scaling Numerical Features**

Numerical features were scaled using `MinMaxScaler` to normalize their range between 0 and 1, improving model performance:

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
cols_to_scale = X_train.select_dtypes(np.number).columns

X_train[cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])
X_test[cols_to_scale] = scaler.transform(X_test[cols_to_scale])
```

---

### **6. Encoding Categorical Variables**

Nominal categorical variables were one-hot encoded using `pd.get_dummies`. This process ensures the model interprets categorical features without introducing multicollinearity:

```python
X_train = pd.get_dummies(X_train, drop_first=True)
X_test = pd.get_dummies(X_test, drop_first=True)
```

---

### **7. Final Prepared Data**

The processed datasets for training and testing are now ready for modeling, with the following structure:

- **Numerical features:** Scaled between 0 and 1.
- **Categorical features:** One-hot encoded with k-1 categories.
- **Aligned Features:** Training and testing datasets have the same columns.

```python
X_train.shape, X_test.shape
```

---

### **8. Deployment Preparation**

The transformation pipeline was saved for deployment, ensuring consistency during model inference:

```python
model_data = {
    'model': None,  # Placeholder for the trained model
    'scaler': scaler,
    'features': X_train.columns.tolist(),
    'cols_to_scale': cols_to_scale,
}
```

## **Model Building**

This section details the steps for building and evaluating machine learning models for predicting customer satisfaction. Four baseline models were implemented and evaluated: Logistic Regression, Random Forest, XGBoost, and LightGBM.

---

### **1. Target Label Encoding**

The target variable (`satisfaction`) was mapped to binary labels for machine learning:

- **Satisfied:** 1
- **Dissatisfied:** 0

```python
class_mapping = {'satisfied': 1, 'dissatisfied': 0}
y_train = y_train.map(class_mapping)
y_test = y_test.map(class_mapping)
```

---

### **2. Baseline Models**

A function was created to train, evaluate, and visualize the performance of each model:

```python
def build_evaluate_model(model, model_name, train_x, train_y, test_x, test_y):
    model_ = model.fit(train_x, train_y)
    print(f"Training Score for {model_name}: {model_.score(train_x, train_y)}\n")

    pred = model_.predict(test_x)

    labels = ['satisfied', 'dissatisfied']
    print(f"Accuracy Score for {model_name}: {accuracy_score(test_y, pred)}\n")
    print(f"Classification Report for {model_name}:\n{classification_report(test_y, pred, target_names=labels)}\n")

    sns.heatmap(confusion_matrix(test_y, pred), annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f"Confusion Matrix for {model_name}")
    plt.show()

    fpr, tpr, thresholds = roc_curve(test_y, model_.predict_proba(test_x)[:, 1])
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc_score(test_y, model_.predict_proba(test_x)[:, 1]):.2f})")
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()

    return model_
```

---

### **3. Logistic Regression**

- **Training Accuracy:** ~76.34%
- **Test Accuracy:** ~75.91%
- **AUC Score:** ~0.84

**Observations:**  
Logistic Regression demonstrated good generalization, with balanced precision, recall, and F1-scores for both classes.

---

### **4. Random Forest**

- **Training Accuracy:** ~100%
- **Test Accuracy:** ~85.04%
- **AUC Score:** ~0.93

**Observations:**  
The model showed high performance on the test data but perfect training accuracy indicates potential overfitting. Despite this, it generalized well.

---

### **5. XGBoost**

- **Training Accuracy:** ~91.01%
- **Test Accuracy:** ~85.65%
- **AUC Score:** ~0.93

**Observations:**  
XGBoost exhibited slightly better test performance than Random Forest, maintaining strong discriminatory power and balanced metrics across classes.

---

### **6. LightGBM**

- **Training Accuracy:** ~87.79%
- **Test Accuracy:** ~85.82%
- **AUC Score:** ~0.93

**Observations:**  
LightGBM achieved the highest test accuracy among the models, with balanced precision and recall, suggesting strong generalization.

---

### **7. Model Comparison**

| Model                  | Training Accuracy | Test Accuracy | AUC   |
|------------------------|-------------------|---------------|-------|
| Logistic Regression    | ~76.34%          | ~75.91%       | ~0.84 |
| Random Forest          | ~100%            | ~85.04%       | ~0.93 |
| XGBoost                | ~91.01%          | ~85.65%       | ~0.93 |
| LightGBM               | ~87.79%          | ~85.82%       | ~0.93 |

---

### **Findings**

All models performed well, with **LightGBM** achieving the highest test accuracy and competitive AUC, making it a strong candidate for deployment.

---

### **Hyperparameter Tuning for Machine Learning Models**
This section explains the use of **Optuna**, a powerful framework for hyperparameter optimization, to tune three machine learning models: Logistic Regression, Random Forest, and LightGBM. The tuned models are evaluated on training and test data, with results compared using metrics such as accuracy, classification reports, and ROC-AUC curves.

---

#### **Installing Optuna**
```python
!pip install optuna
```
This installs the Optuna library, used for optimizing hyperparameters.

---

### **Logistic Regression Hyperparameter Tuning**

#### **Objective Function**
The `objective_logreg` function defines the hyperparameter search space:
- `C`: Regularization strength, selected from a log-uniform distribution.
- `solver`: Algorithm to solve optimization, chosen between `"liblinear"` and `"saga"`.
- `penalty`: Regularization type (`l1` or `l2`), determined based on solver choice.

```python
def objective_logreg(trial):
    # Hyperparameter search space
    C = trial.suggest_loguniform("C", 1e-3, 1e3)
    solver = trial.suggest_categorical("solver", ["liblinear", "saga"])
    penalty = trial.suggest_categorical("penalty", ["l1", "l2"]) if solver != "saga" else "l2"

    # Model definition
    model = LogisticRegression(C=C, solver=solver, penalty=penalty, random_state=42, max_iter=1000)

    # Cross-validation
    score = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy').mean()
    return score
```

#### **Optimization Process**
```python
study_logreg = optuna.create_study(direction="maximize")
study_logreg.optimize(objective_logreg, n_trials=100)

print("Best hyperparameters for Logistic Regression:", study_logreg.best_params)
print("Best accuracy for Logistic Regression:", study_logreg.best_value)
```
- **Results**: 
  - Best hyperparameters are displayed.
  - Best cross-validation accuracy: 0.7634.

#### **Evaluation**
The optimized model is built, evaluated, and results are analyzed:
```python
lr_model_optuna = build_evaluate_model(
    model=LogisticRegression(**study_logreg.best_params, random_state=42, max_iter=1000),
    model_name="Logistic Regression with Optuna",
    train_x=X_train, train_y=y_train, test_x=X_test, test_y=y_test
)
```

---

### **Random Forest Hyperparameter Tuning**

#### **Objective Function**
Defines search space for Random Forest parameters:
- `n_estimators`: Number of trees in the forest.
- `max_depth`: Maximum tree depth.
- `min_samples_split`: Minimum samples for node splitting.
- `min_samples_leaf`: Minimum samples in a leaf node.

```python
def objective_rf(trial):
    n_estimators = trial.suggest_int("n_estimators", 50, 500)
    max_depth = trial.suggest_int("max_depth", 5, 30)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10)

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42, n_jobs=-1
    )

    score = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy').mean()
    return score
```

#### **Optimization Process**
```python
study_rf = optuna.create_study(direction="maximize")
study_rf.optimize(objective_rf, n_trials=50)

print("Best hyperparameters for Random Forest:", study_rf.best_params)
print("Best accuracy for Random Forest:", study_rf.best_value)
```
- **Results**: 
  - Best hyperparameters are displayed.
  - Best cross-validation accuracy: 0.8615.

#### **Evaluation**
```python
rf_model_optuna = build_evaluate_model(
    model=RandomForestClassifier(**study_rf.best_params, random_state=42),
    model_name="Random Forest with Optuna",
    train_x=X_train, train_y=y_train, test_x=X_test, test_y=y_test
)
```

---

### **LightGBM Hyperparameter Tuning**

#### **Objective Function**
Defines search space for LightGBM parameters:
- `learning_rate`: Step size for updating weights.
- `num_leaves`: Maximum number of leaves in a tree.
- `max_depth`: Maximum tree depth.
- `min_child_samples`: Minimum samples required to create a new node.
- `subsample`: Proportion of data used for training each tree.
- `colsample_bytree`: Proportion of features used per tree.
- `n_estimators`: Number of boosting iterations.

```python
def objective_lgb(trial):
    learning_rate = trial.suggest_loguniform("learning_rate", 0.01, 0.5)
    num_leaves = trial.suggest_int("num_leaves", 10, 300)
    max_depth = trial.suggest_int("max_depth", -1, 20)
    min_child_samples = trial.suggest_int("min_child_samples", 5, 100)
    subsample = trial.suggest_uniform("subsample", 0.5, 1.0)
    colsample_bytree = trial.suggest_uniform("colsample_bytree", 0.5, 1.0)
    n_estimators = trial.suggest_int("n_estimators", 50, 700)

    model = LGBMClassifier(
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        max_depth=max_depth,
        min_child_samples=min_child_samples,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        n_estimators=n_estimators,
        random_state=42
    )

    score = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy").mean()
    return score
```

#### **Optimization Process**
```python
study_lgb = optuna.create_study(direction="maximize")
study_lgb.optimize(objective_lgb, n_trials=100)

print("Best hyperparameters for LightGBM:", study_lgb.best_params)
print("Best accuracy for LightGBM:", study_lgb.best_value)
```
- **Results**: 
  - Best hyperparameters are displayed.
  - Best cross-validation accuracy: 0.8687.

#### **Evaluation**
```python
lgb_model_optuna = build_evaluate_model(
    model=LGBMClassifier(**study_lgb.best_params, random_state=42),
    model_name="Light Gradient Boosting with Optuna",
    train_x=X_train, train_y=y_train, test_x=X_test, test_y=y_test
)
```

---

### **Final Notes**

- **Performance Summary**:
  - Logistic Regression: Accuracy = 0.7634
  - Random Forest: Accuracy = 0.8615
  - LightGBM: Accuracy = 0.8687

- **Saving the Model**:
  ```python
  import joblib
  joblib.dump(model_data, "model_data.pkl")
  ```
