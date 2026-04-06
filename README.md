# Day-88-Model-tuning

###  Overview

**Model Tuning** is the process of improving a machine learning model by **adjusting its hyperparameters** to achieve better performance.

It helps in making models more **accurate, stable, and reliable**.

---

##  What is Model Tuning?

After training a model, we fine-tune it by:

* Adjusting hyperparameters
* Testing different configurations
* Selecting the best-performing model

---

##  Common Hyperparameters

Examples:

* Learning rate
* Number of estimators (Random Forest)
* Depth of tree
* Value of K (KNN)

---

##  Tuning Techniques

### 1️⃣ Grid Search

Tries all possible parameter combinations.

```python id="7g4h3k"
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()

params = {
    "n_estimators": [10, 50],
    "max_depth": [2, 4]
}

grid = GridSearchCV(model, params, cv=3)
grid.fit(X_train, y_train)

print(grid.best_params_)
```

---

### 2️⃣ Random Search

Randomly selects parameter combinations.

```python id="2j8l5p"
from sklearn.model_selection import RandomizedSearchCV

random = RandomizedSearchCV(model, params, n_iter=5, cv=3)
random.fit(X_train, y_train)
```

---

### 3️⃣ Cross-Validation

Evaluates model performance on different subsets of data.

---

##  Benefits

- Improved model accuracy
- Reduced overfitting
- Better generalization

---

##  Challenges

* Time-consuming
* Requires computational power

---

##  Key Takeaways

- Model tuning improves performance
- Hyperparameters control model behavior
- Grid Search and Random Search are widely used

