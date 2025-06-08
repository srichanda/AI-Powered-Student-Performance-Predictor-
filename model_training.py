import joblib
from sklearn.linear_model import LogisticRegression
import numpy as np

# Dummy training data
X = np.array([[0.9, 8, 7], [0.6, 5, 5], [0.3, 3, 2], [0.8, 7, 6]])
y = [1, 1, 0, 1]  # 1: Good Performance, 0: At Risk

model = LogisticRegression()
model.fit(X, y)

joblib.dump(model, 'model.pkl')
