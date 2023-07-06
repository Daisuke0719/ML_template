import mlflow

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes

db = load_diabetes()
X_train, X_test, y_train, y_test = train_test_split(db.data, db.target)

logged_model = 'runs:/a43d63b239684834bda5c3960f8c68d7/model'

model = mlflow.sklearn.load_model(logged_model)
predictions = model.predict(X_test)
print(predictions)