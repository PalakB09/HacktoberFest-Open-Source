import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

# Load dataset
cc_apps = pd.read_csv('./dataset/cc_approvals.data', header=None)

# Handling missing values by replacing '?' with NaN
cc_apps = cc_apps.replace('?', np.nan)

# Impute missing values in numeric columns with mean
cc_apps.fillna(cc_apps.mean(), inplace=True)

# Impute missing values in non-numeric columns with most frequent value
for col in cc_apps.columns:
    if cc_apps[col].dtypes == 'object':
        cc_apps = cc_apps.fillna(cc_apps[col].value_counts().index[0])

# Convert non-numeric data to numeric using LabelEncoder
le = LabelEncoder()
for col in cc_apps.columns:
    if cc_apps[col].dtype == 'object':
        cc_apps[col] = le.fit_transform(cc_apps[col])

# Drop unnecessary columns and split data into features and labels
cc_apps = cc_apps.drop([11, 13], axis=1)
cc_apps = cc_apps.values
X, y = cc_apps[:, 0:13], cc_apps[:, 13]

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Rescale the data using MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
rescaledX_train = scaler.fit_transform(X_train)
rescaledX_test = scaler.fit_transform(X_test)

# Instantiate and fit a Logistic Regression model
logreg = LogisticRegression()
logreg.fit(rescaledX_train, y_train)

# Evaluate the model
y_pred = logreg.predict(rescaledX_test)
accuracy = logreg.score(rescaledX_test, y_test)
conf_matrix = confusion_matrix(y_test, y_pred)
print("Accuracy of logistic regression classifier: ", accuracy)
print("Confusion Matrix:")
print(conf_matrix)

# Grid search for parameter tuning
tol = [0.01, 0.001, 0.0001]
max_iter = [100, 150, 200]
param_grid = dict(tol=tol, max_iter=max_iter)
grid_model = GridSearchCV(estimator=logreg, param_grid=param_grid, cv=5)
rescaledX = scaler.fit_transform(X)
grid_model_result = grid_model.fit(rescaledX, y)
best_score, best_params = grid_model_result.best_score_, grid_model_result.best_params_
print("Best: %f using %s" % (best_score, best_params))
