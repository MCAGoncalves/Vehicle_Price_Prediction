import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# Load dataset
database = pd.read_excel('https://github.com/MCAGoncalves/dataset/raw/main/Certificacao_Projetos_LEED_v1.xlsx')

# Exploratory data analysis (EDA)
database.describe()
database.head()

# Data cleaning
database.fillna(database.mean(), inplace=True)

# Define features and target variable
X = database.drop(['Projetos Certificados'], axis=1)
y = database['Projetos Certificados']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.3)

# Initialize regression models
models = [
    GradientBoostingRegressor(learning_rate=0.1, n_estimators=100),
    KNeighborsRegressor(n_neighbors=20),
    SVR(),
    DecisionTreeRegressor(random_state=0),
    LinearRegression()
]

# Store predictions and errors
predictions_list = []
MAE_list = []
MAPE_list = []
MSE_list = []
RMSE_list = []

# Loop through each model
for model in models:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    predictions_list.append(y_pred)
    MAE_list.append(metrics.mean_absolute_error(y_test, y_pred))
    MAPE_list.append(metrics.mean_absolute_percentage_error(y_test, y_pred))
    MSE_list.append(metrics.mean_squared_error(y_test, y_pred))
    RMSE_list.append(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

# Display predictions
predictions_df = pd.DataFrame({
    'GB': predictions_list[0],
    'KNN': predictions_list[1],
    'SVM': predictions_list[2],
    'RF': predictions_list[3],
    'LR': predictions_list[4],
})
print(predictions_df)

# Display errors
errors_df = pd.DataFrame({
    'MAE': MAE_list,
    'MAPE': MAPE_list,
    'MSE': MSE_list,
    'RMSE': RMSE_list
}, index=['Gradient Boosting', 'KNN', 'SVM', 'Random Forest', 'Linear Regression'])
print(errors_df)

# Plot best model predictions
best_model_index = np.argmin(MAPE_list)
y_pred_best = predictions_list[best_model_index]

plt.plot(y_pred_best, color='blue', linestyle='--', label='Prediction')
plt.plot(y_test.values, color='orange', label='Test Set')
plt.legend()
plt.xlabel('Index')
plt.ylabel('Price')
plt.title(f'Best Model: {type(models[best_model_index]).__name__} vs Test Set')
plt.show()
