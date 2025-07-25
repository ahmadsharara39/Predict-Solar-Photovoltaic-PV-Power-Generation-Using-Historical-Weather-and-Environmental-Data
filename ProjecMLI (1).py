import pandas as pd
data= pd.read_csv('Solar.csv')
data.head()

data.isnull().sum()

data.duplicated().sum()

data['Date-Hour(NMT)'] = pd.to_datetime(data['Date-Hour(NMT)'], format="%d.%m.%Y-%H:%M")

data.info()

data['Month'] = data['Date-Hour(NMT)'].dt.month
data['Day'] = data['Date-Hour(NMT)'].dt.day

def get_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Autumn'

data['Season'] = data['Month'].apply(get_season)

data.columns

from sklearn.preprocessing import OrdinalEncoder
OE = OrdinalEncoder()
data['Season'] = OE.fit_transform(data[["Season"]])
data.info()

data.drop(['Date-Hour(NMT)'], axis=1, inplace=True)

Y = data['SystemProduction']
X = data.drop(['SystemProduction'], axis=1)
import matplotlib.pyplot as plt
X.hist(figsize=(12, 8), bins=30, edgecolor='black')
plt.suptitle("Histogram of Features", fontsize=16)
plt.show()

plt.figure(figsize=(12, 6))
for x in X:
  plt.boxplot(X[x])
  plt.title(x)
  plt.show()

from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
X_Scaled= scaler.fit_transform(X)

import seaborn as sns
for x in X:
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=X , x=x , y=Y)
    plt.title(f"{x} vs. System Production")
    plt.xlabel(x)
    plt.ylabel("System Production")
    plt.show()
    
import seaborn as sns
for x in X:
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=X , x=x , y=Y)
    plt.title(f"{x} vs. System Production")
    plt.xlabel(x)
    plt.ylabel("System Production")
    plt.show()

corr_matrix = data.corr()
plt.figure(figsize=(10, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5, fmt=".2f")
plt.title("Correlation Matrix Heatmap")
plt.show()

from sklearn.feature_selection import chi2,SelectKBest
import numpy as np
from sklearn.decomposition import PCA
pca = PCA(n_components=5)
X_pca = pca.fit_transform(X_Scaled)

plt.figure(figsize=(8, 5))
plt.plot(range(1, 6), np.cumsum(pca.explained_variance_ratio_), marker='o', linestyle='--')
plt.xlabel("Number of Principal Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("PCA Explained Variance")
plt.show()

print(f"Explained Variance Ratio: {pca.explained_variance_ratio_}")

top_features_per_component = []
for i in range(5):
        component = pca.components_[i]
        feature_index = np.argsort(abs(component))[-1]  # Get the most contributing feature index
        top_features_per_component.append(X.columns[feature_index])  # Get feature name

print("Top contributing features for each principal component:")
print(top_features_per_component)

from sklearn.preprocessing import KBinsDiscretizer

discretizer = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='uniform')
Y_discretized = discretizer.fit_transform(data[['SystemProduction']])
chi_selector = SelectKBest(score_func=chi2, k=5)
X_selected = chi_selector.fit_transform(abs(X), Y_discretized)
chi_2_selected_features = X.columns[chi_selector.get_support()]
print("Selected Features:", chi_2_selected_features)

from scipy.stats import pearsonr, spearmanr
correlations = X.corrwith(Y, method='spearman')

selected_features = correlations[abs(correlations) >=0.2].index.tolist()
feature_scores = pd.DataFrame({'Feature': X.columns, 'Spearman Correlation': correlations})
feature_scores = feature_scores.sort_values(by='Spearman Correlation', ascending=False, key=abs)
selected_features_S = feature_scores.head(5)['Feature'].tolist()

print("Spearman Correlation Scores for Features:")
print(feature_scores)
print("------------------------")
print(X[selected_features_S])

correlation_P = X.corrwith(Y, method='pearson')
feature_scores_P = pd.DataFrame({'Feature': X.columns, 'Pearson Correlation': correlation_P})
feature_scores_P = feature_scores.sort_values(by='Pearson Correlation', ascending=False, key=abs)
selected_features_P = feature_scores.head(5)['Feature'].tolist()
print("Pearson Correlation Scores for Features:")
print(feature_scores_P)
print("------------------------")
print(X[selected_features_P])

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

X_train, X_test, y_train, y_test = train_test_split(X_Scaled, Y, test_size=0.2, random_state=42)
rf_params = {
    'n_estimators': [10, 50,100],  # Number of trees
    'max_depth': [None,10, 20],  # Tree depth
    'min_samples_split': [2,5,7]  # Minimum samples to split
}

rf = RandomForestRegressor(random_state=42)
rf_grid = GridSearchCV(rf, rf_params, cv=3)
rf_grid.fit(X_train, y_train)

# Best RF model
rf_best = rf_grid.best_estimator_
y_pred_rf = rf_best.predict(X_test)

# RF Performance
print("\nRandom Forest Best Parameters:", rf_grid.best_params_)

print("RF R² Score:", r2_score(y_test, y_pred_rf))
print("RF MAE:", mean_absolute_error(y_test, y_pred_rf))
print("RF RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_rf)))
print("RF NRMSE:", (np.sqrt(mean_squared_error(y_test, y_pred_rf)))/np.std(y_test))

xgb_params = {
    'n_estimators': [100, 200],
    'max_depth': [3, 6],
    'learning_rate': [0.01, 0.1],
}

xgb = XGBRegressor(random_state=42)
xgb_grid = GridSearchCV(xgb, xgb_params, cv=3, scoring='r2', n_jobs=-1)
xgb_grid.fit(X_train, y_train)

# Best XGBoost model
xgb_best = xgb_grid.best_estimator_
y_pred_xgb = xgb_best.predict(X_test)

# XGBoost Performance
print("\nXGBoost Best Parameters:", xgb_grid.best_params_)
print("XGBoost R² Score:", r2_score(y_test, y_pred_xgb))
print("XGBoost MAE:", mean_absolute_error(y_test, y_pred_xgb))
print("XGBoost RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_xgb)))
print("XGBoost NRMSE:", (np.sqrt(mean_squared_error(y_test, y_pred_xgb)))/np.std(y_test))

