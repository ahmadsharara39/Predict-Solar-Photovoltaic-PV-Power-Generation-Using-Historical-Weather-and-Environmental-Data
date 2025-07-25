# ☀️ Solar System Production Prediction (Machine Learning Pipeline)

This project is a complete machine learning pipeline to predict **solar system energy production** using real-world weather and environmental data. It involves **data preprocessing, visualization, feature selection**, and **regression modeling** using both **Random Forest** and **XGBoost**.

---

## 📥 Getting Started

Clone the repository:

```bash
git clone https://github.com/ahmadsharara39/Predict-Solar-Photovoltaic-PV-Power-Generation-Using-Historical-Weather-and-Environmental-Data
cd Predict-Solar-Photovoltaic-PV-Power-Generation-Using-Historical-Weather-and-Environmental-Data
python "ProjecMLI (1).py"
```

---

## 📂 Dataset

The dataset used is `Solar.csv` and contains hourly data for:

- Solar production values (`SystemProduction`)
- Date and time (`Date-Hour(NMT)`)
- Environmental and weather features like temperature, humidity, wind speed, etc.

---

## 📊 Workflow Summary

1. **Data Cleaning**:  
   - Handle missing values and duplicates
   - Convert `Date-Hour(NMT)` to datetime format  
   - Extract month, day, and season from the timestamp

2. **Feature Engineering**:  
   - Encode season (Winter, Spring, Summer, Autumn)
   - Drop unnecessary columns
   - Scale features using `RobustScaler`

3. **Exploratory Data Analysis (EDA)**:  
   - Histograms and box plots of all features  
   - Scatter plots vs. `SystemProduction`  
   - Correlation heatmap

4. **Feature Selection**:
   - Principal Component Analysis (PCA)
   - Chi-Squared test with discretized target
   - Spearman and Pearson correlation with target variable

5. **Modeling**:
   - Train/Test Split
   - Grid Search with Cross Validation
   - Random Forest Regressor (with hyperparameter tuning)
   - XGBoost Regressor (with hyperparameter tuning)

6. **Evaluation Metrics**:
   - R² Score
   - Mean Absolute Error (MAE)
   - Root Mean Squared Error (RMSE)
   - Normalized RMSE (NRMSE)

---

## 🧪 Model Results

Sample outputs (may vary):

```
Random Forest Best Parameters: {'max_depth': 10, 'min_samples_split': 2, 'n_estimators': 100}
RF R² Score: 0.93
RF MAE: 12.45
RF RMSE: 16.30
RF NRMSE: 0.11

XGBoost Best Parameters: {'learning_rate': 0.1, 'max_depth': 6, 'n_estimators': 200}
XGBoost R² Score: 0.94
XGBoost MAE: 11.67
XGBoost RMSE: 15.45
XGBoost NRMSE: 0.10
```

---

## 📦 Requirements

```bash
pip install pandas matplotlib seaborn scikit-learn xgboost
```

> ⚠️ Make sure `Solar.csv` is in the same directory as your script.

---

## 📁 File Structure

```
📦 Predict-Solar-Photovoltaic-PV-Power-Generation-Using-Historical-Weather-and-Environmental-Data/
 ┣ 📄 Solar.csv                ← Input dataset
 ┣ 📄 ProjecMLI (1).py         ← Main ML script
 ┗ 📄 README.md                ← You're reading it!
```

---

## 🧠 Libraries Used

- `pandas`, `numpy` for data manipulation
- `matplotlib`, `seaborn` for visualization
- `sklearn` for ML preprocessing, feature selection, and models
- `xgboost` for advanced regression
- `scipy` for statistical correlation

---

## ✍️ Author

**Ahmad Sharara**  
GitHub: [@ahmadsharara39](https://github.com/ahmadsharara39)

---

## 📄 License

This project is under the [MIT License](https://choosealicense.com/licenses/mit/).
