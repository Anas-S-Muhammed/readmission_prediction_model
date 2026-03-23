# 🏥 Diabetes Prediction — WiDS 2021

A machine learning project to predict whether an ICU patient 
has diabetes using the WiDS Datathon 2021 dataset.

## 📁 Dataset
- **Source:** WiDS Datathon 2021 (Kaggle)
- **Size:** 130,157 patients, 181 features
- **Target:** `diabetes_mellitus` (0 = No, 1 = Yes)

## 🔧 Tools
- Python, Pandas, NumPy, Scikit-learn, Matplotlib

## 📋 Project Phases
1. Data loading and exploration
2. Data cleaning and preprocessing
3. Exploratory data analysis
4. Feature engineering
5. Model training
6. Model evaluation
7. Conclusion

## 📊 Results
| Model | F1 Score | Accuracy |
|---|---|---|
| Logistic Regression | 0.39 | 0.81 |
| Decision Tree | 0.43 | 0.75 |
| Random Forest | 0.46 | 0.82 |

**Winner: Random Forest 🏆**

## ⚠️ Challenges
- Highly imbalanced dataset (75% non-diabetic vs 25% diabetic)
- 170+ features required careful preprocessing
- Models struggled to detect diabetic patients (low recall)

## 🚀 Future Improvements
- Handle class imbalance using SMOTE or class_weight
- Tune hyperparameters with GridSearchCV
- Try more powerful models (XGBoost, LightGBM)
- Add more feature engineering
- Build a simple prediction interface

## 📝 Lessons Learned
- Always check data shape after every cleaning step
- Accuracy is misleading on imbalanced datasets
- Feature engineering can significantly improve model performance