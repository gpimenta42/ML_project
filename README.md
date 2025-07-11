# Claim Injury Type prediction 


## Feature Selection 

<img width="1181" height="423" alt="image" src="https://github.com/user-attachments/assets/167f98f8-ddab-46b4-bec3-cde2278412e4" />

## Model selection and tuning

<img width="1181" height="441" alt="image" src="https://github.com/user-attachments/assets/629a71e3-c857-4a99-8b8b-f599bf38d967" />


## Results 
| Model                        | Parameters                                                                                                                                    | Train Score | Validation Score |
| ---------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- | ----------- | ---------------- |
| **CatBoost**                 | {'iterations': 1000, 'depth': 6, 'boosting\_type': 'Ordered', 'auto\_class\_weights': 'SqrtBalanced', 'loss\_function': 'MultiClassOneVsAll'} | 0.560       | 0.487            |
| **Neural Network**           | {'hidden\_layer\_sizes': (25, 8), 'learning\_rate\_init': 0.01}                                                                               | 0.452       | 0.447            |
| **Random Forest**            | {'max\_depth': 6, 'class\_weight': 'balanced'}                                                                                                | 0.369       | 0.405            |
| **Logistic Regression**      | {'C': 1, 'solver': 'lbfgs', 'class\_weight': None}                                                                                            | 0.426       | 0.395            |
| **Gaussian NB**              | {'var\_smoothing': 0.1}                                                                                                                       | 0.351       | 0.317            |
| **OVR Random Forest**        | {'max\_depth': 6, 'class\_weight': 'balanced'}                                                                                                | 0.332       | 0.314            |
| **Stacking (CatBoost, MLP)** | {'C': 1, 'multi\_class': 'ovr', 'class\_weight': 'balanced'}                                                                                  | 0.456       | 0.432            |
| **Stacking (CatBoost, NB)**  | {'C': 1, 'multi\_class': 'ovr', 'class\_weight': 'balanced'}                                                                                  | 0.455       | 0.434            |


## Final results on Kaggle Competition

**F1-macro: 0.408** (2nd place on the [leaderboard](https://www.kaggle.com/competitions/to-grant-or-not-to-grant/leaderboard))

## EXTRA: Feature importances with SHAP 

<img width="958" height="790" alt="image" src="https://github.com/user-attachments/assets/9618c66f-3649-4200-a551-54b1689ec88a" />
