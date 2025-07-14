# Claim Injury Type prediction 

[Dataset](https://www.kaggle.com/competitions/to-grant-or-not-to-grant/data) 

## Preprocessing 
### Feature Engineering
### Data preparation  
- Missing data: Mean or outlier imputation 
- Outliers: Domain-specific  
### Categorical encoding 
Two encoders were used for non-target categorical features to capture complementary information:
- **Frequency Encoding**
- **Target Ordinal Encoding**: a variant of target encoding, where categories are assigned integer values based on the ordered mean of the target variable 
- **Ordinal Encoding**: for target
  
## Feature Selection 

- **Mean cross-correlation dropout**: drops features with high mean correlation with other features 
- **Recursive Feature Elimination (RFE)**: selects features that maximized validation F1-macro score with Random Forest Classifier 

<img width="1181" height="423" alt="image" src="https://github.com/user-attachments/assets/167f98f8-ddab-46b4-bec3-cde2278412e4" />

*Note: Image typo — Train set is 2/3, not 1/3*

## Model selection and hyperparameter tuning 
| **Model**                | **Parameters**                                                                   | **Train Score** | **Mean Validation Score** |
| ------------------------ | -------------------------------------------------------------------------------- | --------------- | -------------------- |
| **CatBoost**             | `depth=6`, `auto_class_weights=SqrtBalanced`, `loss_function=MultiClassOneVsAll` | **0.560**       | **0.487**            |
| **Neural Network**       | `hidden_layer_sizes=(25, 8)`, `learning_rate_init=0.01`                          | 0.452           | 0.447                |
| **Random Forest**        | `max_depth=6`, `class_weight=balanced`                                           | 0.429           | 0.405                |
| **Logistic Regression**  | `C=1`, `solver=lbfgs`, `class_weight=None`                                       | 0.426           | 0.395                |
| **Gaussian Naive Bayes** | `var_smoothing=0.1`                                                              | 0.351           | 0.317                |
| **OVR Random Forest**    | `max_depth=6`, `class_weight=balanced`                                           | 0.332           | 0.314                |

> **Best Validation Score:** CatBoost (0.487)

## Final Kaggle Submission

* **F1-macro score:** **0.408**
* **Leaderboard Position:** 🥈 2nd place
  [View Leaderboard](https://www.kaggle.com/competitions/to-grant-or-not-to-grant/leaderboard)


## SHAP Feature Importances

Most Important Features Influencing Claim Severity:
- Wage: Higher wages are generally associated with higher claim values.
- Days to First Hearing: A shorter time to the first hearing often correlates with more severe claims.
- Attorney Involvement: Claims involving attorneys tend to be of higher severity.
- IME-3 Count: A higher number of IME-3 evaluations is associated with increased claim severity.
- Missing C-3 Form: The absence of the C-3 form is often linked to fatal claims, as deceased individuals cannot complete the form.
- Part of Body Injured: Certain body parts are more commonly associated with fatal or severe injuries.
- Accident–Assembly Gap (Days): A long delay between the accident and the corresponding result in canceled claims.
  
*(See visual for detailed breakdown)*
<img width="1446" height="1182" alt="Screenshot 2025-07-14 181201" src="https://github.com/user-attachments/assets/b48d47d3-0b28-4195-845d-92ae26ba21d4" />

