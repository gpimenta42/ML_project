# Claim Injury Type prediction 

## Encoding 

Two encoders were used for categorical features to capture complementary information:
- Frequency Encoding
- Target Ordinal Encoding 

## Feature Selection 

- Mean cross-correlation dropout: drops features with high mean correlation with other features 
- Recursive Feature Elimination (RFE): selects features that maximized validation F1-macro score with Random Forest Classifier 

<img width="1181" height="423" alt="image" src="https://github.com/user-attachments/assets/167f98f8-ddab-46b4-bec3-cde2278412e4" />
*Note: Image typo — Train set is 2/3, not 1/3*

## Results 
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

Each injury type class has its own set of important predictors:

*(See visual for detailed breakdown)*
<img width="958" height="790" alt="image" src="https://github.com/user-attachments/assets/9618c66f-3649-4200-a551-54b1689ec88a" />
