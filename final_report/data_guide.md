## **2017-March 2020 Pre-Pandemic Demographics Data**
### Cholesterol - Total ([P_TCHOL](https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/P_TCHOL.htm))
- LBXTC - total cholesterol level in mg/dL (used as the response variable)
### Body Measures ([P_BMX](https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/P_BMX.htm))
- BMXWT - weight in kg
- BMXBMI - body mass index, a nonlinear combination of weight & height
### Lead, Cadmium, Total Mercury, Selenium, & Manganese - Blood ([P_PBCD](https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/P_PBCD.htm))
- LBXBPB - Blood lead concentration in ug/dL
- LBXBCD - Blood cadmium concentration in ug/dL
- LBXTHG - Blood mercury concentration in ug/dL
- LBXBSE - Blood selenium concentration in ug/dL
- LBXBMN - Blood manganese concentration in ug/dL
### Demographic Variables and Sample Weights ([P_DEMO](https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/P_DEMO.htm))
- RIDAGEYR - age in years
- INDFMPIR - Ratio of family income to poverty; lower means poorer
- RIAGENDR - gender
  - 1: male
  - 2: female

## **Basic Setup**
- Do binary classification
  - cholesterol level (LBXTC) > 200 mg/dL means unhealthy
  - 1 if LBXTC > 200, 0 otherwise
  - All other variables mentioned above are predictors
- All four datasets are joined together with Respondent Sequence Number (SEQN)
- Total data points are 9286

## **Questions Investigated**
- Discover potential relationships between features and cholesterol level
    - Treat age as a predictor in this question
    - Use SHAP value to compare feature importance between SVM and XGB
    - Relate to possible explanations in real-life
- Discover how might feature importance differ across age groups (based on age in year)
  - Young group: \[0, 30\)
  - Middle group: \[30, 60\)
  - Old group: \[60, 80]
  - Only use the better model structure from previous question