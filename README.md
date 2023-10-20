#### <a name="_psu2m8wnrw18"></a>Group members: Shuxian Chen, Haiming Li, Chenyan Wen
### <a name="_b5z6khdpzs3v"></a>**Dataset Used: 2017-March 2020 Pre-Pandemic Demographics Data**
### <a name="_q7t6o3jcnv4e"></a>Cholesterol - Total ([P_TCHOL](https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/P_TCHOL.htm))
- LBXTC - total cholesterol level in mg/dL (used as the response variable)
### <a name="_xvp7zz1th6q5"></a>Body Measures ([P_BMX](https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/P_BMX.htm))
- BMXWT - weight in kg
- BMXHT - height in cm [not selected by bic]
- BMXBMI - body mass index, a nonlinear combination of weight & height
### <a name="_6cx3w7uvhmos"></a>Lead, Cadmium, Total Mercury, Selenium, & Manganese - Blood ([P_PBCD](https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/P_PBCD.htm))
- LBXBPB - Blood lead concentration in ug/dL
- LBXBCD - Blood cadmium concentration in ug/dL
- LBXTHG - Blood mercury concentration in ug/dL
- LBXBSE - Blood selenium concentration in ug/dL
- LBXBMN - Blood manganese concentration in ug/dL
### <a name="_1fp7dt3muk2h"></a>Demographic Variables and Sample Weights ([P_DEMO](https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/P_DEMO.htm))
- RIDAGEYR - age in years
- INDFMPIR - Ratio of family income to poverty; lower means poorer
- RIDRETH3 - Race/Hispanic origin w/ NH Asian [not selected by bic]
  - 1: Mexican American
  - 2: Other Hispanic
  - 3: Non-Hispanic White
  - 4: Non-Hispanic Black
  - 6: Non-Hispanic Asian
  - 7: Other Race - Including Multi-Racial
- ### <a name="_lt5m2qrwwjar"></a>RIAGENDR - gender
  - 1: male
  - 2: female

**Basic Setup**

- Do binary classification
  - cholesterol level (LBXTC) > 200 mg/dL means unhealthy
  - 1 if LBXTC > 200, 0 otherwise
  - All other variables mentioned above are predictors
- All four datasets are joined together with Respondent Sequence Number (SEQN)

**Questions Investigated**

- Try to create models that predicate cholesterol risk using selected features
  - Try SVM and XGB, fine-tune with CV
  - Apply L1/L2 regularization for both models 
- Discover potential relationships between features and cholesterol level
  - Use SHAP value to compare feature importance between SVM and XGB
  - Relate to possible real-life causes
