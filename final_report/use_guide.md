
## **Computing Environment**
- Python version $\ge$ 3.10
- R version $\ge$ 3.6
- Having a virtual environment for python at current directory is ideal
    - Create virtual environment: `python3 -m venv env`
    - Activate virtual environment: `source env/bin/activate`

## **Shell Script Automation**
- Usage: `./run.sh [options] [arguments]`
    - Note: make sure [run.sh](./run.sh) is an executable (use `chmod +x run.sh`)
- Options:
    - `-r`: an email address, optional
        - Used for execution notification in cloud environment
        - Currently not usable, due to repository being public
    - `-m`: execution mode, required
        - `setup`: install all required libraries & preprocess data
        - `svm_full`: train a SVC with all data
        - `svm_group`: train a SVC for each of the three groups
        - `xgb_full`: train a XGBClassifier with all data
        - `xgb_group`:  train a XGBClassifier for each of the three groups
        - `cleanup`: delete all cache files used in training
        - `visualize`: produce SHAP value graphs & performance plot for all models
- Example:
    - Train SVM with all data and without notification: `./run.sh -m svm_full`
    - Train SVM for each group and with notification: `./run.sh -m svm_group -r example@domain.com`

## **Description for Directories & Scripts**
### [datasets](./datasets/)
- Contains all raw & processed datasets
### [figures](./figures)
- Contain performance plot for all model
    - Cross validated AUC score
- Contain SHAP plots for all model
    - bar chart
    - bee swarm plot
    - heat map
### [models](./models)
- Contains all trained model object
- Models can be loaded using `joblib` library in python
### [perf_log](./perf_log)
- Contain performance log for all model
    - Cross validated F1 score
    - Cross validated AUC score
    - Cross validated Balanced Accuracy
### [*create_data.R*](./create_data.R)
- Read all `.XPT` file and merge them into one dataframe
- Remove all NULL values from the dataset
- Save processed dataframes as `.csv` files in datasets folder
### [*train_svm.py*](./train_svm.py)
- Contain functions that train SVC for given datasets
- Automatically save trained model and log performance
### [*train_xgb.py*](./train_xgb.py)
- Contain functions that train XGBClassifier for given datasets
- Automatically save trained model and log performance
### [*visualize_model.py*](./visualize_model.py)
- Contain all functions that make plots for a given model
- Save all plots in figures folder
### [*main.py*](./main.py)
- Script that takes command line input and execute code accordingly
- Only used by `run.sh` for automation
### [*send_mail.py*](./send_mail.py)
- Use a SMTP server to send email notification after script execution is complete
- Currently not usable, server credential is redacted due to the repository being public