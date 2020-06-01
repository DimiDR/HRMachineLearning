# Obtain data
import io
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from currency_converter import CurrencyConverter
import pandas as pd
import numpy as np
import pickle

# making data frame from csv file 
df = pd.read_csv("Data/report_Atos_EmployeeData.csv")

#df = pd.read_csv(io.StringIO(data), sep=";")
# rename and delete columns
df = df.rename(columns={"start-date.1":"hire-date", "end-date.2":"termination-date", "end-date":"data-date"})
df.drop(['start-date', 'pay-component', 'end-date.1'], axis=1, inplace=True)
#drop dublicate if rows are identical
df.drop_duplicates( keep = 'first', inplace = True)
#FTE should be lower equal 1
df.drop(df[df.fte > 1].index, inplace=True)
# delete combinations: "inactive user“ + „termination-date = NaN”
df.drop(df[(df.status == "Inactive User") & (pd.isnull(df["termination-date"]))].index, inplace=True)
# delete status = x
df.drop(df[df.status == "x"].index, inplace=True)
# delete data date = NaN
df.drop(df[pd.isnull(df["data-date"])].index, inplace=True)
# overwrite data-date with other date format for sorting
pd.set_option('mode.chained_assignment', None)
table = df["data-date"].str.split(pat = "/")
for i in df["data-date"].index:
    s = pd.Series([table[i][2], table[i][0], table[i][1]]) # year, month, day
    df["data-date"][i] = s.str.cat(sep='.')
#take newest data.date and delete the rest
df.sort_values(by=['user-id', 'data-date'], ascending = [1, 0], inplace = True)
df.drop_duplicates(subset =["user-id"],  keep = 'first', inplace = True)
# delete date-of.birth = NaN
df.drop(df[pd.isnull(df["date-of-birth"])].index, inplace=True)
#Set up todays day for further processing
from datetime import datetime
end_date = datetime.now()
now_year = end_date.year
now_month = end_date.month
now_day = end_date.day

import datetime
df["left_at_age"] = ""
# age when left the company
# if still in the company then age now
table = df["date-of-birth"].str.split(pat = "/")
table_term = df["termination-date"].str.split(pat = "/")
for i in df["date-of-birth"].index:
    birth_date = datetime.date(int(table[i][2]), int(table[i][0]), int(table[i][1])) # year, month, day
    if df["status"][i] == "Active User":
        end_date = datetime.date(now_year, now_month, now_day)
    elif df["status"][i] == "Inactive User":
        end_date = datetime.date(int(table_term[i][2]), int(table_term[i][0]), int(table_term[i][1])) # year, month, day
    time_difference = end_date - birth_date
    age_days = time_difference.days 
    age = int(age_days / 365)
    df["left_at_age"][i] = age

df["years_at_company"] = ""
# years at company
# if employee is still in the company, then years at company untill now
table_hire = df["hire-date"].str.split(pat = "/")
table_term = df["termination-date"].str.split(pat = "/")
for i in df["hire-date"].index:
    hire_date = datetime.date(int(table_hire[i][2]), int(table_hire[i][0]), int(table_hire[i][1])) # year, month, day
    if df["status"][i] == "Inactive User":
        term_date = datetime.date(int(table_term[i][2]), int(table_term[i][0]), int(table_term[i][1])) # year, month, day
    elif df["status"][i] == "Active User":
        # here termination date is now
        term_date = datetime.date(now_year, now_month, now_day)
    years_at_company_difference = term_date - hire_date
    years_at_company_days = years_at_company_difference.days
    years_at_company = int(years_at_company_days / 365)
    df["years_at_company"][i] = years_at_company

# calculate monthly income

df["monthly_income_lc"] = ""
# all month have 4 weeks, all weeks have 5 working days with 8 hours of work
for i in df["frequency"].index:
    if df["frequency"][i] == "ANN":
        #annualy
        df["monthly_income_lc"][i] = df["paycompvalue"][i] / 12
    elif df["frequency"][i] == "BIM" or df["frequency"][i] == "MON" or df["frequency"][i] == "Monthly":
        #monthly
        df["monthly_income_lc"][i] = df["paycompvalue"][i]
    elif df["frequency"][i] == "BWK" or df["frequency"][i] == "SMT":
        #biweekly
        df["monthly_income_lc"][i] = df["paycompvalue"][i] * 2
    elif df["frequency"][i] == "HOURLY":
        #hourly
        df["monthly_income_lc"][i] = df["paycompvalue"][i] * 8 * 5 * 4
    elif df["frequency"][i] == "WKL":
        #weekly
        df["monthly_income_lc"][i] = df["paycompvalue"][i] * 4


# currency conversion to EUR
# columns needed: currency-code, monthly_income_lc
c = CurrencyConverter()
# new column monthly_income
df["monthly_income"] = ""
# Conversion of string to float
df['monthly_income_lc'] = df['monthly_income_lc'].astype(float)
table_currency = df["currency-code"]
table_income = df["monthly_income_lc"]
for i in df["monthly_income_lc"].index:
    if table_currency[i] == 'QAR':
        monthly_income = table_income[i] * 0.255127
    elif table_currency[i] == 'TWD':
        monthly_income = table_income[i] * 0.0309717
    elif table_currency[i] == 'SAR':
        monthly_income = table_income[i] * 0.247634
    elif table_currency[i] == 'AED':
        monthly_income = table_income[i] * 0.25286
    elif table_currency[i] == 'ARS':
        monthly_income = table_income[i] * 0.01382
    elif table_currency[i] == 'CLP':
        monthly_income = table_income[i] * 0.00110754
    elif table_currency[i] == 'COP':
        monthly_income = table_income[i] * 0.000236016
    else:
        monthly_income = c.convert(table_income[i], table_currency[i] ) #Default target currency is EUR
    df["monthly_income"][i] = monthly_income

# copy df to df_output with only the needed columns
# user-id, 
# left_at_age, years_at_company, monthly_income
# Columns: status AS 1 = Active and 0 = Inactive

df_output = df.copy()
df_output.drop(['monthly_income_lc', 'username', 'currency-code', 'paycompvalue', 'data-date', 'frequency', 'date-of-birth','hire-date', 'fte', 'termination-date'], axis=1, inplace=True)
df_output.status.replace(['Active User', 'Inactive User'], [1, 0], inplace=True)
df_output['left_at_age'] = pd.to_numeric(df_output['left_at_age'])
df_output['years_at_company'] = pd.to_numeric(df_output['years_at_company'])
df_output['monthly_income'] = pd.to_numeric(df_output['monthly_income'])

# Machine Learning +++++++++++++++++
df_HR = df_output.copy()
# assign the target to a new dataframe and convert it to a numerical feature
#df_target = df_HR[['Attrition']].copy()
target = df_HR['status'].copy()
# let's remove the target feature and redundant features from the dataset
df_HR.drop(['lastName', 'firstName', 'status',
            'user-id'], axis=1, inplace=True)

# Since we have class imbalance (i.e. more employees with turnover=0 than turnover=1)
# let's use stratify=y to maintain the same ratio as in the training dataset when splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(df_HR,
                                                    target,
                                                    test_size=0.25,
                                                    random_state=7,
                                                    stratify=target)  
print("Number transactions X_train dataset: ", X_train.shape)
print("Number transactions y_train dataset: ", y_train.shape)
print("Number transactions X_test dataset: ", X_test.shape)
print("Number transactions y_test dataset: ", y_test.shape)

# random forest+++++++++++++++++++++++++++++
rf_classifier = RandomForestClassifier(class_weight = "balanced",
                                        random_state=7)
param_grid = {'n_estimators': [50, 75, 100, 125, 150, 175],
                'min_samples_split':[2,4,6,8,10],
                'min_samples_leaf': [1, 2, 3, 4],
                'max_depth': [5, 10, 15, 20, 25]}

grid_obj = GridSearchCV(rf_classifier,
                        iid=True,
                        return_train_score=True,
                        param_grid=param_grid,
                        scoring='roc_auc',
                        cv=10)

grid_fit = grid_obj.fit(X_train, y_train)
rf_opt = grid_fit.best_estimator_





# Create & send the model blob to the output port, The Artifact Producer
# operator will use this to persist the model and create an artifact ID
import pickle
#model = pickle.dumps(rf_opt)
#save model
filename = "randomForest_SF_PY"
pickle.dump(rf_opt, open(filename, 'wb'))