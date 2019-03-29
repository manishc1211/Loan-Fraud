import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import xgboost

classifier = KNeighborsClassifier()
print('Loading train data..')
Train = pd.read_csv('train_indessa.csv')
print(Train.shape)
print('Loading test data..')
Test = pd.read_csv('train_indessa.csv')
print('Data Loaded.\n')

Train = Train[['member_id', 'loan_amnt', 'funded_amnt', 'addr_state', 'funded_amnt_inv', 'sub_grade', 'term', 'emp_length', 'int_rate', 'annual_inc', 'dti', 'delinq_2yrs', 'inq_last_6mths', 'mths_since_last_delinq', 'mths_since_last_record', 'open_acc', 'pub_rec', 'revol_bal', 'revol_util', 'total_acc', 'total_rec_int', 'total_rec_late_fee', 'recoveries', 'collection_recovery_fee', 'collections_12_mths_ex_med', 'mths_since_last_major_derog', 'last_week_pay', 'acc_now_delinq', 'tot_coll_amt', 'tot_cur_bal', 'total_rev_hi_lim', 'loan_status']]

Test = Test[['member_id', 'loan_amnt', 'funded_amnt', 'addr_state', 'funded_amnt_inv', 'sub_grade', 'term', 'emp_length', 'int_rate', 'annual_inc', 'dti', 'delinq_2yrs', 'inq_last_6mths', 'mths_since_last_delinq', 'mths_since_last_record', 'open_acc', 'pub_rec', 'revol_bal', 'revol_util', 'total_acc', 'total_rec_int', 'total_rec_late_fee', 'recoveries', 'collection_recovery_fee', 'collections_12_mths_ex_med', 'mths_since_last_major_derog', 'last_week_pay', 'acc_now_delinq', 'tot_coll_amt', 'tot_cur_bal', 'total_rev_hi_lim']]


#%%
print("\n* Cleaning dataset.." )

print("\nCleaning sub_grade (converting to numerals)..")

Train['sub_grade'].replace(to_replace='A', value='0', regex=True, inplace=True)
Train['sub_grade'].replace(to_replace='B', value='1', regex=True, inplace=True)
Train['sub_grade'].replace(to_replace='C', value='2', regex=True, inplace=True)
Train['sub_grade'].replace(to_replace='D', value='3', regex=True, inplace=True)
Train['sub_grade'].replace(to_replace='E', value='4', regex=True, inplace=True)
Train['sub_grade'].replace(to_replace='F', value='5', regex=True, inplace=True)
Train['sub_grade'].replace(to_replace='G', value='6', regex=True, inplace=True)
Test['sub_grade'].replace(to_replace='A', value='0', regex=True, inplace=True)
Test['sub_grade'].replace(to_replace='B', value='1', regex=True, inplace=True)
Test['sub_grade'].replace(to_replace='C', value='2', regex=True, inplace=True)
Test['sub_grade'].replace(to_replace='D', value='3', regex=True, inplace=True)
Test['sub_grade'].replace(to_replace='E', value='4', regex=True, inplace=True)
Test['sub_grade'].replace(to_replace='F', value='5', regex=True, inplace=True)
Test['sub_grade'].replace(to_replace='G', value='6', regex=True, inplace=True)
Train['sub_grade'] = pd.to_numeric(Train['sub_grade'], errors='coerce')
Test['sub_grade'] = pd.to_numeric(Test['sub_grade'], errors='coerce')
print('sub_grade Done.')

print('\nCleaning term (removing \'months\' part)..')
Test['term'].replace(to_replace = ' months', value = '', regex=True, inplace = True)
Test['term'].replace(to_replace = ' months', value = '', regex=True, inplace = True)
Train['term'] = pd.to_numeric(Train['term'], errors='coerce')
Test['term'] = pd.to_numeric(Test['term'], errors='coerce')
print('term Done.')

print('\nCleaning emp_length..')
Train['emp_length'].replace('n/a', '0', inplace=True)
Train['emp_length'].replace(to_replace='\+ years', value='', regex=True, inplace=True)
Train['emp_length'].replace(to_replace=' years', value='', regex=True, inplace=True)
Train['emp_length'].replace(to_replace='< 1 year', value='0', regex=True, inplace=True)
Train['emp_length'].replace(to_replace=' year', value='', regex=True, inplace=True)
Test['emp_length'].replace('n/a', '0', inplace=True)
Test['emp_length'].replace(to_replace='\+ years', value='', regex=True, inplace=True)
Test['emp_length'].replace(to_replace=' years', value='', regex=True, inplace=True)
Test['emp_length'].replace(to_replace='< 1 year', value='0', regex=True, inplace=True)
Test['emp_length'].replace(to_replace=' year', value='', regex=True, inplace=True)
Train['emp_length'] = pd.to_numeric(Train['emp_length'], errors='coerce')
Test['emp_length'] = pd.to_numeric(Test['emp_length'], errors='coerce')
print('emp_length Done.')

print('\nCleaning last_week_pay..')
Train['last_week_pay'].replace(to_replace='NA', value='', regex=True, inplace=True)
Test['last_week_pay'].replace(to_replace='NA', value='', regex=True, inplace=True)
Train['last_week_pay'].replace(to_replace='th week', value='', regex=True, inplace=True)
Test['last_week_pay'].replace(to_replace='th week', value='', regex=True, inplace=True)
Train['last_week_pay'] = pd.to_numeric(Train['last_week_pay'], errors='coerce')
Test['last_week_pay'] = pd.to_numeric(Test['last_week_pay'], errors='coerce')
print('last_week_pay Done.')


#%%
print('Filling missing values')
colset1 = ['term', 'loan_amnt', 'funded_amnt', 'last_week_pay', 'int_rate', 'sub_grade', 'annual_inc', 'dti', 'mths_since_last_delinq', 'mths_since_last_record', 'open_acc', 'revol_bal', 'revol_util', 'total_acc', 'total_rec_int', 'mths_since_last_major_derog', 'tot_coll_amt', 'tot_cur_bal', 'total_rev_hi_lim', 'emp_length']
print('\nFilling with medians')
for col in colset1:
    print('Filling ',col,'..',sep='')
    Train[col].fillna(Train[col].median(), inplace=True)
    Test[col].fillna(Test[col].median(), inplace=True)

colset2 = ['acc_now_delinq', 'total_rec_late_fee', 'recoveries', 'collection_recovery_fee', 'collections_12_mths_ex_med']
print('\nFilling with Zeroes')
for col in colset2:
    print('Filling ',col,'..',sep='')
    Train[col].fillna(0, inplace=True)
    Test[col].fillna(0, inplace=True)
    
print('\nFilling Done.')
print('\nDataset cleaning Done.')

test_id = pd.DataFrame(Test['member_id'])
train_target = pd.DataFrame(Train['loan_status'])

print('Copying dataset..')
col_copy = ['member_id', 'emp_length', 'loan_amnt', 'funded_amnt', 'funded_amnt_inv', 'sub_grade', 'int_rate', 'annual_inc', 'dti', 'mths_since_last_delinq', 'mths_since_last_record', 'open_acc', 'revol_bal', 'revol_util', 'total_acc', 'total_rec_int', 'total_rec_late_fee', 'mths_since_last_major_derog', 'last_week_pay', 'tot_cur_bal', 'total_rev_hi_lim', 'tot_coll_amt', 'recoveries', 'collection_recovery_fee', 'term', 'acc_now_delinq', 'collections_12_mths_ex_med']
finalTrain = Train[col_copy]
finalTest = Test[col_copy]

print('Calculating loan to income ratio..')
finalTrain['loan_to_income'] = finalTrain['annual_inc']/finalTrain['funded_amnt_inv']
finalTest['loan_to_income'] = finalTest['annual_inc']/finalTest['funded_amnt_inv']


finalTrain['repay_stat'] = finalTrain['acc_now_delinq'] + (finalTrain['total_rec_late_fee']/finalTrain['funded_amnt_inv']) + (finalTrain['recoveries']/finalTrain['funded_amnt_inv']) + (finalTrain['collection_recovery_fee']/finalTrain['funded_amnt_inv']) + (finalTrain['collections_12_mths_ex_med']/finalTrain['funded_amnt_inv'])
finalTest['repay_stat'] = finalTest['acc_now_delinq'] + (finalTest['total_rec_late_fee']/finalTest['funded_amnt_inv']) + (finalTest['recoveries']/finalTest['funded_amnt_inv']) + (finalTest['collection_recovery_fee']/finalTest['funded_amnt_inv']) + (finalTrain['collections_12_mths_ex_med']/finalTest['funded_amnt_inv'])

print('Calculating total number of available/unused credit lines..')
finalTrain['avl_lines'] = finalTrain['total_acc'] - finalTrain['open_acc']
finalTest['avl_lines'] = finalTest['total_acc'] - finalTest['open_acc']

print('Calculating total interest paid so far..')
finalTrain['int_paid'] = finalTrain['total_rec_int'] + finalTrain['total_rec_late_fee']
finalTest['int_paid'] = finalTest['total_rec_int'] + finalTest['total_rec_late_fee']

print('Calculating EMIs paid (in terms of percent)..')
finalTrain['emi_paid_progress_perc'] = ((finalTrain['last_week_pay']/(finalTrain['term']/12*52+1))*100)
finalTest['emi_paid_progress_perc'] = ((finalTest['last_week_pay']/(finalTest['term']/12*52+1))*100)

print('Calculating total repayments received so far..')
finalTrain['total_repayment_progress'] = ((finalTrain['last_week_pay']/(finalTrain['term']/12*52+1))*100) + ((finalTrain['recoveries']/finalTrain['funded_amnt_inv']) * 100)
finalTest['total_repayment_progress'] = ((finalTest['last_week_pay']/(finalTest['term']/12*52+1))*100) + ((finalTest['recoveries']/finalTest['funded_amnt_inv']) * 100)

print('\n* Cleaning dataset Done.')

#%%
print('\nTraining data..')
X_train, X_test, y_train, y_test = train_test_split(np.array(finalTrain), np.array(train_target), train_size=0.80)
eval_set=[(X_test, y_test)]

classifier = xgboost.sklearn.XGBClassifier(objective="binary:logistic", learning_rate=0.05, seed=9616, max_depth=20, gamma=10, n_estimators=500)
classifier.fit(X_train, y_train, early_stopping_rounds=20, eval_metric="auc", eval_set=eval_set, verbose=True)


print('Training data Done.')

print('\nPredicting..')
y_pred = classifier.predict(X_test)
file_name = 'Result'

accuracy = accuracy_score(np.array(y_test).flatten(), y_pred)
print("Accuracy: %.10f%%" % (accuracy * 100.0))
file_name = file_name + (" - Accuracy %.6f" % (accuracy * 100))

final_pred = pd.DataFrame(classifier.predict_proba(np.array(finalTest)))
print('Prediction done..')


print('Loading predictions..')
dfRes = pd.concat([test_id, final_pred.ix[:, 1:2]], axis=1)
dfRes.rename(columns={1:'loan_status'}, inplace=True)
dfRes.to_csv((('%s.csv') % (file_name)), index=False)
print('Loading predictions Done.')

print('\n* Refer file %s to see predictions.' %file_name)