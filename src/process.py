import pandas as pd
import sklearn
import sklearn.model_selection


url = "https://raw.githubusercontent.com/daniel88516/diabetes-data/main/10k.csv"
df_data = pd.read_csv(url)
    
df_data = df_data.drop(df_data[df_data['diabetes'] == 'No Info'].index)
df_data = df_data[['gender','age', 'bmi', 'HbA1c_level', 'blood_glucose_level', 'diabetes']]
df_data = df_data.dropna(thresh=4)
    
gender_map = {'Male': 0 , 'Female': 1  , 'Other': 2}
df_data['gender'] = df_data['gender'].map(gender_map)
df_data = df_data[df_data['gender'] != 2]
df_data['age'] = df_data['age'].replace('No Info', df_data['age'].mean())
df_data['bmi'] = df_data['bmi'].replace('No Info', df_data['bmi'].mean())
df_data['HbA1c_level'] = df_data['HbA1c_level'].replace('No Info', df_data['HbA1c_level'].mean())
df_data['blood_glucose_level'] = df_data['blood_glucose_level'].replace('No Info', df_data['blood_glucose_level'].mean())

x = df_data.drop(labels=['diabetes'], axis=1)
y = df_data[['diabetes']]
    
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2, random_state=42)
    
x_train_df = pd.DataFrame(x_train)
x_test_df = pd.DataFrame(x_test)
y_train_df = pd.DataFrame(y_train)
y_test_df = pd.DataFrame(y_test)

x_train_df.to_csv('src/datasets/x_train.csv', index=False)
x_test_df.to_csv('src/datasets/x_test.csv', index=False)
y_train_df.to_csv('src/datasets/y_train.csv', index=False)
y_test_df.to_csv('src/datasets/y_test.csv', index=False)