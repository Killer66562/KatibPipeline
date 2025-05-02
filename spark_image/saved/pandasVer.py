import pandas as pd

    # 讀取與程式碼位於同一個資料夾中的 stroke.csv
df_data_1 = pd.read_csv('https://raw.githubusercontent.com/s102401002/kubeflowPipeline0722/main/stroke.csv')

    # 移除不需要的欄位
df_data_1 = df_data_1.drop(columns=['id', 'ever_married', 'work_type'])

    # 定義映射
gender_map = {'Male': 0, 'Female': 1}
smoking_status_map = {'Unknown': 0, 'never smoked': 0, 'formerly smoked': 1, 'smokes': 1}
Residence_type_map = {'Urban': 1, 'Rural': 0}

    # 補齊資料
    # gender
df_data_1 = df_data_1[(df_data_1['gender'] != 'N/A') & (~df_data_1['gender'].isna())]
df_data_1['gender'] = df_data_1['gender'].map(gender_map)  # map

    # age
df_data_1 = df_data_1[(df_data_1['age'] != 'N/A') & (~df_data_1['age'].isna())]

    # hypertension
df_data_1 = df_data_1[(df_data_1['hypertension'] != 'N/A') & (~df_data_1['hypertension'].isna())]

    # heart_disease
df_data_1 = df_data_1[(df_data_1['heart_disease'] != 'N/A') & (~df_data_1['heart_disease'].isna())]

    # Residence_type
df_data_1 = df_data_1[(df_data_1['Residence_type'] != 'N/A') & (~df_data_1['Residence_type'].isna())]
df_data_1['Residence_type'] = df_data_1['Residence_type'].map(Residence_type_map)  # map

    # avg_glucose_level
df_data_1 = df_data_1[(df_data_1['avg_glucose_level'] != 'N/A') & (~df_data_1['avg_glucose_level'].isna())]

    # bmi
df_data_1 = df_data_1[(df_data_1['bmi'] != 'N/A') & (~df_data_1['bmi'].isna())]

    # smoking_status
df_data_1 = df_data_1[(df_data_1['smoking_status'] != 'N/A') & (~df_data_1['smoking_status'].isna())]
df_data_1['smoking_status'] = df_data_1['smoking_status'].map(smoking_status_map)  # map

#df_data_1 = df_data_1.drop(3116)#特殊處理

#df_data_1 = df_data_1.sample(frac=1).reset_index(drop=True)

df_data_2 = pd.read_csv('https://raw.githubusercontent.com/s102401002/kubeflowPipeline0722/main/stroke_2.csv')
df_data_2 = df_data_2.drop(columns=['ever_married', 'work_type'])
df_data_2.rename(columns={'sex': 'gender'}, inplace=True)
    #合併
df_data = pd.concat([df_data_1, df_data_2], ignore_index=True)

    # 删除指定的行
rows_to_delete = [27386, 33816, 40092]
df_data = df_data.drop(index=rows_to_delete)