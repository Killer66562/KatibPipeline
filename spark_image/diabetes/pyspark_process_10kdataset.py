from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, mean, round
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType, DecimalType, FloatType
from sklearn.model_selection import train_test_split
import os

if __name__ == "__main__":
  spark = SparkSession.builder.appName('pyspark-process-10kDataset').getOrCreate()
  #dataset_path = os.getenv("datasetPath")
  df_data = spark.read.csv('/tmp/dataset/diabetes/10k.csv', header=True, inferSchema=True)
    
  df_data = df_data.select('gender', 'age', 'bmi', 'HbA1c_level', 'blood_glucose_level', 'diabetes')
  df_data = df_data.filter((df_data['diabetes'] == 0) | (df_data['diabetes'] == 1))
  df_data = df_data.dropna(thresh=2)
  df_data = df_data.replace(['Male', 'Female', 'Other'], ['0', '1', '2'], 'gender')
  df_data = df_data.filter((df_data['gender'] == 1)|(df_data['gender'] == 0))
  df_data = df_data.withColumn('gender', df_data['gender'].cast(IntegerType()))

  mean_age = df_data.select(mean(col('age')).alias('mean_age')).collect()[0]['mean_age']
  mean_bmi = df_data.select(mean(col('bmi')).alias('mean_bmi')).collect()[0]['mean_bmi']
  mean_HbA1c_level = df_data.select(mean(col('HbA1c_level')).alias('mean_HbA1c_level')).collect()[0]['mean_HbA1c_level']
  mean_blood_glucose_level = df_data.select(mean(col('blood_glucose_level')).alias('mean_blood_glucose_level')).collect()[0]['mean_blood_glucose_level']
  '''
  mean_age = df_data.filter(df_data['age'] != 'No Info').select(mean(col('age')).alias('mean_age')).collect()[0]['mean_age']
  mean_bmi = df_data.filter(df_data['bmi'] != 'No Info').select(mean(col('bmi')).alias('mean_bmi')).collect()[0]['mean_bmi']
  mean_HbA1c_level = df_data.filter(df_data['HbA1c_level'] != 'No Info').select(mean(col('HbA1c_level')).alias('mean_HbA1c_level')).collect()[0]['mean_HbA1c_level']
  mean_blood_glucose_level = df_data.filter(df_data['blood_glucose_level'] != 'No Info').select(mean(col('blood_glucose_level')).alias('mean_blood_glucose_level')).collect()[0]['mean_blood_glucose_level']
  '''
  print(f'{mean_age}, {mean_bmi}, {mean_HbA1c_level}, {mean_blood_glucose_level}')

  #df_data = df_data.fillna({'age': mean_age, 'bmi': mean_bmi, 'HbA1c_level': mean_HbA1c_level, 'blood_glucose_level': mean_blood_glucose_level})
  df_data = df_data.withColumn('age', when((df_data['age'] == 'No Info') | (df_data['age'].isNull()), mean_age).otherwise(df_data['age']))
  df_data = df_data.withColumn('bmi', when((df_data['bmi'] == 'No Info') | (df_data['bmi'].isNull()), mean_bmi).otherwise(df_data['bmi']))
  df_data = df_data.withColumn('HbA1c_level', when((df_data['HbA1c_level'] == 'No Info') | (df_data['HbA1c_level'].isNull()), mean_HbA1c_level).otherwise(df_data['HbA1c_level']))
  df_data = df_data.withColumn('blood_glucose_level', when((df_data['blood_glucose_level'] == 'No Info') | (df_data['blood_glucose_level'].isNull()), mean_blood_glucose_level).otherwise(df_data['blood_glucose_level']))

  df_data = df_data.withColumn('age', round(df_data['age'], 2))
  df_data = df_data.withColumn('bmi', round(df_data['bmi'], 2))
  df_data = df_data.withColumn('HbA1c_level', round(df_data['HbA1c_level'], 1))
  df_data = df_data.withColumn('blood_glucose_level', round(df_data['blood_glucose_level'], 2))
    
  df_data.show()
  df_data.printSchema()

  df_data_x = df_data.select('gender', 'age', 'bmi', 'HbA1c_level', 'blood_glucose_level').toPandas()
  df_data_y = df_data.select('diabetes').toPandas()
  x_train, x_test, y_train, y_test = train_test_split(df_data_x, df_data_y, test_size=0.2, random_state=42)

  #df_data.coalesce(1).write.csv('mycsv2.csv')
  #df_data.toPandas().to_csv('/tmp/processed_dataset/diabetes/processed_diabetes_dataset.csv', header=True, index=False)
  x_train.to_csv('/tmp/processed_dataset/diabetes/x_train.csv', header=True, index=False)
  x_test.to_csv('/tmp/processed_dataset/diabetes/x_test.csv', header=True, index=False)
  y_train.to_csv('/tmp/processed_dataset/diabetes/y_train.csv', header=True, index=False)
  y_test.to_csv('/tmp/processed_dataset/diabetes/y_test.csv', header=True, index=False)
  #df_data.write.csv('./processed_dataset.csv', header=False, mode='overwrite')
  #spark.stop()
