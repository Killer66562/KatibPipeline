from pyspark.sql import SparkSession
from pyspark.sql.types import IntegerType, StringType
from pyspark.sql.functions import col, when, mean
import os

if __name__ == "__main__":
  spark = SparkSession.builder.appName('pyspark-process-10kDataset').getOrCreate()
  #dataset_path = os.getenv("datasetPath")
  df_data = spark.read.csv('/tmp/dataset/10kdataset.csv'  , header=True, inferSchema=True)

  df_data = df_data.withColumn('diabetes', df_data['diabetes'].cast(StringType()))
  df_data = df_data.filter(df_data['diabetes'] != 'No Info')

  df_data = df_data.select('gender', 'age', 'bmi', 'HbA1c_level', 'blood_glucose_level', 'diabetes')
  df_data = df_data.dropna(thresh=4)
  df_data = df_data.replace(['Male', 'Female', 'Other'], ['0', '1', '2'], 'gender')
  df_data = df_data.filter(df_data['gender'] != 2)
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

  df_data = df_data.withColumn('age', when(df_data['age'] == 'No Info', mean_age).otherwise(df_data['age']))
  df_data = df_data.withColumn('bmi', when(df_data['bmi'] == 'No Info', mean_bmi).otherwise(df_data['bmi']))
  df_data = df_data.withColumn('HbA1c_level', when(df_data['HbA1c_level'] == 'No Info', mean_HbA1c_level).otherwise(df_data['HbA1c_level']))
  df_data = df_data.withColumn('blood_glucose_level', when(df_data['blood_glucose_level'] == 'No Info', mean_blood_glucose_level).otherwise(df_data['blood_glucose_level']))

  df_data.show()
  df_data.printSchema()

  #df_data.coalesce(1).write.csv('mycsv2.csv')

  df_data.toPandas().to_csv('/tmp/processed_dataset/processed_diabetes_dataset.csv', header=True, index=False)

  #df_data.write.csv('./processed_dataset.csv', header=False, mode='overwrite')
  spark.stop()