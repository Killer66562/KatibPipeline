from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, mean
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType, BooleanType, DecimalType
from pyspark.sql import functions as F
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
  spark = SparkSession.builder.appName('pyspark-process-heartDisease_Dataset').getOrCreate()
  df_data = spark.read.csv('/tmp/dataset/heart_disease/heart_2020_cleaned.csv', header=True, inferSchema=True)
  df_data = df_data.drop('PhysicalHealth', 'MentalHealth', 'Race' , 'GenHealth')
  df_data.na.drop()

  df_data = df_data.replace(['Yes', 'No'], ['1', '0'], 'HeartDisease')
  df_data = df_data.replace(['Yes', 'No'], ['1', '0'], 'Smoking')
  df_data = df_data.replace(['Yes', 'No'], ['1', '0'], 'AlcoholDrinking')
  df_data = df_data.replace(['Yes', 'No'], ['1', '0'], 'Stroke')
  df_data = df_data.replace(['Yes', 'No'], ['1', '0'], 'DiffWalking')
  df_data = df_data.replace(['Male', 'Female'], ['0', '1'], 'Sex')
  AGE_MAP_LIST = ['0-4', '5-9', '10-14', '15-17', '18-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54', '55-59', '60-64', '65-69', '70-74', '75-79', '80 or older']
  AGE_MAP_VAL_LIST = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16']
  df_data = df_data.replace(AGE_MAP_LIST, AGE_MAP_VAL_LIST, 'AgeCategory')
  df_data = df_data.replace(['Yes (during pregnancy)', 'Yes', 'No', 'No, borderline diabetes'], ['1', '1', '0', '0'], 'Diabetic')
  df_data = df_data.replace(['Yes', 'No'], ['1', '0'], 'PhysicalActivity')
  df_data = df_data.replace(['Yes', 'No'], ['1', '0'], 'Asthma')
  df_data = df_data.replace(['Yes', 'No'], ['1', '0'], 'KidneyDisease')
  df_data = df_data.replace(['Yes', 'No'], ['1', '0'], 'SkinCancer')

  columns_order = ['Sex', 'AgeCategory'] + [col for col in df_data.columns if col not in ['Sex', 'AgeCategory']]
  df_data = df_data[columns_order]
  
  #datatype exchange
  df_data = df_data.withColumn('Sex', df_data['Sex'].cast(IntegerType()))
  df_data = df_data.withColumn('AgeCategory', df_data['AgeCategory'].cast(IntegerType()))
  df_data = df_data.withColumn('HeartDisease', df_data['HeartDisease'].cast(IntegerType()))
  df_data = df_data.withColumn('Smoking', df_data['Smoking'].cast(IntegerType()))
  df_data = df_data.withColumn('AlcoholDrinking', df_data['AlcoholDrinking'].cast(IntegerType()))
  df_data = df_data.withColumn('Stroke', df_data['Stroke'].cast(IntegerType()))
  df_data = df_data.withColumn('DiffWalking', df_data['DiffWalking'].cast(IntegerType()))
  df_data = df_data.withColumn('Diabetic', df_data['Diabetic'].cast(IntegerType()))
  df_data = df_data.withColumn('PhysicalActivity', df_data['PhysicalActivity'].cast(IntegerType()))
  df_data = df_data.withColumn('Asthma', df_data['Asthma'].cast(IntegerType()))
  df_data = df_data.withColumn('KidneyDisease', df_data['KidneyDisease'].cast(IntegerType()))
  df_data = df_data.withColumn('SkinCancer', df_data['SkinCancer'].cast(IntegerType()))

  df_data_x = df_data.select('Sex', 'AgeCaategory', 'BMI', 'Smoking', 'AlcoholDrinking', 'Stroke', 'DiffWalking', 'Diabetic', 'PhysicalActivity', 'SleepTime', 'Asthma', 'KidneyDisease', 'SkinCancer').toPandas()
  df_data_y = df_data.select('HeartDisease').toPandas()

  x_train, x_test, y_train, y_test = train_test_split(df_data_x, df_data_y, test_size=0.2, random_state=42)

  x_train.to_csv('/tmp/processed_dataset/heart_disease/x_train.csv', header=True, index=False)
  x_test.to_csv('/tmp/processed_dataset/heart_disease/x_test.csv', header=True, index=False)
  y_train.to_csv('/tmp/processed_dataset/heart_disease/y_train.csv', header=True, index=False)
  y_test.to_csv('/tmp/processed_dataset/heart_disease/y_test.csv', header=True, index=False)
  df_data.toPandas().to_csv('/tmp/processed_dataset/heart_disease/processed_heartDisease_dataset.csv', header=True, index=False)
  df_data.show()
  df_data.printSchema()