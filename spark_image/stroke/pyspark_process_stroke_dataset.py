from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, mean
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType, BooleanType, DecimalType
from pyspark.sql import functions as F
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
  spark = SparkSession.builder.appName('pyspark-process-stroke-dataset').getOrCreate()
  df_data = spark.read.csv('/tmp/dataset/stroke/stroke.csv', header=True, inferSchema=True)
  df_data = df_data.drop('id', 'ever_married', 'work_type')

  GENDER_MAP = {'Male': 0, 'Female': 1}
  SMOKING_STATUS_MAP = {'Unknown': 0, 'never smoked': 0, 'formerly smoked': 1, 'smokes': 1}
  RESIDENCE_TYPE_MAP = {'Urban': 1, 'Rural': 0}

  #gender data process
  df_data = df_data.filter((df_data['gender'].isNotNull()) & (df_data['gender'] != 'N/A'))
  df_data_ha = df_data.filter((df_data['gender'] == 'Male') | (df_data['gender'] == 'Female'))
  df_data = df_data.replace(['Male', 'Female'], ['0', '1'], 'gender')

  #age data process
  df_data = df_data.filter(df_data['age'].isNotNull())

  #hypertension data process
  df_data = df_data.filter(df_data['hypertension'].isNotNull())

  df_data = df_data.filter(df_data['heart_disease'].isNotNull())

  df_data = df_data.filter(df_data['Residence_type'].isNotNull())
  df_data = df_data.replace(['Rural', 'Urban'], ['0', '1'], 'Residence_type')

  df_data = df_data.filter(df_data['avg_glucose_level'].isNotNull())

  df_data = df_data.filter((df_data['bmi'].isNotNull()) & (df_data['bmi'] != 'N/A'))

  df_data = df_data.filter((df_data['smoking_status'].isNotNull()) & (df_data['smoking_status'] != 'N/A'))
  df_data = df_data.replace(['Unknown', 'never smoked', 'formerly smoked', 'smokes'], ['0', '0', '1', '1'], 'smoking_status')

  #2nd stroke dataset
  df_data_2 = spark.read.csv('/tmp/dataset/stroke_2.csv', header=True, inferSchema=True)
  df_data_2 = df_data_2.drop('ever_married', 'work_type')
  df_data_2.withColumnRenamed('sex', 'gender')

  df_data = df_data.union(df_data_2)

  #datatype exchange
  df_data = df_data.withColumn("gender", df_data["gender"].cast(IntegerType()))
  df_data = df_data.withColumn("age", df_data["age"].cast(DoubleType()))
  df_data = df_data.withColumn("hypertension", df_data["hypertension"].cast(IntegerType()))
  df_data = df_data.withColumn("heart_disease", df_data["heart_disease"].cast(IntegerType()))
  df_data = df_data.withColumn("Residence_type", df_data["Residence_type"].cast(IntegerType()))
  df_data = df_data.withColumn("avg_glucose_level", df_data["avg_glucose_level"].cast(DoubleType()))
  df_data = df_data.withColumn("bmi", df_data["bmi"].cast(DoubleType()))
  df_data = df_data.withColumn("smoking_status", df_data["smoking_status"].cast(IntegerType()))

  #save processed dataset
  '''
  df_data.coalesce(1) \
    .write.option("header", True) \
    .format("csv") \
    .mode("overwrite") \
    .save("./test.csv")
  '''

  df_data_x = df_data.select('gender', 'age',' hypertension', 'heart_disease', 'Residence_type', 'avg_glucose_level', 'bmi', 'smoking_status').toPandas()
  df_data_y = df_data.select('stroke')
  x_train, x_test, y_train, y_test = train_test_split(df_data_x, df_data_y, test_size=0.2, random_state=42)

  x_train.to_csv('/tmp/processed_dataset/stroke/x_train.csv', header=True, index=False)
  x_test.to_csv('/tmp/processed_dataset/stroke/x_test.csv', header=True, index=False)
  y_train.to_csv('/tmp/processed_dataset/stroke/y_train.csv', header=True, index=False)
  y_test.to_csv('/tmp/processed_dataset/stroke/y_test.csv', header=True, index=False)
  df_data.toPandas().to_csv('/tmp/processed_dataset/stroke/processed_stroke_dataset.csv', header=True, index=False)
  df_data.show()

