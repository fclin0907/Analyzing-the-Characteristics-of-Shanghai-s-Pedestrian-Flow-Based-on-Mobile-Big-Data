from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf


conf = SparkConf().setAppName("Shanghai Days").setMaster("spark://192.168.80.141:7077")
sc = SparkContext(conf=conf)
spark = SparkSession \
    .builder \
    .master('spark://192.168.80.141:7077') \
    .appName("Build Shanghai dates") \
    .getOrCreate()

df = spark.read.option('delimiter', '\t').option('header','true').csv('hdfs://master:9000/user/honlan/shanghai.txt')

def date(ts):
    return ts.split(' ')[0]

udf_date = udf(date)

df2 = df.withColumn('date', udf_date(df.ts))

df2.write.option('header', True).partitionBy('date').format('parquet').save('hdfs://master:9000/user/honlan/shanghai.parquet')