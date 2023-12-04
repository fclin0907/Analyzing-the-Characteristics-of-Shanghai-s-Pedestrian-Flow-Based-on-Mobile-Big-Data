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

df = spark.read.option('delimiter', '\t').option('header','true').csv('/mnt/data/shanghai.txt')

def date(ts):
    return ts.split(' ')[0]

def time(ts):
    return ts.split(' ')[1]

udf_date = udf(date)
udf_time = udf(time)

df2 = df.withColumn('date', udf_date(df.ts))

df3 = df2.filter(df2['date']=='2019-07-12')
df4 = df3.withColumn('time', udf_time(df.ts))
df4.registerTempTable('data')
spark.catalog.cacheTable('data')


from dask.distributed import Client, progress
client = Client(n_workers=2, threads_per_worker=2, memory_limit='1GB')

import dask.dataframe as dd

df = dd.read_csv('data.txt/*.csv', delimiter='\t')

