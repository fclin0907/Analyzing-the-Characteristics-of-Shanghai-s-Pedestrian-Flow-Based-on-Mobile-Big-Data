from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession

conf = SparkConf().setAppName("Community Detect").setMaster("spark://192.168.80.141:7077")
#通过SparConf对象构建SparkContext对象
sc = SparkContext(conf=conf)

spark = SparkSession.builder \
    .master("spark://192.168.80.141:7077") \
    .appName("Community Detect") \
    .getOrCreate()

df = spark.read.csv('/mnt/data/shanghai.txt', sep='\t', header=True)
df.createOrReplaceTempView('shanghai')

sqlDF = spark.sql('select count(distinct imsi) from shanghai')
sqlDF.show()
