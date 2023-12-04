# -*- coding: utf-8 -*-
"""
@author: Bangyan Lin, Yue Wu
"""


from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import IntegerType
from pyspark.sql.window import Window
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from numpy.matlib import repmat
from datetime import timedelta
import pickle
import Geohash as geohash


spark = SparkSession \
    .builder \
    .appName("Check ppl in region per hour") \
    .getOrCreate()

def read_data():
    df_ppl = spark.sql("SELECT imei_id, lgt, ltt, ts FROM parquet.`hdfs://192.168.2.3:9000/shanghai.parquet`")
    # df_ppl = spark.sql("SELECT imei_id, ts, geohash6, geohash7, date FROM parquet.`shanghai_region_geohash.parquet`")
    # df_ppl = df_ppl.withColumn('hour', date_trunc('hour', df_ppl['ts']))
    # df_region = spark.read.parquet('region_geohash3.parquet')
    # df_region = spark.read.options(header='True').csv('0809Newregion174185.csv')
    # df_region = df_region.select('longitude', 'latitude', 'type', 'Title')
    return df_ppl

def cal_ppl_portion(df_ppl, df_region):
    # region 7 geohash
    # df_ppl_region = df_ppl.join(broadcast(df_region_7), df_ppl.geohash7 == df_region_7.geohash7, 'left')
    # df_ppl_region_7 = df_ppl_region.filter(df_ppl_region.agent_id.isNotNull()).select('imei_id', 'agent_id', 'date', 'ts')
    # df_ppl_null_region = df_ppl_region.filter(df_ppl_region.agent_id.isNull())
    # region 6 geohash
    df_region_7 = df_region.filter(col('type')==7).select('type', 'geohash8')
    df_region_6 = df_region.filter(col('type')==6).select('type', 'geohash7')
    df_region_5 = df_region.filter(col('type')==5).select('type', 'geohash7')
    df_region_1 = df_region.filter(col('type')==1).select('type', 'geohash6')
    df_region_2 = df_region.filter(col('type')==2).select('type', 'geohash6')
    df_region_3 = df_region.filter(col('type')==3).select('type', 'geohash6')
    df_region_4 = df_region.filter(col('type')==4).select('type', 'geohash6')
    # type 7
    df_ppl_region = df_ppl.join(broadcast(df_region_7), df_ppl.geohash8 == df_region_7.geohash8, 'left')
    df_ppl_region_7 = df_ppl_region.filter(col('type').isNotNull()).select('id', 'agent_id', 'date', 'ts', 'type')
    df_ppl_null_region = df_ppl_region.filter(col('type').isNull())
    # type 6
    df_ppl_region = df_ppl_null_region.join(broadcast(df_region_6), df_ppl_null_region.geohash7 == df_region_6.geohash7, 'left')
    df_ppl_region_6 = df_ppl_region.filter(col('type').isNotNull()).select('id', 'agent_id', 'date', 'ts', 'type')
    df_ppl_null_region = df_ppl_region.filter(col('type').isNull())
    # type 5
    df_ppl_region = df_ppl_null_region.join(broadcast(df_region_5), df_ppl_null_region.geohash7 == df_region_5.geohash7, 'left')
    df_ppl_region_5 = df_ppl_region.filter(col('type').isNotNull()).select('id', 'agent_id', 'date', 'ts', 'type')
    df_ppl_null_region = df_ppl_region.filter(col('type').isNull())
    # type 4
    df_ppl_region = df_ppl_null_region.join(broadcast(df_region_4), df_ppl_null_region.geohash6 == df_region_4.geohash6, 'left')
    df_ppl_region_4 = df_ppl_region.filter(col('type').isNotNull()).select('id', 'agent_id', 'date', 'ts', 'type')
    df_ppl_null_region = df_ppl_region.filter(col('type').isNull())
    # type 3
    df_ppl_region = df_ppl_null_region.join(broadcast(df_region_3), df_ppl_null_region.geohash6 == df_region_3.geohash6, 'left')
    df_ppl_region_3 = df_ppl_region.filter(col('type').isNotNull()).select('id', 'agent_id', 'date', 'ts', 'type')
    df_ppl_null_region = df_ppl_region.filter(col('type').isNull())
    # type 2
    df_ppl_region = df_ppl_null_region.join(broadcast(df_region_2), df_ppl_null_region.geohash6 == df_region_2.geohash6, 'left')
    df_ppl_region_2 = df_ppl_region.filter(col('type').isNotNull()).select('id', 'agent_id', 'date', 'ts', 'type')
    df_ppl_null_region = df_ppl_region.filter(col('type').isNull())
    # type 1
    df_ppl_region = df_ppl_null_region.join(broadcast(df_region_1), df_ppl_null_region.geohash6 == df_region_1.geohash6, 'left')
    df_ppl_region_1 = df_ppl_region.filter(col('type').isNotNull()).select('id', 'agent_id', 'date', 'ts', 'type')
    # type 8
    df_ppl_null_region = df_ppl_region.filter(col('type').isNull()).withColumn('type', 8)
    return df_ppl_region_7.union(df_ppl_region_6).union(df_ppl_region_5).union(df_ppl_region_4).union(df_ppl_region_3).union(df_ppl_region_2).union(df_ppl_region_1).union(df_ppl_null_region), df_ppl_null_region


def region_geohash(ltt, lgt, precision=5):
    return geohash.encode(float(ltt), float(lgt), precision)


def haversine(coord1, lat2, lon2):
    R = 6372800  # Earth radius in meters
    lat1, lon1 = coord1
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2) 
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + \
        np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
    return 2*R*np.atan2(np.sqrt(a), np.sqrt(1 - a))


def check_single_type_subarea(subarea_list):
    single_type_subareas = {}
    for i in range(1, 9):
        single_type_subareas[i] = []
    for k in subarea_list.keys():
        type_list = subarea_list[k][2]
        if not type_list:
            single_type_subareas[7].append(k)
            continue
        is_single_type = True
        t = type_list[0]
        for _ in type_list:
            if _ != t:
                is_single_type = False
        if is_single_type:
            single_type_subareas[int(t)].append(k)
    return single_type_subareas


def cal_subarea_list(subarea_df):
    # create dict
    subarea_list = {}
    for i in range(2, 5081):
        subarea_list[i] = [[],[],[]]
    for r in(subarea_df):
        idx = r['agent_id']
        if idx == 2:
            print(r['type'])
        if r['type'] == '7':
            continue
        subarea_list[idx][0].append(r['latitude'])
        subarea_list[idx][1].append(r['longitude'])
        subarea_list[idx][2].append(r['type'])
    return subarea_list

def cal_single_type(df_cal, single_type_subareas):
    df_cal = df_cal.withColumn('type', when(df_cal.agent_id.isin(single_type_subareas[1]), 1)
                                        .when(df_cal.agent_id.isin(single_type_subareas[2]), 2)
                                        .when(df_cal.agent_id.isin(single_type_subareas[3]), 3)
                                        .when(df_cal.agent_id.isin(single_type_subareas[4]), 4)
                                        .when(df_cal.agent_id.isin(single_type_subareas[5]), 5)
                                        .when(df_cal.agent_id.isin(single_type_subareas[6]), 6)
                                        .when(df_cal.agent_id.isin(single_type_subareas[7]), 7)
                                        .when(df_cal.agent_id.isin(single_type_subareas[8]), 8)
                                        .otherwise(df_cal.type))


def cal_type(ltt, lgt, agent_id, subarea_dict):
    subarea = subarea_dict[int(agent_id)]
    dists = haversine((ltt, lgt), subarea[0], subarea[1])
    return subarea[2][np.argmin(dists)]

calTypeUDF = udf(lambda x, y, z: cal_type(x, y, z, subarea_dict), IntegerType())

df_cal_left = df_cal_left.withColumn('type', calTypeUDF(df_cal_left['ltt'], df_cal_left['lgt'], df_cal_left['agent_id']))


geohashUDF5 = udf(lambda x,y: region_geohash(x, y, 5))
geohashUDF6 = udf(lambda x,y: region_geohash(x, y, 6))
geohashUDF7 = udf(lambda x,y: region_geohash(x, y, 7))
geohashUDF8 = udf(lambda x,y: region_geohash(x, y, 8))

def cal_region_geohash(df_region):
    df_region = df_region.withColumn('geohash7', geohashUDF7(df_region['latitude'],df_region['longitude']))
    df_region = df_region.withColumn('geohash8', geohashUDF8(df_region['latitude'],df_region['longitude']))
    return df_region


def cal_shanghai_geohash(df_region):
    df_region = df_region.withColumn('geohash6', geohashUDF6(df_region['ltt'], df_region['lgt']))
    df_region = df_region.withColumn('geohash7', geohashUDF7(df_region['ltt'], df_region['lgt']))
    df_region = df_region.withColumn('geohash8', geohashUDF8(df_region['ltt'], df_region['lgt']))
    return df_region
    

def cal_region_adj(df_region):
    df_region_list = df_region.select('agent_id', 'geohash6','geohash7','geohash8').collect()
    adj_df_list = []
    for row in df_region_list:
        agent_id_list = [row['agent_id'] for i in range(9)]
        geohash8_list = geohash.expand(row['geohash8'])
        for r in zip(agent_id_list, geohash8_list):
            adj_df_list.append(r)
    return spark.createDataFrame(data=adj_df_list, schema=['agent_id', 'geohash5', 'geohash6', 'geohash7'])

df_region = spark.read.options(header='True').csv('0809Newregion174185.csv')
df_region = df_region.select('longitude', 'latitude', 'type', 'Title', 'Larea')
df_region_cal = cal_region_geohash(df_region)


df_cal_portion = cal_ppl_portion(df_ppl_cal, df_region_cal)

df_ppl_region_7 = df_ppl_region_7.withColumn('hour', date_trunc('hour', df_ppl_region_7['ts']))
df_ppl_region_6 = df_ppl_region_6.withColumn('hour', date_trunc('hour', df_ppl_region_6['ts']))
df_ppl_region_5 = df_ppl_region_5.withColumn('hour', date_trunc('hour', df_ppl_region_5['ts']))
df_ppl_region_4 = df_ppl_region_4.withColumn('hour', date_trunc('hour', df_ppl_region_4['ts']))
df_ppl_region_3 = df_ppl_region_3.withColumn('hour', date_trunc('hour', df_ppl_region_3['ts']))
df_ppl_region_2 = df_ppl_region_2.withColumn('hour', date_trunc('hour', df_ppl_region_2['ts']))
df_ppl_region_1 = df_ppl_region_1.withColumn('hour', date_trunc('hour', df_ppl_region_1['ts']))


def get_data():
    df_1 = spark.read.parquet('shanghai_region_type_df1.parquet')
    df_2 = spark.read.parquet('shanghai_region_type_df2.parquet')
    df_shanghai = spark.read.parquet('shanghai_region_type.parquet')
    #df_1 = spark.sql("SELECT * FROM parquet.`shanghai_region_type.parquet` where date=='2019-07-01'")
   # df_2 = spark.sql("SELECT * FROM parquet.`shanghai_region_type.parquet` where date=='2019-07-09'")
   # df_shanghai = spark.sql("SELECT * FROM parquet.`shanghai_region_full_geohash_id.parquet` where date>='2019-07-02' and date<='2019-07-08'")
    df_shanghai = df_shanghai.withColumn('hour', date_trunc('hour', df_shanghai['ts'])).fillna({'type': '8'})
    df_1 = df_1.withColumn('hour', date_trunc('hour', df_1['ts'])).fillna({'type': '8'})
    df_2 = df_2.withColumn('hour', date_trunc('hour', df_2['ts'])).fillna({'type': '8'})
    return df_1, df_2, df_shanghai

#Window（）函数看一下
def cal_full_records_max(df1, df2, df_shanghai):
    # add df1 to df_shanghai
    w = Window().partitionBy('imei_id').orderBy(col('ts').desc())
    _df = df1.withColumn('rn', row_number().over(w)).filter(col('rn')==1).drop('rn')
    # get max ts as the position of a user within that hour
    w = Window().partitionBy([col('imei_id'), col('hour')]).orderBy(col('ts').desc())
    _df2 = df_shanghai.withColumn('rn', row_number().over(w)).filter(col('rn')==1).drop('rn')
    df_shanghai = _df2.union(_df)
    # add df2 to df_shanghai
    w = Window().partitionBy('imei_id').orderBy(col('ts'))
    _df = df2.withColumn('rn', row_number().over(w)).filter(col('rn')==1).drop('rn')
    df_shanghai = df_shanghai.union(_df)
    # check time diff
    w = Window().partitionBy('imei_id').orderBy('ts')
    df_shanghai = df_shanghai.withColumn('pre_hour', lag('hour', 1).over(w)).withColumn('pre_agent_id', lag('agent_id', 1).over(w)).withColumn('next_agent_id', lead('agent_id', 1).over(w)) \
                .withColumn('pretype', lag('type', 1).over(w))
    # fill pre_hour na as 2019-07-01 00:00:00
    df_shanghai = df_shanghai.fillna({'pre_hour': '2019-07-01 00:00:00', 'pretype': '8'})
    # cal hour diff
    df_shanghai = df_shanghai.withColumn('hour', to_timestamp(col('hour'))).withColumn('pre_hour', to_timestamp(col('pre_hour'))) \
                            .withColumn('hour_diff', round((unix_timestamp('hour') - unix_timestamp('pre_hour')) / 3600))
    # find hour diff within 24
    # _df = df_shanghai.filter((col('hour_diff') > 1))
    # add records
    records_rdd = df_shanghai.rdd.flatMap(lambda x: _add_records(x))
    # return rdd
    return records_rdd

def _map_row_type(row):
    idx = str(int(row[1])-1)
    return (idx+';'+row[2]+';'+str(row[3]), 1)
#补全一个小时不移动也不交互的空白数据
def _add_records(row):
    # no previous agent_id. first record in the dataset
    if not row['pre_agent_id'] or row['pre_agent_id']=='null':
        if not row['next_agent_id'] or row['next_agent_id']=='null':
            _item1 = (row['imei_id'], row['agent_id'], row['hour'].strftime('%Y-%m-%d %H:%M:%S'), row['type'])
            _item2 = (row['imei_id'], '0', (row['hour']+timedelta(hours=1)).strftime('%Y-%m-%d %H:%M:%S'), '0')
            return [_item1, _item2]
        else:
            return [(row['imei_id'], row['agent_id'], row['hour'].strftime('%Y-%m-%d %H:%M:%S'), row['type'])]
    _count = int(row['hour_diff'])
    if _count <= 1:
        _x = [(row['imei_id'], row['agent_id'], row['hour'].strftime('%Y-%m-%d %H:%M:%S'), row['type'])]
    elif _count > 1 and _count <= 24:
        dt_list = pd.date_range(start=row['pre_hour'],end=row['hour'], freq='1h').to_pydatetime().tolist()[0:-1]
        _list = [(row['imei_id'], row['pre_agent_id'], dt_list[i].strftime('%Y-%m-%d %H:%M:%S'), row['pretype']) for i in range(1, _count)]
        _x = _list + [(row['imei_id'], row['agent_id'], row['hour'].strftime('%Y-%m-%d %H:%M:%S'), row['type'])]
    else:
        _item1 = (row['imei_id'], '0', (row['pre_hour']+timedelta(hours=1)).strftime('%Y-%m-%d %H:%M:%S'), '0')
        _item2 = (row['imei_id'], row['agent_id'], row['hour'].strftime('%Y-%m-%d %H:%M:%S'), row['type'])
        _x = [_item1, _item2]
    if not row['next_agent_id'] or row['next_agent_id']=='null':
        _x = _x + [(row['imei_id'], '0', (row['hour']+timedelta(hours=1)).strftime('%Y-%m-%d %H:%M:%S'), '0')]
    return _x

def _map_type_hour_ppl(x):
    agent_id, hour, t = x[0].split(';')
    return (hour, ((agent_id, t), x[1]))
#取回验证正确性（to_df
def _to_list(a):
    return [a]

def _append(a, b):
    a.append(b)
    return a

def _extend(a, b):
    a.extend(b)
    return a


df1 = spark.read.parquet('shanghai_region_type_df1.parquet')
df2 = spark.read.parquet('shanghai_region_type_df2.parquet')
df_shanghai = spark.read.parquet('shanghai_region_type.parquet')
# fill out all the records
rdd_shanghai = cal_full_records_max(df1, df2, df_shanghai)
# region rdd
region_rdd =rdd_shanghai.map(lambda a: _map_row_type(a)).reduceByKey(lambda a,b: a+b)
# region type
# cal region ppl number vectors
ppl_region_rdd = region_rdd.map(lambda a: _map_type_hour_ppl(a)).combineByKey(_to_list, _append, _extend)
ppl_region = ppl_region_rdd.collect()
# sort by ts
ppl_region.sort(key=lambda x: x[0])
ppl_region_dict = {}
for i in range(22, 191):
    _l = [[0 for k in range(8)] for j in range(5081)]
    for p in ppl_region[i][1]:
        _x, _y = p[0]
        if int(_x) <= 0:
            continue
        _l[int(_x)][int(_y)-1] = int(p[1])
    ppl_region_dict[ppl_region[i][0]] = np.matrix(_l)

with open('region_type_portion.pkl', 'wb') as wf:
    pickle.dump(ppl_region_dict, wf)
