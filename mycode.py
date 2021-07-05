import sys
import math
from pyspark.sql import SparkSession
from pyspark import SparkContext, SparkConf

# k-means variables.
k = 5
MAX_ITERATIONS = 3

def haversineDist(point, centroid):
	pointLon, pointLat = point
	centrLon, centrLat = centroid

	dLat = math.radians(centrLat - pointLat)
	dLon = math.radians(centrLon - pointLon)

	a = math.pow(math.sin(dLat/2), 2) + math.cos(math.radians(pointLat))*math.cos(math.radians(centrLat))*math.pow(math.sin(dLon/2), 2)
	c = 2*math.atan2(math.sqrt(a), math.sqrt(1 - a))
	
	return 6371*c

def minDist(point, centroids):
	res = haversineDist(point, centroids[0])
	ind = 0

	for i in range(1, k):
		temp = haversineDist(point, centroids[i])
		if(temp < res):
			res = temp
			ind = i

	return ind, (point[0], point[1], 1) 

def pairing(line):
	fields = line.split(',')
	lon, lat = fields[3:5]

	return float(lon), float(lat) 


if __name__ == '__main__':
	conf = SparkConf().setAppName('MapReduce_k-means').setMaster('spark://master:7077')#.setSparkHome(...)
	sc = SparkContext(conf = conf)
	data = sc.textFile('hdfs://master:9000/yellow_tripdata_1m.csv')

	points = data.map(pairing).filter(lambda tmp: (tmp[0] != 0 and tmp[1] != 0))

	centroids = points.take(k)
	print('centroids', centroids)

	for i in range(MAX_ITERATIONS):
		centroids = points.map(lambda point: minDist(point, centroids))\
			.reduceByKey(lambda tmp1, tmp2: (
				(tmp1[0] + tmp2[0]), 
				(tmp1[1] + tmp2[1]), 
				(tmp1[2] + tmp2[2])))\
			.map(lambda (k,(lon, lat, c)): (lon/c, lat/c))\
			.coalesce(1, True)\
			.collect()
		print(centroids)
	
	sc.parallelize(centroids, 1).saveAsTextFile('hdfs://master:9000/centroids')