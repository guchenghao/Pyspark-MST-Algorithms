from pyspark.sql import SparkSession
import time
from pyspark import SparkContext
sc = SparkContext('yarn')
sc.addPyFile("s3a://rogerzhuo/graphframes-0.6.0-spark2.3-s_2.11.jar")
from pyspark.sql.functions import *
from graphframes import *

spark = SparkSession.builder.appName("Prime_algorithm").getOrCreate()

# Prepare data.
v1 = spark.createDataFrame([
    (0,),
    (1,),
    (2,),
    (3,),
    (4,),
    (5,),
    (6,),
    (7,),
    (8,),
    (9,)], ["id"])

# Edges DataFrame
e1 = spark.createDataFrame([
    (1, 2, 1),
    (2, 3, 7),
    (1, 9, 5),
    (1, 8, 10),
    (9, 0, 2),
    (9, 5, 6),
    (8, 4, 4),
    (4, 5, 11),
    (2, 4, 9), (3, 4, 16), (4, 6, 13), (3, 6, 17),
    (6, 7, 18), (5, 7, 19), (4, 7, 15),
    (5, 0, 3)], ["src", "dst", "distance"])
e1.show()

e2 = spark.createDataFrame([
    (2, 1, 1),
    (3, 2, 7),
    (9, 1, 5),
    (8, 1, 10),
    (0, 9, 2),
    (5, 9, 6),
    (4, 8, 4),
    (5, 4, 11),
    (4, 2, 9), (4, 3, 16), (6, 4, 13), (6, 3, 17),
    (7, 6, 18), (7, 5, 19), (7, 4, 15),
    (0, 5, 3)], ["src", "dst", "distance"])

e3 = e1.union(e2)
e3.show()
g = GraphFrame(v1, e3)
e1 = g.edges


# Randomly select vertex 1 and put it into source dataframe.
r = g.edges.filter("src = 1").sort(g.edges.distance).take(1)
source = sc.parallelize(((r[0][0],), (r[0][1],))
                        ).toDF().withColumnRenamed('_1', 'src')

temp1 = g.edges.select('dst')
temp2 = g.edges.select('src')

# Other vertexes will be put into dest dataframe.
dest = temp1.union(temp2).distinct()
dest = dest.subtract(source)
print(dest.count())
source.show()
dest.show()
ini = [r[0][0], r[0][1], r[0][2]]


start = time.clock()
l = []
while dest.count() > 0:
    # Join three dataframes.
    temp_e1 = source.join(e1, "src")
    temp_e1 = temp_e1.join(dest, "dst")
    # Found the minmimum weight.
    r = temp_e1.rdd.reduce(lambda a, b: a if (a[2] < b[2]) else b)
    l.append([r[0], r[1], r[2]])
    # update the source and dest datarame
    t_source = sc.parallelize(
        ((r[0],), (r[1],))).toDF().withColumnRenamed('_1', 'src')
    source = source.union(t_source).distinct()
    dest = dest.subtract(t_source)
    temp_e1.unpersist()

print((time.clock() - start))

# print MST tree.
l2 = [ini]+l
sub = spark.createDataFrame(l2, ["src", "dst", "distance"])
sub.show()
