import time
from pyspark import SparkContext
sc = SparkContext('local')

from pyspark.sql import SparkSession
spark = SparkSession \
    .builder \
    .appName("Boruvka_MST_algorithm") \
    .getOrCreate()
sc.addPyFile("graphframes-0.6.0-spark2.3-s_2.11.jar")

from graphframes import *
from pyspark.sql.functions import col

# ! read CSV file and format input RDD
path_to_file = "/Users/guchenghao/Desktop/BDC-5003/Team_project/edges_generate_2.csv"
edge_dataframe = sc.textFile(path_to_file, 8)

edge_dataframe = edge_dataframe.map(lambda line: line.split(',')).map(lambda edge:
                                                                   (edge[0], int(edge[1]), int(edge[2])))

begin_vex = edge_dataframe.map(lambda line: (line[0], line[0])).distinct()


v = spark.createDataFrame(begin_vex, ["id", "label"])
e = spark.createDataFrame(edge_dataframe, ["src", "dst", "weight"])

# ! Create a GraphFrame
g = GraphFrame(v, e)


mst = spark.createDataFrame([['', '', '']], ["src", "dst", "weight"])


def read_data(line):
    l_rdd = line.split(" ")
    return (int(l_rdd[0]), int(l_rdd[1]))


def largeStarMap(tup):
    return [(tup[0], tup[1]), (tup[1], tup[0])]

# ! large-star
def largeStarReduce(tup):
    node = int(tup[0])
    data = tup[1]
    m = min(data)
    m = min(m, node)
    to_return = []
    for v in data:
        if v > node:
            to_return.append((v, m))
    return to_return

# ! small-star
def smallStarMap(tup):
    u = tup[0]
    v = tup[1]

    if (u > v):
        return (u, v)
    else:
        return (v, u)


def smallStarReduce(tup):
    node = tup[0]
    data = [i for i in tup[1]]
    data.append(node)
    m = min(data)
    to_return = []
    for v in data:
        if v != m:
            to_return.append((v, m))
    return to_return


def outputEdges(tup):
    key = tup[0]
    val = tup[1]
    return str(key) + " " + str(val)

# ! check if algorithm convergence
def check_convergence(rdd1, rdd2):
    diff_rdd = rdd2.subtract(rdd1).union(rdd1.subtract(rdd2))
    if diff_rdd.count() == 0:
        return True
    return False


def Connnected_Components(rdd):
    # ! small-star
    while True:
        # ! large-star
        while True:
            prev_rdd = rdd
            rdd = rdd.flatMap(largeStarMap).distinct().groupByKey().flatMap(
                largeStarReduce)
            if check_convergence(prev_rdd, rdd):
                break
        prev_rdd = rdd
        rdd = rdd.map(smallStarMap).distinct().groupByKey().flatMap(smallStarReduce)
        if check_convergence(prev_rdd, rdd):
            break

    vals_rdd = rdd.values().distinct()
    rdd = rdd.union(vals_rdd.map(lambda k: (k, k)))

    # ! generate the final (id, label) pairs
    rdd = rdd.map(outputEdges)
    final_rdd = rdd.map(read_data)

    return final_rdd


start = time.clock()
while g.vertices.select('label').distinct().count() > 1:
    filter_df = g.find(
        "(a)-[e]->(b)").filter("a.label != b.label").select("e.*")
    filter_df.cache()

    inter_graph = GraphFrame(g.vertices, filter_df)
    # ! find each vertex`s minimum weight edge
    min_edges = inter_graph.triplets.groupBy('src.label').agg(
        {'edge.weight': 'min'}).withColumnRenamed('min(edge.weight AS `weight`)', 'min_weight')

    final_edges = min_edges.join(inter_graph.triplets, (min_edges.label == inter_graph.triplets.src.label)
                                 & (min_edges.min_weight == inter_graph.triplets.edge.weight)) \
                           .select(col('src.id').alias('src'), col('dst.id').alias('dst'), col('min_weight').alias('weight'))
    final_edges.cache()

    edges_rdd = final_edges.rdd.map(
        lambda item: (int(item.src), int(item["dst"])))

    mst = mst.union(final_edges).distinct().filter("src != ''")
    mst.cache()

    # ! Connected Components Labeling
    final_rdd = Connnected_Components(edges_rdd)

    if final_rdd.isEmpty():
        break

    vertis = spark.createDataFrame(final_rdd, ['id', 'label'])
    g = GraphFrame(vertis, g.edges)
    g.cache()

elapsed = (time.clock() - start)
print(elapsed)
mst.show()
