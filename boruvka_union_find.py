import time
from pyspark import SparkContext
sc = SparkContext('yarn')

from pyspark.sql import SparkSession
spark = SparkSession \
    .builder \
    .appName("Boruvka_MST_algorithm") \
    .getOrCreate()
sc.addPyFile("s3a://rogerzhuo/graphframes-0.6.0-spark2.3-s_2.11.jar")

from graphframes import *
from pyspark.sql.functions import *

path_to_file="s3a://rogerzhuo/4elt.edges"
edge_dataframe = sc.textFile(path_to_file)

edge_dataframe = edge_dataframe.map(lambda line: line.split()).map(lambda edge:
                                                                   (edge[0], int(edge[1]), int(edge[2])))

begin_vex = edge_dataframe.map(lambda line: (line[0], line[0])).distinct()


v = spark.createDataFrame(begin_vex.collect(), ["id", "label"])
e = spark.createDataFrame(edge_dataframe.collect(), ["src", "dst", "weight"])

# Create a GraphFrame
g = GraphFrame(v, e)


mst = spark.createDataFrame([['', '', '']], ["src", "dst", "weight"])


# ! QucikFind (union-find set)
class QuickFind(object):
    id = []
    count = 0

    def __init__(self, n):
        self.count = n
        i = 0
        while i < n:
            self.id.append(i)
            i += 1

    def connected(self, p, q):
        return self.find(p) == self.find(q)

    def find(self, p):
        return self.id[p]

    def union(self, p, q):
        idp = self.find(p)
        if not self.connected(p, q):
            for i in range(len(self.id)):
                if self.id[i] == idp:
                    self.id[i] = self.id[q]
            self.count -= 1

    def union_connectedComponents(self, rdd_edges):
        edges = rdd_edges.collect()
        for item in edges:
            self.union(item[0], item[1])


start = time.clock()
while g.vertices.select('label').distinct().count() > 1:
    filter_df = g.find(
        "(a)-[e]->(b)").filter("a.label != b.label").select("e.*")
    filter_df.cache()

    inter_graph = GraphFrame(g.vertices, filter_df)

    min_edges = inter_graph.triplets.groupBy('src.label').agg(
        {'edge.weight': 'min'}).withColumnRenamed('min(edge.weight AS `weight`)', 'min_weight')

    final_edges = min_edges.join(inter_graph.triplets, (min_edges.label == inter_graph.triplets.src.label)
                                 & (min_edges.min_weight == inter_graph.triplets.edge.weight)) \
                           .select(col('src.id').alias('src'), col('dst.id').alias('dst'), col('min_weight').alias('weight'))
    final_edges.cache()

    edges_rdd = final_edges.rdd.map(
        lambda item: (int(item.src), int(item["dst"])))

    # ! save MST result
    mst = mst.union(final_edges).distinct().filter("src != ''")
    mst.cache()

    num_edges = final_edges.select('src').distinct().count()

    # ! union-find labeling
    qf = QuickFind(num_edges)
    qf.union_connectedComponents(edges_rdd)

    # ! generate final graphgrame via new label data
    connected_rdd = sc.parallelize([str(x) for x in qf.id], 4).zipWithIndex().map(
        lambda item: (item[1], item[0]))
    id_rdd = sc.parallelize(range(num_edges), 4).zipWithIndex().map(
        lambda item: (item[1], item[0]))
    connected_df = spark.createDataFrame(connected_rdd, ['id_inx', 'label'])
    id_df = spark.createDataFrame(id_rdd, ['id_inx', 'id'])
    vertis = id_df.join(connected_df, 'id_inx').select('id', 'label')
    # ! According to new label data, we generate new graphframe
    g = GraphFrame(vertis, g.edges)
    g.cache()

elapsed = (time.clock() - start)
print(elapsed)