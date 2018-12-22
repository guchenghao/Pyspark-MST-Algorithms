import networkx as ne
import matplotlib.pyplot as mp
import random
import pandas as pd
rg = ne.connected_caveman_graph(50, 90) # Generate connected caveman graph, which has no weights
A = ne.adjacency_matrix(rg) # Transfer it into adjacency matrix
# ps = ne.shell_layout(rg)
# ne.draw(rg, ps, with_labels=False, node_size=30)
# mp.show()
weights = random.sample(list(range(10*len(ne.edges(rg)))), len(ne.edges(rg))) # Generate weights randomly
B = A.todense()
count = 0
data = []
# Assign weights to edges
for i in range(B.shape[0]):
    for j in range(i):
        if B[i, j] > 0:
            B[i, j] = weights[count]
            count += 1
            B[j, i] = B[i, j]
            data.append([i, j, B[i, j]])
            data.append([j, i, B[j, i]])
print(count) # record the number of edges
data = pd.DataFrame(data=data, columns=['src', 'dst', 'weight'])
data.to_csv('edges_generate_7.csv', index=False) # Store the graph in csv file
