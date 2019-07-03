from pyclustering.cluster import cluster_visualizer_multidim
from pyclustering.cluster.optics import optics, ordering_analyser, ordering_visualizer
from pyclustering.samples.definitions import FCPS_SAMPLES
from pyclustering.utils import read_sample
# Read sample for clustering from some file.
sample = read_sample("../dataset/filteredDiffMaterial.txt")
# Run cluster analysis where connectivity radius is bigger than real.
radius = 6
neighbors = 4
optics_instance = optics(sample, radius, neighbors)
# Performs cluster analysis.
optics_instance.process()
# Obtain results of clustering.
clusters = optics_instance.get_clusters()
noise = optics_instance.get_noise()
ordering = optics_instance.get_ordering()
print(len(noise))
print(clusters)
# Visualize clustering results.
visualizer = cluster_visualizer_multidim()
visualizer.append_clusters(clusters, sample)
visualizer.show()
# Display ordering.
analyser = ordering_analyser(ordering)
ordering_visualizer.show_ordering_diagram(analyser, 2)
