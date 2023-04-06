import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from ripser import ripser
from persim import plot_diagrams

def getData(matrixfilename, regionfilename, xyzfilename):
    adjMatrix = np.loadtxt(matrixfilename)
    with open(regionfilename) as f:
        names = [line.rstrip() for line in f]
    labels = {}
    for i in range(len(names)):
        labels[i] = names[i]
    coords = np.loadtxt(xyzfilename)
    return adjMatrix, labels, coords


def plotHomologies(adjMatrix, title, picFilename):
    distMatrix = np.ones_like(adjMatrix) - adjMatrix
    result = ripser(distMatrix, distance_matrix=True, maxdim=3, do_cocycles=True, thresh=1)
    diagrams = result['dgms']
    plot_diagrams(diagrams, show=False)
    plt.title(title)
    plt.savefig(picFilename, dpi=200)
    plt.show()
    plt.close()

def plotCarpet(adjMatrix, picFilename, thereshold = -1):
    plt.style.use('_mpl-gallery-nogrid')
    plt.set_cmap("gist_heat")
    fig, ax = plt.subplots()
    im =  ax.imshow(np.where(adjMatrix < thereshold, 0, adjMatrix))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    plt.savefig(picFilename, dpi=200)
    plt.show()
    plt.close()

def makeGraph(adjMatrix, thereshold):
    A = np.where(adjMatrix < thereshold, 0, adjMatrix)
    Graph = nx.from_numpy_matrix(A)
    w_max = np.max(adjMatrix)
    return Graph, w_max

def plotComponentsStatistics(adjMatrix, picFilename):
    T = np.linspace(0, 1.2, num = 20)
    comp = []
    for t in T:
        G, w = makeGraph(adjMatrix, t)
        comp.append(nx.number_connected_components(G))
    plt.plot(T, comp)
    plt.title('Components Statistics')
    plt.xlabel('thereshold')
    plt.ylabel('number of components')
    plt.savefig(picFilename, dpi=200)
    plt.show()
    plt.close()

def plotCliqueStatistics(adjMatrix, picFilename):
    T = np.linspace(-0.1, 1.2, num=20)
    clique = []
    for t in T:
        G, w = makeGraph(adjMatrix, t)
        c, weight = nx.max_weight_clique(G, weight = None)
        clique.append(weight)
    plt.plot(T, clique)
    plt.title('Clique Statistics')
    plt.xlabel('thereshold')
    plt.ylabel('max weight of clique')
    plt.savefig(picFilename, dpi=200)
    plt.show()
    plt.close()


def plotGraph(Graph, w_max, picFilename, names, label = False):
    pos = nx.spring_layout(Graph)
    edge_widths = []
    edge_colors = []
    for u, v, weight in Graph.edges.data("weight"):
        if weight is not None:
            edge_widths.append(weight * 0.5)
            edge_colors.append(weight/w_max)

    communities = list(nx.community.greedy_modularity_communities(Graph, resolution=2))
    N = len(communities)
    node_colors = [0 for _ in nx.nodes(G)]
    for i in range(N):
        for node in communities[i]:
            node_colors[node] = i / N

    nx.draw_networkx_nodes(Graph, pos, node_size=10, node_color=node_colors, cmap="gnuplot")
    nx.draw_networkx_edges(Graph, pos, width=edge_widths, edge_color=edge_colors)
    if label:
        nx.draw_networkx_labels(Graph, pos, names, font_size=4, alpha=0.7)

    plt.savefig(picFilename, dpi=200)
    plt.show()
    plt.close()

def plotGraph3D(Graph,w_max, Coords, edges = False):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    # Extract node and edge positions from the layout
    node_xyz = Coords

    # Clustering
    communities = list(nx.community.greedy_modularity_communities(Graph, resolution=2))
    N = len(communities)
    node_colors = [0 for _ in nx.nodes(G)]
    for i in range(N):
        for node in communities[i]:
            node_colors[node] = i / N


    edge_xyz = np.array([(node_xyz[u], node_xyz[v]) for u, v, weight in Graph.edges.data("weight") if weight>w_max*0.75])
    ax.scatter(*node_xyz.T,c=node_colors,s=100, ec="w")
     #Plot the edges
    if edges:
        for vizedge in edge_xyz:
            ax.plot(*vizedge.T, color="tab:gray")

    """Visualization options for the 3D axes."""
    # Turn gridlines off
    ax.grid(False)
    # Suppress tick labels
    for dim in (ax.xaxis, ax.yaxis, ax.zaxis):
        dim.set_ticks([])
    # Set axes labels
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    fig.tight_layout()
    plt.show()


A, labels, cords = getData('connectmat.txt', 'region_abbrevs.txt', 'region_xyz_centers.txt')
G, norm = makeGraph(A, 0.3)

#plotGraph(G, norm, 'BrainGraph.png', labels)
plotCarpet(A, 'carpet.png')
#plotComponentsStatistics(A, 'components.png')
#plotHomologies(A, 'BrainHomologies','BrainHomologies.png')
#plotCliqueStatistics(A, 'clique.png')
#plotGraph3D(G, norm, cords, edges=True)


