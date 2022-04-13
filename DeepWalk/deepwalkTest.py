from getDataset import *

from tqdm import tqdm
from collections import defaultdict
import numpy as np
import random
import networkx as nx
import matplotlib.pyplot as plt
from gensim.models import Word2Vec

global G
global nbrs_with_weight_dict

def random_walk_with_weight(walk_length, start_node):
    """ random walk with weight starts at start_node """
    global nbrs_with_weight_dict
    walk = [start_node]
    while len(walk) < walk_length:
        cur = walk[-1]
        cur_nbrs_with_weight = nbrs_with_weight_dict[cur]
        if len(cur_nbrs_with_weight) > 0:
            walk.append(random.choice(cur_nbrs_with_weight))
        else:
            break
    return walk


def random_walk(walk_length, start_node):
    """ random walk starts at start_node """
    walk = [start_node]
    while len(walk) < walk_length:
        cur = walk[-1]
        cur_nbrs = list(G.neighbors(cur))
        if len(cur_nbrs) > 0:
            walk.append(random.choice(cur_nbrs))
        else:
            break
    return walk


def _simulate_walks(nodes, num_walks, walk_length, with_weight_flag=False):
    """ generate random walk sequence """
    global nbrs_with_weight_dict
    walks = []
    if with_weight_flag:
        # generate the nbrs_with_weight_dict
        nbrs_with_weight_dict = defaultdict(list)
        edge_weights = nx.get_edge_attributes(G, "weight")
        for node in G.nodes():
            nbrs_with_weight = []
            for nbr in list(G.neighbors(node)):
                edge_weight = edge_weights[(node, nbr)]
                nbrs_with_weight.extend([nbr]*int(edge_weight))
            nbrs_with_weight_dict[node] = nbrs_with_weight
        # random walk consider the weight
        for _ in range(num_walks):
            random.shuffle(nodes)
            for v in nodes:
                walks.append(random_walk_with_weight(walk_length=walk_length, start_node=v))
    else:
        for _ in range(num_walks):
            random.shuffle(nodes)
            for v in nodes:
                walks.append(random_walk(walk_length=walk_length, start_node=v))
    return walks


def draw_graph(G, weight_edge_flag=False, save_file_path='Graph.png', pos=None):
    """ generate a viewable view of the graph """
    print('drawing the graph...')
    plt.rcParams['figure.figsize']= (40, 40)
    # positions for all nodes
    if not pos:
        pos = nx.shell_layout(G) 
    # draw nodes and labels
    nx.draw_networkx_nodes(G, pos, node_size=700, alpha=0.7)
    nx.draw_networkx_labels(G, pos, font_size=10, font_color='white', font_weight='bold')
    # draw edges
    if not weight_edge_flag:
        nx.draw_networkx_edges(G, pos, edgelist=G.edges, width=0.8, edge_color='red', alpha=0.7)
    else:
        # weights for all edges
        weights = nx.get_edge_attributes(G, 'weight')
        # # change weights to int
        # for edge in weights.keys():
        #     weights[edge] = int(weights[edge])
        weightmax = max(list(weights.values()))
        for edge in G.edges(data=True):
            w = edge[2]['weight']
            nx.draw_networkx_edges(G, pos, edgelist=[edge], width=w/weightmax*8, edge_color='red', alpha=0.7)
        
    plt.savefig(save_file_path, dpi=500)
    print('drawing done. The drawing is saved in [' + save_file_path +']')
    #plt.show()
    plt.close('all')
    return pos


def draw_walks(G, walks, save_file_path='Graph_Walks.png', pos=None):
    """ generate a viewable view of the random walks of graph """
    print('drawing the graph of walks...')
    plt.rcParams['figure.figsize']= (40, 40)
    # positions for all nodes
    if not pos:
        pos = nx.shell_layout(G) 
    # draw nodes and labels
    nx.draw_networkx_nodes(G, pos, node_size=700, alpha=0.7)
    nx.draw_networkx_labels(G, pos, font_size=10, font_color='white', font_weight='bold')
    # draw edges
    walk2weight = defaultdict(int)
    for walk in walks:
        if len(walk) < 2:
            continue
        for i in range(len(walk)-1):
            edge = (walk[i], walk[i+1])
            if edge in walk2weight:
                walk2weight[edge] += 1
            else:
                walk2weight[edge] = 1
    # weights for all edges
    weightmax = max(list(walk2weight.values()))
    for edge in walk2weight:
        w = walk2weight[edge]
        nx.draw_networkx_edges(G, pos, edgelist=[edge], width=w/weightmax*8, edge_color='red', alpha=0.7)
    plt.savefig(save_file_path, dpi=500)
    print('drawing done. The drawing is saved in [' + save_file_path +']')
    #plt.show()
    plt.close('all')
    return pos


def save_walks(walks, save_file_path='walks.txt'):
    """ save walks to file """
    save_str = ''
    print('write walks to the file: ')
    for walk in tqdm(walks):
        for node in walk:
            save_str += str(node) + ' '
        save_str += '\n'
    f = open(save_file_path, 'w')
    f.write(save_str)
    f.close()
    print('writing done. The walks is saved in [' + save_file_path +']')
    return save_file_path


def test():
    global G,nbrs_with_weight_dict
    # Generate Graph
    print('--------- Generate Graph ---------')
    # get the graph file from original dataset file
    dataset = getDatasetFromFile('./dataset/Enron_TimeFromTo.txt')
    simplest_graph, simplest_graph_file = getSimplestGraph(dataset)
    weight_graph, weight_graph_file = getWeightGraph(dataset)
    # use networkx to create graph from file
    G_s = nx.read_edgelist(simplest_graph_file, create_using=nx.DiGraph())
    G_w = nx.read_weighted_edgelist(weight_graph_file, create_using=nx.DiGraph())
    

    # Random Walk of Graph
    print('--------- Random Walks from Graph ---------')
    # set the parameters
    num_walks, walk_length = 80, 30
    # [graph G_s]
    G = G_s
    print('** random walk without weight')
    # draw the graph
    pos = draw_graph(G, weight_edge_flag=False, save_file_path='./deepwalk/figs/Graph_Simplest_test.png')
    # random walk of graph
    walks_s = _simulate_walks(list(G.nodes()), num_walks=num_walks, walk_length=walk_length, with_weight_flag=False)
    # save walks to file 
    save_walks(walks_s, save_file_path='./deepwalk/walks/walks_s_test.txt')
    # draw the deepwalk
    draw_walks(G, walks_s, save_file_path='./deepwalk/figs/Graph_Walks_G_s_test.png', pos=pos)
    # [graph G_w]
    G = G_w
    print('** random walk with weight')
    # draw the graph
    pos = draw_graph(G, weight_edge_flag=True, save_file_path='./deepwalk/figs/Graph_Weight_test.png', pos=pos)
    # deepwalk of graph
    walks_w = _simulate_walks(list(G.nodes()), num_walks=num_walks, walk_length=walk_length, with_weight_flag=True)
    # save walks to file 
    save_walks(walks_w, save_file_path='./deepwalk/walks/walks_w_test.txt')
    # draw the deepwalk
    draw_walks(G, walks_w, save_file_path='./deepwalk/figs/Graph_Walks_G_w_test.png', pos=pos)

    exit()



if __name__ == '__main__':
    # test()

    # Generate Graph
    print('--------- Generate Graph ---------')
    # get the graph file from original dataset file
    dataset = getDatasetFromFile('./dataset/Enron_TimeFromTo.txt')
    simplest_graph, simplest_graph_file = getSimplestGraph(dataset)
    weight_graph, weight_graph_file = getWeightGraph(dataset)
    # use networkx to create graph from file
    G_s = nx.read_edgelist(simplest_graph_file, create_using=nx.DiGraph())
    G_w = nx.read_weighted_edgelist(weight_graph_file, create_using=nx.DiGraph())
    

    # Random Walk of Graph
    print('--------- Random Walks from Graph ---------')
    # set the parameters
    num_walks, walk_length = 80, 30
    # [graph G_s]
    G = G_s
    print('** random walk without weight')
    # draw the graph
    pos = draw_graph(G, weight_edge_flag=False, save_file_path='./deepwalk/figs/Graph_Simplest.png')
    # random walk of graph
    walks_s = _simulate_walks(list(G.nodes()), num_walks=num_walks, walk_length=walk_length, with_weight_flag=False)
    # save walks to file 
    save_walks(walks_s, save_file_path='./deepwalk/walks/walks_s.txt')
    # draw the deepwalk
    draw_walks(G, walks_s, save_file_path='./deepwalk/figs/Graph_Walks_G_s.png', pos=pos)
    # [graph G_w]
    G = G_w
    print('** random walk with weight')
    # draw the graph
    pos = draw_graph(G, weight_edge_flag=True, save_file_path='./deepwalk/figs/Graph_Weight.png', pos=pos)
    # deepwalk of graph
    walks_w = _simulate_walks(list(G.nodes()), num_walks=num_walks, walk_length=walk_length, with_weight_flag=True)
    # save walks to file 
    save_walks(walks_w, save_file_path='./deepwalk/walks/walks_w.txt')
    # draw the deepwalk
    draw_walks(G, walks_w, save_file_path='./deepwalk/figs/Graph_Walks_G_w.png', pos=pos)
   

    # using word2vec to get the embedding of every nodes
    print('--------- Word2Vec from Random Walks ---------')
    # set the parameters
    vector_size, window, sg, hs = 10, 5, 1, 1
    w2v_model_s = Word2Vec(walks_s, vector_size=vector_size, window=window, sg=sg, hs=hs)
    w2v_model_w = Word2Vec(walks_w, vector_size=vector_size, window=window, sg=sg, hs=hs)
    # 打印其中一个节点的嵌入向量
    print('embedding of node [24]:')
    print(' - graph without weight:')
    print(w2v_model_s.wv['24'])
    print(' - graph with weight')
    print(w2v_model_w.wv['24'])
    




