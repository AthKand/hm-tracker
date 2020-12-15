import networkx as nx
from matplotlib import pyplot as plt
from tools import mask

import numpy as np


def maze_graph():
    flower_graph = {1: [2, 7],
                2: [1, 3],
                3: [2, 4, 9],
                4: [3, 5],
                5: [4, 11],
                6: [7, 13],
                7: [6, 1, 8],
                8: [7, 9, 15],
                9: [3, 8, 10],
                10: [9, 11, 17],
                11: [5, 10, 12],
                12: [11, 19],
                13: [6, 14],
                14: [13, 15, 20],
                15: [8, 14, 16],
                16: [15, 17, 22],
                17: [10, 16, 18],
                18: [17, 19, 24],
                19: [12, 18],
                20: [14, 21],
                21: [20, 22],
                22: [16, 21, 23],
                23: [22, 24],
                24: [18, 23]}
    
    island_prefixes = ['1', '2', '3', '4']
    
    bridge_edges = [('124', '201'),
                ('121', '302'),
                ('223', '404'),
                ('324', '401'),
                ('305', '220')]

    graph_prototype= {}
    
    for letter in island_prefixes:
        for node_suffix, edges in flower_graph.items():
            if node_suffix < 10:
                first_point = letter +'{}{}'.format(0, str(node_suffix))
            else:
                first_point = letter +'{}'.format(node_suffix)
            edge_list = []
            for n in edges:
                if n < 10:
                    second_point = letter + '{}{}'.format(0, str(n))
                else:
                    second_point = letter + '{}'.format(n)
                edge_list.append(second_point)
            graph_prototype[first_point] = edge_list
        
    mg = nx.from_dict_of_lists(graph_prototype)
    mg.add_edges_from(bridge_edges)

    return mg


def path_graph(inputfile):
    pg = nx.DiGraph()
    node_list = []
    
    with open(inputfile) as file:
        lines = file.read().splitlines(keepends = False)
        for line in lines:
            for node in line.split(','):
                pg.add_node(node)
                node_list.append(node)
            break
        for num, edge_points in enumerate(node_list):
            if num:
                pg.add_edge(node_list[num - 1], node_list[num])
            
    return pg, node_list


if __name__ == "__main__":
    nodelist = 'new_new.csv'
    inputfile = 'trial_shiz/2020-05-05_04-05_5.txt'
    
    nodes_dict = mask.create_node_dict(nodelist)
    G = maze_graph()
    H, nl= path_graph(inputfile)
    
    with plt.xkcd():
        nx.draw(G, pos = nodes_dict)
        nx.draw_networkx_edges(H, nodes_dict, width = 2, edge_color = 'orange')
        nx.draw_networkx_nodes(H, nodes_dict, nodelist = [nl[0]], node_color = 'g')
        nx.draw_networkx_nodes(H, nodes_dict, nodelist = [nl[-1]], node_color = 'r')
    plt.show()

    





