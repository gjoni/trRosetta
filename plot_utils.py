from itertools import groupby
from collections import defaultdict
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
def get_ss_elements(ss):
    ss_count = defaultdict(int)
    ss_elements = []
    last_index = 0
    for ss_type, size in [(k, len(list(g))) for k, g in groupby(ss)]:
        end = last_index + size
        ss_count[ss_type] += 1
        if ss_type == 'E' or ss_type == 'H':
            ss_elements.append(('{}{}'.format(ss_type, ss_count[ss_type]), last_index, end))
        last_index = end

    return ss_elements

def find_pos_in_ss_elements(pos, elements):
    for e, start, end in elements:
        if start <= pos < end:
            return e
        if end >= pos:
            return '-'

def contacts_to_ss_element_graph(contacts, ss_elements):
    G = nx.Graph()
    for ss_e in ss_elements:
        G.add_node(ss_e[0])
    for i, j in contacts:
        e1, e2 = find_pos_in_ss_elements(i,ss_elements), find_pos_in_ss_elements(j,ss_elements)
        if e1 =='-' or e2=='-': 
            continue
        if (e1,e2) in G.edges:
            G.edges[e1,e2]['weight'] += 1
        else:
            G.add_edges_from([(e1, e2, {'weight':1})])

    return G
        
def plot_contacts(true_non_local_contacts, correctly_predicted_non_local_contacts, non_local_false_positive_contacts, ss_elements, ax1, ax2):

    G1 = contacts_to_ss_element_graph(true_non_local_contacts, ss_elements)
    
    nodes = G1.nodes if len(G1.nodes) == 0 else []
    edges = G1.edges if len(G1.edges) == 0 else []
    node_colors = ['red' if n[0] == 'H' else 'yellow' for n in nodes]

    edge_labels = dict([((i,j), G1.edges[i,j]['weight']) for i,j in edges])
    weights = [2*G1[u][v]['weight'] for u, v in G1.edges()]

    pos = nx.planar_layout(G1)
    nx.draw(G1, with_labels=True, font_weight='bold', node_size=1000, pos=pos, node_color=node_colors, ax=ax1)

    max_value = max([np.abs(n) for n in edge_labels.values()])

    nx.draw_networkx_edge_labels(G1,pos,edge_labels=edge_labels, ax=ax1)
    nx.draw_networkx_edges(G1, pos, 
                           edgelist=G1.edges(), edge_color=edge_labels.values(),
                           alpha=0.5, edge_vmin=-1*max_value, edge_vmax=max_value, edge_cmap=plt.cm.coolwarm_r, width=weights, ax=ax1)
    

    G2 = contacts_to_ss_element_graph(correctly_predicted_non_local_contacts, ss_elements)

    G3 = nx.Graph()
    for i, j in non_local_false_positive_contacts:
        e1, e2 = find_pos_in_ss_elements(i,ss_elements), find_pos_in_ss_elements(j,ss_elements)
        if e1 =='-' or e2=='-': 
            continue
        if (e1,e2) in G3.edges:
            G3.edges[e1,e2]['weight'] -= 1
        else:
            G3.add_edges_from([(e1, e2, {'weight':-1})])
    for e in G3.edges:
        ax2.annotate("",
                    xy=pos[e[0]], xycoords='data',
                    xytext=pos[e[1]], textcoords='data',
                    arrowprops=dict(arrowstyle="-", linestyle='--', color='red',linewidth=2,
                                    shrinkA=6, shrinkB=6,
                                    patchA=None, patchB=None,
                                    connectionstyle="arc3,rad=.35"
                                    ),
                    )




    node_colors = ['red' if n[0] == 'H' else 'yellow' for n in G2.nodes]

    edge_labels = {}
    for i, j in set(list(G2.edges) + list(G3.edges)):
        tp = G2.edges[i,j]['weight'] if (i,j) in G2.edges else 0
        fp = G3.edges[i,j]['weight'] if (i,j) in G3.edges else 0


        edge_labels[(i,j)] = '{}/{}'.format(tp, fp) if fp else str(tp)
    weights = [2*G2[u][v]['weight'] for u, v in G2.edges()]

    nx.draw(G2, with_labels=True, font_weight='bold', node_size=1000, pos=pos, node_color=node_colors, ax=ax2)

#     max_value = max([np.abs(n) for n in edge_labels.values()])
    max_value = max([np.abs(n) for n in weights])


    nx.draw_networkx_edge_labels(G2,pos,edge_labels=edge_labels, ax=ax2)
    nx.draw_networkx_edges(G2, pos, 
                           edgelist=G2.edges(), edge_color=weights,
                           alpha=0.5, edge_vmin=-1*max_value, edge_vmax=max_value, edge_cmap=plt.cm.coolwarm_r, width=weights, ax=ax2)
    
    ax1.collections[0].set_edgecolor("#000000")
    ax2.collections[0].set_edgecolor("#000000")

