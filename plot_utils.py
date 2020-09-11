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
        
def plot_contacts(true_non_local_contacts, correctly_predicted_non_local_contacts, non_local_false_positive_contacts, ss_elements, ax1, ax2):
    
    G1 = nx.Graph()
    for ss_e in ss_elements:
        G1.add_node(ss_e[0])
    for i, j in true_non_local_contacts:
        e1, e2 = find_pos_in_ss_elements(i,ss_elements), find_pos_in_ss_elements(j,ss_elements)
        if e1 =='-' or e2=='-': 
            continue
        if (e1,e2) in G1.edges:
            G1.edges[e1,e2]['weight'] += 1
        else:
            G1.add_edges_from([(e1, e2, {'weight':1})])


    node_colors = ['red' if n[0] == 'H' else 'yellow' for n in G1.nodes]

    edge_labels = dict([((i,j), G1.edges[i,j]['weight']) for i,j in G1.edges])
    weights = [2*G1[u][v]['weight'] for u, v in G1.edges()]

    pos = nx.planar_layout(G1)
    nx.draw(G1, with_labels=True, font_weight='bold', node_size=1000, pos=pos, node_color=node_colors, ax=ax1)

    max_value = max([np.abs(n) for n in edge_labels.values()])

    nx.draw_networkx_edge_labels(G1,pos,edge_labels=edge_labels, ax=ax1)
    nx.draw_networkx_edges(G1, pos, 
                           edgelist=G1.edges(), edge_color=edge_labels.values(),
                           alpha=0.5, edge_vmin=-1*max_value, edge_vmax=max_value, edge_cmap=plt.cm.coolwarm_r, width=weights, ax=ax1)
    

    
    G2 = nx.Graph()
    for ss_e in ss_elements:
        G2.add_node(ss_e[0])
    for i, j in correctly_predicted_non_local_contacts:
        e1, e2 = find_pos_in_ss_elements(i,ss_elements), find_pos_in_ss_elements(j,ss_elements)
        if e1 =='-' or e2=='-': 
            continue
        if (e1,e2) in G2.edges:
            G2.edges[e1,e2]['weight'] += 1
        else:
            G2.add_edges_from([(e1, e2, {'weight':1})])


    for i, j in non_local_false_positive_contacts:
        e1, e2 = find_pos_in_ss_elements(i,ss_elements), find_pos_in_ss_elements(j,ss_elements)
        if e1 =='-' or e2=='-': 
            continue
        if (e1,e2) in G2.edges:
            G2.edges[e1,e2]['weight'] -= 1
        else:
            G2.add_edges_from([(e1, e2, {'weight':-1})])

    node_colors = ['red' if n[0] == 'H' else 'yellow' for n in G2.nodes]

    edge_labels = dict([((i,j), G2.edges[i,j]['weight']) for i,j in G2.edges])
    weights = [2*G2[u][v]['weight'] for u, v in G2.edges()]

    # pos = nx.spring_layout(G, k=4)
    nx.draw(G2, with_labels=True, font_weight='bold', node_size=1000, pos=pos, node_color=node_colors, ax=ax2)

    max_value = max([np.abs(n) for n in edge_labels.values()])

    nx.draw_networkx_edge_labels(G2,pos,edge_labels=edge_labels, ax=ax2)
    nx.draw_networkx_edges(G2, pos, 
                           edgelist=G2.edges(), edge_color=edge_labels.values(),
                           alpha=0.5, edge_vmin=-1*max_value, edge_vmax=max_value, edge_cmap=plt.cm.coolwarm_r, width=weights, ax=ax2)
    
    ax1.collections[0].set_edgecolor("#000000")
    ax2.collections[0].set_edgecolor("#000000")
