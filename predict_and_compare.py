#!/usr/bin/env python
import numpy as np
import pandas as pd
from Bio import PDB
from Bio.PDB.DSSP import DSSP
from Bio.PDB.vectors import rotaxis
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
from time import time
import seaborn as sns
from itertools import chain
from scipy.spatial import distance_matrix

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from os import path, system, environ
from subprocess import check_call, DEVNULL
from tempfile import NamedTemporaryFile

from plot_utils import get_ss_elements, plot_contacts, contacts_to_ss_element_graph

dir_path = path.dirname(path.realpath(__file__)) if '__file__' in locals() else '.'

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

TR_ROSETTA_DEFAULT_PATH = environ['TR_ROSETTA_PATH'] if 'TR_ROSETTA_PATH' in environ else dir_path

parser.add_argument('-c', '--chain', help='Chain to use for contact prediction and comparison')
parser.add_argument('-p', '--plot-distance-map', action='store_true', help='save plot of distance map as png')
parser.add_argument('-o', '--output', default='contacts.csv', help='Name of the output csv file')
parser.add_argument('--pymol', action='store_true', help='save pymol script that plots the contacts')

parser.add_argument('--tr-rosetta-path', default=TR_ROSETTA_DEFAULT_PATH, help='Path to where tr rosetta is installed, can also be set using the TR_ROSETTA_PATH environment variable')


parser.add_argument('pdbs', nargs='+', help='input pdbs')

args = parser.parse_args()
# args = parser.parse_args(['-p', '-c', 'A', '--pymol', 'ctc445/CTC-445.pdb'])
# args = parser.parse_args(['-p', '-c', 'A', '--pymol', 'ctc445/CTC-640.pdb'])



for pdb in args.pdbs:
    init_time = time()
    input_basename, _ = path.splitext(path.basename(pdb))
    # parse input pdb and prepare for prediction
    input_structure = PDB.PDBParser().get_structure('input_structure', pdb)
    dssp = DSSP(input_structure[0], pdb)
    ss = [dssp[k][2] for k in filter(lambda k: k[0] == args.chain if args.chain else True, dssp.keys())]

    if args.chain:
        input_structure = next(filter(lambda c: c.id ==args.chain, input_structure.get_chains()))
     
#     ca_atoms = list(filter(lambda a: a.name == 'CA', input_structure.get_atoms()))
    n_atoms = list(filter(lambda a: a.name == 'CA', input_structure.get_atoms()))
    c_atoms = list(filter(lambda a: a.name == 'CA', input_structure.get_atoms()))
    ca_atoms = list(filter(lambda a: a.name == 'CA', input_structure.get_atoms()))
    
    cb_xyz = []
    for n_at,c_at, ca_at in zip(n_atoms, c_atoms, ca_atoms):
        n = n_at.get_vector()
        c = c_at.get_vector()
        ca = ca_at.get_vector()

        n = n - ca
        c = c - ca
        rot = rotaxis(-1*np.pi * 120.0/180.0, c)
        cb_at_origin = n.left_multiply(rot)
        cb = cb_at_origin+ca
        cb_xyz.append(cb)    
    
    input_distance_matrix = pd.DataFrame(np.array([[np.linalg.norm(at2 - at1) for at2 in cb_xyz] for at1 in cb_xyz]))
    input_distance_matrix[input_distance_matrix >20] = 0



    ppb = PDB.Polypeptide.PPBuilder()
    input_sequence = ppb.build_peptides(input_structure).pop().get_sequence()

    # dump input sequence into a temporary file and run TrRosetta
    with NamedTemporaryFile('+w') as fasta:
        fasta.write('> input\n{}'.format(input_sequence))
        fasta.flush()
        with NamedTemporaryFile('r', suffix='.npz') as output_npz:
            print('Predicting contacts for {}'.format(pdb))
            init_pred_time = time()
            check_call(['python', '{}/network/predict.py'.format(args.tr_rosetta_path), '-m', '{}/model2019_07'.format(args.tr_rosetta_path), fasta.name, output_npz.name], stdout=DEVNULL, stderr=DEVNULL)
            print('Done predicting contacts for {} in {:.2f} seconds'.format(pdb, time() - init_pred_time))

            predictions = np.load(output_npz.name)
            distances = predictions['dist']
            bins = np.array([0, *np.arange(2,20,.5)])
            predicted_distance_matrix = pd.DataFrame((bins[np.argmax(distances, axis=2)]))

    rmsd_true_contacts = np.sqrt(((input_distance_matrix[(input_distance_matrix != 0.0) & (input_distance_matrix < 6)] - predicted_distance_matrix[(input_distance_matrix != 0.0) & (input_distance_matrix < 6)])**2))
    per_res_rmsd = (rmsd_true_contacts.sum() / pd.notna(rmsd_true_contacts).sum())


    # calculate relevant metrics
    rmsd = np.sqrt(np.concatenate((input_distance_matrix - predicted_distance_matrix)**2).sum()/(len(input_distance_matrix)**2))
    binned_input = np.digitize(input_distance_matrix, bins)
    logsum = np.sum(np.log([distances[i][j][binned_input[i][j]-1] for j in range(len(binned_input)) for i in range(len(binned_input))]))
    
    true_contacts = np.argwhere((input_distance_matrix[(input_distance_matrix < 6)] > 0.0).to_numpy())
    true_uniq_contacts = set([tuple(sorted((i,j))) for i,j in true_contacts])
    true_non_local_contacts = set(filter(lambda c: (c[1] - c[0]) >= 5, true_uniq_contacts))
    
    correctly_predicted_contacts = np.argwhere((predicted_distance_matrix[(input_distance_matrix < 6) & (predicted_distance_matrix < 6)] > 0.0).to_numpy())
    correctly_predicted_unique_contacts = set([tuple(sorted((i,j))) for i,j in correctly_predicted_contacts])
    correctly_predicted_non_local_contacts = set(filter(lambda c: (c[1] - c[0]) >= 5, correctly_predicted_unique_contacts))
    
    false_positive_contacts = np.argwhere((predicted_distance_matrix[((input_distance_matrix > 12)| (input_distance_matrix == 0)) & ((predicted_distance_matrix < 6) & (predicted_distance_matrix > 0))] > 0.0).to_numpy())
    uniq_false_positive_contacts = set([tuple(sorted((i,j))) for i,j in false_positive_contacts])
    non_local_false_positive_contacts = set(filter(lambda c: (c[1] - c[0]) >= 5, uniq_false_positive_contacts))

    ss_elements = get_ss_elements(ss)
    designed_contact_graph = contacts_to_ss_element_graph(true_non_local_contacts, ss_elements)
    tp_contact_graph = contacts_to_ss_element_graph(correctly_predicted_non_local_contacts, ss_elements)
    fp_contact_graph = contacts_to_ss_element_graph(non_local_false_positive_contacts, ss_elements)

    designed_ss_interactions = np.mean([1 if e in tp_contact_graph.edges else 0 for e in  designed_contact_graph.edges()])

    
    if args.plot_distance_map:

        fig = plt.figure(figsize=(30, 25), constrained_layout=True)
    
        widths = [1, 1]
        heights = [4, 3, 1]

    
        spec = fig.add_gridspec(nrows=3, ncols=2, width_ratios=widths,height_ratios=heights)
    
        ax1 = fig.add_subplot(spec[0, 0]) # row 0, col 0
        ax2 = fig.add_subplot(spec[0, 1]) # row 0, col 2
        ax3 = fig.add_subplot(spec[2, :]) # row 2, span all columns
        ax4 = fig.add_subplot(spec[1, 0]) # row 1, col 0
        ax5 = fig.add_subplot(spec[1, 1]) # row 1, col 1


        sns.heatmap(input_distance_matrix, ax=ax1).set_title('designed contacts', {'fontsize':22})
        sns.heatmap(predicted_distance_matrix, ax=ax2).set_title('predicted contacts', {'fontsize':22})

        for ax in [ax1, ax2]:        
            ax.axes.set_xticks(np.arange(0, len(input_distance_matrix), 10))
            ax.axes.set_yticks(np.arange(0, len(input_distance_matrix), 10))
            ax.tick_params(which='major', length=7)
            ax.tick_params(which='minor', length=4)
            ax.xaxis.set_major_locator(MultipleLocator(5))
            ax.yaxis.set_major_locator(MultipleLocator(5))
            ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
            ax.yaxis.set_major_formatter(FormatStrFormatter('%d'))
            ax.xaxis.set_minor_locator(MultipleLocator(1))
            ax.yaxis.set_minor_locator(MultipleLocator(1))
        

        ss_colors = {'H':'red', 'E':'yellow'}
        per_res_rmsd.plot.bar(ax=ax3, color=[ss_colors[e] if e in ss_colors else 'gray' for e in ss], linewidth=1, edgecolor='black')
        
        ax3.axes.set_xticks(np.arange(0, len(input_distance_matrix), 10))
        ax3.tick_params(which='major', length=7)
        ax3.tick_params(which='minor', length=4)
        ax3.axhline(per_res_rmsd.mean(), color='orange')


        ax3.xaxis.set_major_locator(MultipleLocator(5))
        ax3.xaxis.set_major_formatter(FormatStrFormatter('%d'))

        # For the minor ticks, use no labels; default NullFormatter.
        ax3.xaxis.set_minor_locator(MultipleLocator(1))

        plot_contacts(true_non_local_contacts, correctly_predicted_non_local_contacts,non_local_false_positive_contacts, ss_elements, ax4, ax5)
        ax4.set_title('designed contacts', {'fontsize':18})
        ax5.set_title('predicted contacts', {'fontsize':18})


        plot_fn = '{}_distance_map.png'.format(input_basename)
        fig.savefig(plot_fn)
        print('Distance map plots saved as {}'.format(plot_fn))

        
    
    if args.pymol:
        pymol_script_fn = '{}_contacts.pml'.format(input_basename)
        with open(pymol_script_fn, 'w') as pymol_script:
            pymol_script.write('load {}\n'.format(pdb))
            pymol_script.write('util.cbc\n')

            pymol_script.write(';'.join(['distance d_{p1}_{p2}, ///{chain}/{p1}/CA, ///{chain}/{p2}/CA'.format(chain=args.chain, p1=i+1,p2=j+1) for i,j in correctly_predicted_non_local_contacts]) + '\n')
            pymol_script.write(';'.join(['distance fp_d_{p1}_{p2}, ///{chain}/{p1}/CA, ///{chain}/{p2}/CA'.format(chain=args.chain, p1=i+1,p2=j+1) for i,j in non_local_false_positive_contacts]) + '\n')
            pymol_script.write('color red, fp_*\n')
            
            pymol_script.write('select  missing_contacts_40, resi ' + '+'.join([str(i+1) for i in chain(*np.argwhere((per_res_rmsd > np.percentile(per_res_rmsd, 60)).to_numpy()))]) + '\n')
            pymol_script.write('select  missing_contacts_20, resi ' + '+'.join([str(i+1) for i in chain(*np.argwhere((per_res_rmsd > np.percentile(per_res_rmsd, 80)).to_numpy()))]) + '\n')
            pymol_script.write('color orange, c. {} and missing_contacts_40\n'.format(chain))
            pymol_script.write('color red,  c. {} and missing_contacts_20\n'.format(chain))

            pymol_script.write('zoom d_* fp_*\n')
            
        print('pymol script with non-local contacts saved as {}'.format(pymol_script_fn))



    
    data = pd.DataFrame.from_dict([{
        'pdb' : pdb,
        'pred_contact_rmsd': rmsd,
        'contact_log_sum': logsum,
        'tp_contacts': len(correctly_predicted_unique_contacts) / len(true_uniq_contacts),
        'fp_contacts': len(uniq_false_positive_contacts) / len(true_uniq_contacts),
        'tp_non_local_contacts': len(correctly_predicted_non_local_contacts) / len(true_non_local_contacts),
        'fp_non_local_contacts': len(non_local_false_positive_contacts) / len(true_non_local_contacts),
        'tp_ss_interactions_fraction': designed_ss_interactions,
        'tp_ss_interactions_count': len(tp_contact_graph.edges()),
        'fp_ss_interactions_count': len(fp_contact_graph.edges()),



        'time': time() - init_time
    }])
    
    data.to_csv(args.output, sep='\t', index=False, mode='a', header=not path.exists(args.output))

print('contact metrics saved as {}'.format(args.output))

