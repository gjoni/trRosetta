#!env python
import numpy as np
import pandas as pd
from Bio import PDB
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)

import seaborn as sns
from itertools import chain
from scipy.spatial import distance_matrix

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from os import path, system, environ
from subprocess import check_call, DEVNULL
from tempfile import NamedTemporaryFile

dir_path = path.dirname(path.realpath(__file__))

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

for pdb in args.pdbs:
    input_basename, _ = path.splitext(path.basename(pdb))
    # parse input pdb and prepare for prediction
    input_structure = PDB.PDBParser().get_structure('input_structure', pdb)
    if args.chain:
        input_structure = next(filter(lambda c: c.id ==args.chain, input_structure.get_chains()))
    
    ca_atoms = list(filter(lambda a: a.name == 'CA', input_structure.get_atoms()))
    input_distance_matrix = pd.DataFrame(np.array([[np.linalg.norm(at2.coord - at1.coord) for at2 in ca_atoms] for at1 in ca_atoms]))
    input_distance_matrix[input_distance_matrix >20] = 0



    ppb = PDB.Polypeptide.PPBuilder()
    input_sequence = ppb.build_peptides(input_structure).pop().get_sequence()

    # dump input sequence into a temporary file and run TrRosetta
    with NamedTemporaryFile('+w') as fasta:
        fasta.write('> input\n{}'.format(input_sequence))
        fasta.flush()
        with NamedTemporaryFile('r', suffix='.npz') as output_npz:
            print('Predicting contacts for {}'.format(pdb))
            check_call(['python', '{}/network/predict.py'.format(args.tr_rosetta_path), '-m', '{}/model2019_07'.format(args.tr_rosetta_path), fasta.name, output_npz.name], stdout=DEVNULL, stderr=DEVNULL)
            print('Done predicting contacts for {}'.format(pdb))

            predictions = np.load(output_npz.name)
            distances = predictions['dist']
            bins = np.array([0, *np.arange(2,20,.5)])
            predicted_distance_matrix = pd.DataFrame((bins[np.argmax(distances, axis=2)]))


    if args.plot_distance_map:

        fig = plt.figure(figsize=(30, 20), constrained_layout=True)
    
        widths = [1, 1]
        heights = [4, 2]

    
        spec = fig.add_gridspec(nrows=2, ncols=2, width_ratios=widths,height_ratios=heights)
    
        ax1 = fig.add_subplot(spec[0, 0]) # row 0, col 0
        ax2 = fig.add_subplot(spec[0, 1]) # row 0, col 0
        ax3 = fig.add_subplot(spec[1, :]) # row 1, span all columns

        sns.heatmap(input_distance_matrix, ax=ax1).set_title('designed contacts', {'fontsize':22})
        sns.heatmap(predicted_distance_matrix, ax=ax2).set_title('predicted contacts', {'fontsize':22})
        
        m = ((input_distance_matrix[(input_distance_matrix != 0.0) & (input_distance_matrix < 6)] - predicted_distance_matrix[(input_distance_matrix != 0.0) & (input_distance_matrix < 6)])**2)
        (m.sum() / pd.notna(m).sum()).plot.bar(ax=ax3)
        
        ax3.axes.set_xticks(np.arange(0, len(input_distance_matrix), 10))
        ax3.tick_params(which='major', length=7)
        ax3.tick_params(which='minor', length=4)


        ax3.xaxis.set_major_locator(MultipleLocator(5))
        ax3.xaxis.set_major_formatter(FormatStrFormatter('%d'))

        # For the minor ticks, use no labels; default NullFormatter.
        ax3.xaxis.set_minor_locator(MultipleLocator(1))

        plot_fn = '{}_distance_map.png'.format(input_basename)
        fig.savefig(plot_fn)
        print('Distance map plots saved as {}'.format(plot_fn))

        

    # calculate relevant metrics
    rmsd = np.sqrt(np.concatenate((input_distance_matrix - predicted_distance_matrix)**2).sum() / 160**2)
    
    true_contacts = np.argwhere((input_distance_matrix[(input_distance_matrix < 6)] > 0.0).to_numpy())
    true_uniq_contacts = set([tuple(sorted((i,j))) for i,j in true_contacts])
    true_non_local_contacts = set(filter(lambda c: (c[1] - c[0]) >= 5, true_uniq_contacts))
    
    correctly_predicted_contacts = np.argwhere((predicted_distance_matrix[(input_distance_matrix < 6) & (predicted_distance_matrix < 6)] > 0.0).to_numpy())
    correctly_predicted_unique_contacts = set([tuple(sorted((i,j))) for i,j in correctly_predicted_contacts])
    correctly_predicted_non_local_contacts = set(filter(lambda c: (c[1] - c[0]) >= 5, correctly_predicted_unique_contacts))
    
    false_positive_contacts = np.argwhere((predicted_distance_matrix[((input_distance_matrix > 12)| (input_distance_matrix == 0)) & ((predicted_distance_matrix < 6) & (predicted_distance_matrix > 0))] > 0.0).to_numpy())
    uniq_false_positive_contacts = set([tuple(sorted((i,j))) for i,j in false_positive_contacts])
    non_local_false_positive_contacts = set(filter(lambda c: (c[1] - c[0]) >= 5, uniq_false_positive_contacts))
    
    if args.pymol:
        pymol_script_fn = '{}_contacts.pml'.format(input_basename)
        with open(pymol_script_fn, 'w') as pymol_script:
            pymol_script.write('load {}\n'.format(pdb))
            pymol_script.write(';'.join(['distance d_{p1}_{p2}, ///{chain}/{p1}/CA, ///{chain}/{p2}/CA'.format(chain=args.chain, p1=i+1,p2=j+1) for i,j in correctly_predicted_non_local_contacts]) + '\n')
            pymol_script.write(';'.join(['distance fp_d_{p1}_{p2}, ///{chain}/{p1}/CA, ///{chain}/{p2}/CA'.format(chain=args.chain, p1=i+1,p2=j+1) for i,j in non_local_false_positive_contacts]) + '\n')
            pymol_script.write('color red, fp_*\n')
            pymol_script.write('zoom d_* fp_*\n')
            pymol_script.write('util.cbc')
            
        print('pymol script with non-local contacts saved as {}'.format(pymol_script_fn))



    
    data = pd.DataFrame.from_dict([{
        'pdb' : pdb,
        'tp_contacts': len(correctly_predicted_unique_contacts) / len(true_uniq_contacts),
        'fp_contacts': len(uniq_false_positive_contacts) / len(true_uniq_contacts),
        'tp_non_local_contacts': len(correctly_predicted_non_local_contacts) / len(true_non_local_contacts),
        'fp_non_local_contacts': len(non_local_false_positive_contacts) / len(true_non_local_contacts),

    }])
    
    data.to_csv(args.output, sep='\t', index=False, mode='a', header=not path.exists(args.output))

print('contact metrics saved as {}'.format(args.output))

