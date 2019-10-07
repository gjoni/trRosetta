# *trRosetta*
This package is a part of ***trRosetta*** protein structure prediction protocol developed in: [Improved protein structure prediction using predicted inter-residue orientations](). It includes tools to predict protein inter-residue geometries from a multiple sequence alignment or a single sequence.


Contact: Ivan Anishchenko, aivan@uw.edu


## Requirements
```tensorflow``` (tested on versions ```1.13``` and ```1.14```)

## Download

```
# download package
git clone https://github.com/gjoni/trRosetta
cd trRosetta

# download pre-trained network
wget https://files.ipd.uw.edu/pub/trRosetta/model2019_07.tar.bz2
tar xf model2019_07.tar.bz2
```

## Usage
```
python ./network/predict.py -m ./model2019_07 example/T1001.a3m example/T1001.npz
```

## Links

* [structure modeling scripts](http://yanglab.nankai.edu.cn/trRosetta/download/) (require [PyRosetta](http://www.pyrosetta.org/))

* [***trRosetta*** server](http://yanglab.nankai.edu.cn/trRosetta/)


## References
J Yang, I Anishchenko, H Park, Z Peng, S Ovchinnikov, D Baker. Improved protein structure prediction using predicted inter-residue orientations. (2019) Submitted 
