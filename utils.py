import numpy as np
from numpy.random import RandomState

rng = RandomState(0)

# Map 3 letter amino acid to integer
def aa_to_int(aa):
    if aa == 'UNK':
        return rng.randint(0, 20)
    return {
        'ALA': 0,
        'ARG': 1,
        'ASN': 2,
        'ASP': 3,
        'CYS': 4,
        'GLN': 5,
        'GLU': 6,
        'GLY': 7,
        'HIS': 8,
        'ILE': 9,
        'LEU': 10,
        'LYS': 11,
        'MET': 12,
        'PHE': 13,
        'PRO': 14,
        'SER': 15,
        'THR': 16,
        'TRP': 17,
        'TYR': 18,
        'VAL': 19
        }[aa]

aa_one_to_three = {
 'C': 'CYS',
 'D': 'ASP',
 'S': 'SER',
 'Q': 'GLN',
 'K': 'LYS',
 'I': 'ILE',
 'P': 'PRO',
 'T': 'THR',
 'F': 'PHE',
 'N': 'ASN',
 'G': 'GLY',
 'H': 'HIS',
 'L': 'LEU',
 'R': 'ARG',
 'W': 'TRP',
 'A': 'ALA',
 'V': 'VAL',
 'E': 'GLU',
 'Y': 'TYR',
 'M': 'MET'
}

with open("BLOSUM62.txt", "r") as f:
    lines = f.readlines()

blosum62 = np.zeros((20, 20))

header = lines[0].split()
for line in lines[1:]:
    line = line.split()
    if line[0] == '*':
        continue
    for i in range(1, len(line)):
        blosum62[aa_to_int(aa_one_to_three[line[0]]), aa_to_int(aa_one_to_three[header[i-1]])] = float(line[i])


