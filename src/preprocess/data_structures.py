# shared data structures
from typing import NamedTuple  # requires python version > 3.6


class Protein(NamedTuple):
    atom_types: list
    xs: list
    ys: list
    zs: list


class Ligand(NamedTuple):
    smiles: str
    atom_types: list
    connections: dict
    xs: list
    ys: list
    zs: list
