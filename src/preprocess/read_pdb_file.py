import sys
import os


def read_pdb(filename: str):
    """Read a protein file to get four atom information lists.
    
    You can copy this function to your project code.
    """
    with open(filename, 'r') as file:
        strline_L = file.readlines()
    strline_L = [strline.strip() for strline in strline_L]

    X_list = [float(strline.split()[-3]) for strline in strline_L]
    Y_list = [float(strline.split()[-2]) for strline in strline_L]
    Z_list = [float(strline.split()[-1]) for strline in strline_L]
    atomtype_list = [strline.split()[-7][0] for strline in strline_L]

    return X_list, Y_list, Z_list, atomtype_list


def read_ligand_pdb(filename: str):
    # read 3d structure
    with open(filename, 'r') as file:
        strline_L = file.readlines()
    strline_L = [strline.strip() for strline in strline_L]

    atom_types, xs, ys, zs = [], [], [], []
    connections = {}
    zero_xyz = (0.000, 0.000, 0.000)
    illegal = True

    for strline in strline_L:
        tokens = strline.split()
        if tokens[0] == "HETATM" or tokens[0] == "ATOM":
            atom_types.append(tokens[2])
            xs.append(float(tokens[5]))
            ys.append(float(tokens[6]))
            zs.append(float(tokens[7]))
            cur_xyz = (xs[-1], ys[-1], zs[-1])
            if illegal and cur_xyz != zero_xyz:  # found illgal 3d structure (duplicated coords)
                illegal = False
        elif tokens[0] == "CONECT":
            connections[int(tokens[1])] = [int(t) for t in tokens[2:]]

    file.close()
    return illegal, atom_types, connections, xs, ys, zs


def perror_and_exit(prompt: str, errno: int = -1):
    print(prompt, file=sys.stderr)
    sys.exit(errno)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        perror_and_exit("Usage: {} PDB_FILE_PATH".format(sys.argv[0]), 1)

    pdb_path = sys.argv[1]
    if not os.path.exists(pdb_path):
        perror_and_exit("Error: file does not exist", 2)

    if not os.path.isfile(pdb_path):
        perror_and_exit("Error: {} is not a file".format(pdb_path), 3)

    print("Reading protein file {}...".format(os.path.abspath(pdb_path)))

    X_list, Y_list, Z_list, atomtype_list = read_pdb(pdb_path)

    print("X_list: {}".format(X_list))
    print("Y_list: {}".format(Y_list))
    print("Z_list: {}".format(Z_list))
    print("atomtype_list: {}".format(atomtype_list))

    sys.exit(0)
