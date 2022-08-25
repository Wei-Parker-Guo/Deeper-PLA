# Predicting Protein-Ligand Binding Affinity
Molecular recognition between proteins and ligands plays an important role in many biological processes, such as membrane receptor signaling and enzyme catalysis. Predicting the structures of protein-ligand complexes and finding ligands by virtual screening of small molecule databases are two long-standing goals in molecular biophysics and medicinal chemistry. Knowledge-based statistical potentials have been developed for modeling protein-ligand interactions. They are based on distributions of intermolecular features in large databases of protein-ligand complexes.

## Implementor

Name: Wei Guo

Contact: wei.parker.guo.sg@gmail.com

## Usage

### Environment Setup
Additional to pytorch, the project uses the following packages:

**OpenBabel 3.1.0**
```commandline
conda install -c source-forge openbabel
```
**RdKit**
```commandline
conda install rdkit
```

**TorchSummary**
```commandline
pip install torch-summary
```

**Ipywidgets**
```commandline
conda install -c conda-forge ipywidgets
jupyter nbextension enable --py widgetsnbextension
```

### Compiling
There is no need to specify source directory since all the modules are referenced with namespace packages.

### Entry Points
The training and testing are done within `notebooks`. You can also try the shell scripts in `src` for preprocessing.

## References
Codes from other sources are referenced with in-line comments. Papers referenced are listed here:

[1] Y. Li, M. A. Rezaei, C. Li, X. Li, and D. Wu, “DeepAtom: A Framework for Protein-Ligand Binding Affinity Prediction,” arXiv:1912.00318 [cs, q-bio], Nov. 2019, Accessed: Feb. 14, 2022. [Online]. Available: http://arxiv.org/abs/1912.00318

[2] H. C. Jubb, A. P. Higueruelo, B. Ochoa-Montaño, W. R. Pitt, D. Ascher, and T. Blundell, “Arpeggio: A Web Server for Calculating and Visualising Interatomic Interactions in Protein Structures,” Feb. 2017, doi: 10.17863/CAM.8104.

[3] S. Jaeger, S. Fulle, and S. Turk, “Mol2vec: Unsupervised Machine Learning Approach with Chemical Intuition,” J. Chem. Inf. Model., vol. 58, no. 1, pp. 27–35, Jan. 2018, doi: 10.1021/acs.jcim.7b00616.

[4] K. He, X. Zhang, S. Ren, and J. Sun, “Identity Mappings in Deep Residual Networks,” arXiv:1603.05027 [cs], Jul. 2016, Accessed: Feb. 17, 2022. [Online]. Available: http://arxiv.org/abs/1603.05027

[5] N. Ma, X. Zhang, H.-T. Zheng, and J. Sun, “ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design,” arXiv:1807.11164 [cs], Jul. 2018, Accessed: Feb. 17, 2022. [Online]. Available: http://arxiv.org/abs/1807.11164
