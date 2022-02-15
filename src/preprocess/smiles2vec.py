# Convert smiles to embeddings using mol2vec, a pretrained molecule representation model.
# Further reference: https://github.com/samoturk/mol2vec
import numpy as np
import torch
from gensim.models import word2vec
from rdkit import Chem
from mol2vec.features import mol2alt_sentence
from sklearn.decomposition import PCA
from src.global_vars import MOL2VEC_MODEL_DIR, EMBED_PADDING_DIM


# takes in a list of smiles and return their embeddings given an output vector dimension
class Smiles2Vec:
    def __init__(self, dim=300):
        self.model = word2vec.Word2Vec.load(MOL2VEC_MODEL_DIR)
        self.pca_model = PCA(n_components=dim)
        self.illegal = set()

    def get_embeddings(self, smiles):
        sentences = []
        i = 1
        for s in smiles:
            try:
                m = Chem.MolFromSmiles(s)
                if m is not None:
                    sentences.append(mol2alt_sentence(m, 1))
                else:
                    self.illegal.add(i)
            except:
                self.illegal.add(i)
            i += 1
        # retrieve unique identifiers
        identifiers = [set(sublist) for sublist in sentences]
        # inference
        raw_vec = []
        for sublist in identifiers:
            r = []
            for x in sublist:
                try:
                    r.append(self.model.wv.word_vec(x))
                except:
                    continue
            raw_vec.append(r)
        # pad with zeros for constant size
        padded_arr = []
        for i in range(len(raw_vec)):
            l = EMBED_PADDING_DIM - len(raw_vec[i])
            if l >= 0:
                padded_arr.append(np.pad(raw_vec[i],
                                         [(0, l), (0, 0)],
                                         mode='constant', constant_values=0).tolist())
            else:  # normally we don't need to truncate, put here just in case
                padded_arr.append(np.array(raw_vec[i][:EMBED_PADDING_DIM]).tolist())
        # project using pca
        reduced_arr = torch.Tensor(np.array(
            [self.pca_model.fit_transform(sublist) for sublist in padded_arr]))
        return reduced_arr
