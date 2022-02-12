import torch
import pandas as pd
from src.model import Model


if __name__ == '__main__':
    CENTROIDS_DIR = './project_test_data/centroids.csv'
    PDBS_DIR = './project_test_data/pdbs'
    LIGAND_DIR = './project_test_data/ligand.csv'
    GT_PAIR_DIR = './project_test_data/pair.csv'

    #read centroids.csv
    centroids = {}
    df = pd.read_csv(CENTROIDS_DIR)
    for i in range(len(df)):
        centroids[str(df.PID[i])] = (float(df.x[i]), float(df.y[i]), float(df.z[i]))

    #read ligand.csv
    ligands = {}
    df = pd.read_csv(LIGAND_DIR)
    for i in range(len(df)):
        ligands[str(df.LID[i])] = (str(df.Smiles[i]))

    #read groundtruth pair.csv for grading
    gt_pairs = {}
    df = pd.read_csv(GT_PAIR_DIR)
    for i in range(len(df)):
        gt_pairs[str(df.PID[i])] = (str(df.LID[i]))

    
    BS = 100                        #Batch size for inference
    TOPK = 10                       #Set top-10 accuracy
    DEVICE = 'cpu'                  #You do not necessarily need a GPU. Of course, you are free to use a GPU if you have one.

    model = Model(device=DEVICE)    #When grading, we will import and call your own model
    model.to(DEVICE)

    #inference
    prediction_correctness = []
    for PID in centroids:
        binding_scores = torch.empty(0, 1)
        LIDs =[LID for LID in ligands]
        #check the binding score of each ligand to one specific protein
        for i in range(0, len(LIDs)-BS+1, BS):
            batch_pred = model.inference(PID, centroids[PID], LIDs[i: i+BS], [ligands[LID] for LID in LIDs[i: i+BS]])
            binding_scores = torch.cat([binding_scores, batch_pred], dim=0)
        if i < len(LIDs)-BS:
            batch_pred = model.inference(PID, centroids[PID], LIDs[i+BS: ], [ligands[LID] for LID in LIDs[i+BS: ]])
            binding_scores = torch.cat([binding_scores, batch_pred], dim=0)

        #transform torch.tensor to list
        binding_scores = binding_scores.squeeze(-1).cpu().detach().numpy().tolist()

        #get top-k scores and corresponding LIDs
        topk_pred = sorted(zip(binding_scores, LIDs), reverse=True)[:TOPK]
        topk_scores, topk_LIDs = zip(*topk_pred)
        #print(topk_LIDs)
        
        #compare with groundtruth
        if str(gt_pairs[PID]) in topk_LIDs:
            prediction_correctness.append(1)
        else:
            prediction_correctness.append(0)

    accuracy = sum(prediction_correctness) / len(prediction_correctness)

    print(f"Inference Prediction Score: {'{:.5f}'.format(accuracy)}.")