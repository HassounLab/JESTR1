# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 11:34:25 2023

@author: apurv
"""

import pickle
from rdkit.Chem import AllChem
from rdkit import Chem
import yaml
from utils import DatasetBuilder, MultiView_data, collate_contr_views, Print, load_models, contrastive_loss
from utils import fp_bce_loss, fp_cos_loss, fp_cos, print_hp, Spectra_data, collate_spectra_data
from utils import MyEarlyStopping, save_all_models
from dataset import load_contrastive_data, load_spectra_wneg_data, load_cand_test_data
import sys
import os
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import pickle
from models import MolEnc, SpecEncMLP_BIN, INTER_MLP2
import matplotlib.pyplot as plt
import time
from train_contr import train_contr
import torch.nn.functional as F
from sklearn.metrics import auc, roc_auc_score, roc_curve, average_precision_score
from collections import defaultdict
import numpy as np

cand_size = 100

if __name__ == "__main__":

    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    with open('params.yaml') as f:
        params = yaml.load(f, Loader=yaml.FullLoader)

    if len(sys.argv) > 1:
        for i in range(len(sys.argv) - 1):
            key, value_raw = sys.argv[i+1].split("=")
            print(str(key) + ": " + value_raw)
            try:
                params[key] = int(value_raw)
            except ValueError:
                try:
                    params[key] = float(value_raw)
                except ValueError:
                    params[key] = value_raw


    dir_path = ""
    ms_intensity_threshold = 0.0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logfile = params['logfile']
    output = open(logfile, "a")
    print_hp(params, output)

    dataset_builder = DatasetBuilder(params['exp'])
    dataset_builder.init(dir_path, params['fp_path'], ms_intensity_threshold)
    
    data_path = dir_path + dataset_builder.data_dir
    
    if os.path.exists(data_path+"molgraph_dict.pkl"):
        with open(os.path.join(data_path, "molgraph_dict.pkl"), 'rb') as f:
            molgraph_dict = pickle.load(f)
    else:
        molgraph_dict = {}
    
    dataset_builder.molgraph_dict = molgraph_dict       

    mol_enc_model, spec_enc_model, models_list = train_contr(dataset_builder, molgraph_dict, params, output, 
                                                             device, data_path, True)
    
    with open(data_path + 'cand_dict_large.pkl', 'rb') as f:
        cand_dict = pickle.load(f)
    
    cand = dataset_builder.split_dict['test']

    with open(data_path + '/inchi_to_id_dict.pkl', 'rb') as f:
        inchi_to_id = pickle.load(f)    
            
    inter_model = INTER_MLP2(params)
    inter_model = inter_model.to(device)
    models_list.append([inter_model, "inter", False, False, False])
    load_models(params, models_list, device, output)

    test_loss_func = torch.nn.CrossEntropyLoss()

    model_time = str(int(round(time.time() * 1000)))
    in_to_id_dict = dataset_builder.in_to_id_dict
    rank_dataf = "results/ranks_" + str(model_time) + ".pkl" 
    rank_totalf = "results/ranks_total_" + str(model_time) + ".pkl" 
    
    Print("Starting Ranking...", output)

    #q_list = list(cand_dict.keys())
    #q_list = cand
    #q_list = list(inchi_to_id_ESP.keys())
    #q_list = new_split_dict['test']
    q_list = dataset_builder.split_dict['test']
    
    for model, idx, frz, _, _ in models_list: model.eval()
    
    rank = []
    rank_dist = []
    rank_list = []
    dist_dict = defaultdict(lambda: []) #mol-mol dist, target-spec dict, cand-spec dist
    totpred = torch.Tensor()
    totinchi = []
    totspec = []
    totdist = []
    dist_target = []
    dist_cand = []
    target_idx = []
    
    for i, ik in enumerate(tqdm(q_list)):

        cand_list = cand_dict.get(ik, None)
        if cand_list == None: # some NIST20 Inchi keys are not in NIST23
            continue
        spec_list = inchi_to_id.get(ik, None)
        if spec_list == None: # some Inchi keys are not in NIST20!! why??
            continue
        if i > 0 and i % 100 == 0:
            Print("Average rank %.3f +- %.3f" % (np.mean(rank), np.std(rank)), output)
        for spec in spec_list:
            cand_test = load_cand_test_data(dataset_builder, params, ik, cand_list, spec, device)
            if len(cand_test) < 2:
                continue
            
            cand_test_ds = Spectra_data(cand_test)
            collate_fn = collate_spectra_data(molgraph_dict, params)

            dl_params = {'batch_size': params['batch_size_val_final'],
                         'shuffle': False}
            cand_test_dl = DataLoader(cand_test_ds, collate_fn=collate_fn, **dl_params)
            predlist = torch.Tensor()
            mol_enc_list = torch.Tensor()
            spec_enc_list = torch.Tensor()
            inchi_list = []
            totdist_local = []

            for batch_id, (batch_g, mz_b, int_b, pad, fp_b, y, lengths, inchi) in enumerate(cand_test_dl):
                batch_g = batch_g.to(torch.device(device))
                mz_b = mz_b.to(torch.device(device))
                int_b = int_b.to(torch.device(device))
                pad = pad.to(torch.device(device))
                fp_b = fp_b.to(torch.device(device))
                y = y.to(torch.device(device))
                with torch.no_grad():
        
                    mol_enc = mol_enc_model(batch_g, batch_g.ndata['h'])
                    spec_enc = spec_enc_model(mz_b, int_b, pad, lengths)
                    
                    prediction = inter_model(mol_enc, spec_enc)
                    prediction = prediction.squeeze(1)
                    
                #loss = test_loss_func(prediction, y)
                # logits = F.softmax(prediction, 1)
                # logits = logits[:,1]
                # logits = logits.cpu()
                # y = y.cpu()
                #prediction = prediction[:,1]
                prediction = prediction.cpu()
                predlist = torch.cat([predlist, prediction])
                spec_enc = spec_enc.cpu()
                spec_enc_list = torch.cat([spec_enc_list, spec_enc])
                mol_enc = mol_enc.cpu()
                mol_enc_list = torch.cat([mol_enc_list, mol_enc])
                inchi_list += inchi
                dist = torch.nn.CosineSimilarity()
                dist = dist(mol_enc, spec_enc)
                dist = dist.tolist()
                totdist_local += dist
                #labellist = torch.cat([labellist, y])
                #test_loss += loss.detach().item()

            totpred = torch.cat([totpred, predlist])
            totinchi += inchi_list
            totspec += [spec]*len(predlist)
            totdist += totdist_local
            dist_target.append(totdist_local[0])
            dist_cand.append(np.mean(totdist_local[1:]))
            target_idx += [1] + [0]*(len(totdist_local)-1)
            test_rank = (predlist[0] > predlist[1:]).sum()  
            test_rank = len(predlist) - test_rank
            totdist_local = np.array(totdist_local)
            test_rank_dist = (totdist_local[0] > totdist_local[1:]).sum()  
            test_rank_dist = len(totdist_local) - test_rank_dist            
            combined_l = zip(predlist, inchi_list, mol_enc_list, spec_enc_list)
            combined_l = sorted(combined_l, reverse=True)
            inchi_list = [i[1] for i in combined_l]
            pred_list = [i[0] for i in combined_l]
            # if test_rank == 1 or test_rank == 2:
            #     dist = torch.nn.CosineSimilarity()
            #     mol1 = combined_l[0][2]
            #     mol1 = mol1.unsqueeze(0)
            #     mol2 = combined_l[1][2]
            #     mol2 = mol2.unsqueeze(0)
            #     spec = combined_l[0][3]
            #     spec = spec.unsqueeze(0)
            #     dist1 = dist(mol1, mol2)
            #     dist2 = dist(mol1, spec)
            #     dist3 = dist(mol2, spec)
            #     dist_dict[test_rank.item()].append([dist1, dist2, dist3, inchi_list[0], inchi_list[1]])


            rank.append(test_rank)
            rank_dist.append(test_rank_dist)
            rank_list.append((ik, spec, len(cand_test), test_rank, inchi_list, pred_list))
            
    rank = np.array(rank)
    rank_dist = np.array(rank_dist)
    
    Print("Average rank %.3f +- %.3f" % (rank.mean(), rank.std()), output)
    for i in range(1, 21):
        Print("Rank at %d %.3f" % (i, (rank <= i).sum() / float(rank.shape[0])), output)
    Print("Average rank dist %.3f +- %.3f" % (rank_dist.mean(), rank_dist.std()), output)
    for i in range(1, 21):
        Print("Rank (dist) at %d %.3f" % (i, (rank_dist <= i).sum() / float(rank_dist.shape[0])), output)
    
    Print("Saved rank file in {}".format(rank_dataf), output)
    with open('results/distfile.pkl', 'wb') as f:
        pickle.dump((totpred, totdist, dist_target, dist_cand, target_idx), f)
    with open(rank_dataf, 'wb') as f:
        pickle.dump(rank, f)
        
