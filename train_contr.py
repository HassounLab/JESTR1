# -*- coding: utf-8 -*-
"""
Created on Sat Aug 12 20:47:14 2023

@author: apurv
"""
import yaml
from utils import DatasetBuilder, MultiView_data, collate_contr_views, Print, load_models, contrastive_loss
from utils import fp_bce_loss, fp_cos_loss, fp_cos, print_hp, Spectra_data, collate_spectra_data, augmented_cand_loss
from utils import MyEarlyStopping, save_all_models, set_saved_best_model_names, augmented_cand_loss_spec
from dataset import load_contrastive_data, load_spectra_data
import sys
import os
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import pickle
from models import MolEnc, SpecEncMLP_BIN
import matplotlib.pyplot as plt
import time
import dgl

def train_contr(dataset_builder, molgraph_dict, params, output, device, data_path, no_data):

    if not no_data:
        cand_dict_train = dataset_builder.cand_dict_train
        Print("Loading Contrastive Data...", output)
        contr_dict, contr_key_list = load_contrastive_data(dataset_builder, params, device)
        Print("Done Loading Contrastive Data with {} objects".format(len(contr_dict)), output)
    
        contr_ds = MultiView_data(molgraph_dict, contr_dict, contr_key_list, 2)
        collate_fn = collate_contr_views(molgraph_dict, params, dataset_builder.cand_dict_train, 
                                         dataset_builder.cand_molgraph_dict, dataset_builder.mol_dict, device)
        
        # with open(os.path.join(data_path, 'molgraph_dict.pkl'), 'wb') as f:
        #     pickle.dump(molgraph_dict, f)
    
    dl_params = {'batch_size': params['batch_size_train_contr'],
                 'shuffle': True}
    models_list = [] # list of lists [model, idx, flag_frz, flag_clip_grad, flag_clip_weight]

    if not no_data:
        train_dl = DataLoader(contr_ds, collate_fn=collate_fn, **dl_params)
        
    sample_g = molgraph_dict[list(molgraph_dict.keys())[0]]
    x_size = sample_g.ndata['h'].shape[1]
    e_size = sample_g.edata['e'].shape[1]
    
    mol_enc_model = MolEnc(params, x_size)
    mol_enc_model = mol_enc_model.to(device)
    models_list.append([mol_enc_model, "mol_enc", False, False, False])
    if params['spec_enc'] == 'MLP_BIN':
        bin_size = int(params['max_mz'] / params['resolution'])
        spec_enc_model = SpecEncMLP_BIN(params, bin_size)
    elif params['spec_enc'] == 'MLP_SIN':
        spec_enc_model = SpecEncMLP_SIN(params)
    elif params['spec_enc'] == 'TFM':
        spec_enc_model = SpecEncTFM(params)
    spec_enc_model = spec_enc_model.to(device)
    models_list.append([spec_enc_model, "spec_enc", False, False, False])
    load_models(params, models_list, device, output)

    if not params['contr_trg'] or no_data:
        return mol_enc_model, spec_enc_model, models_list
    
    parameters, pr_params = [], []
    for model, idx, frz, _, _ in models_list:
        if frz: continue
        else:             parameters += [p for p in model.parameters() if p.requires_grad]

    model_time = str(int(round(time.time() * 1000)))

    optimizer = torch.optim.Adam([{'params':parameters,    'lr':params['contr_lr']}])
    train_contr_losses, train_fp_losses, train_losses = [],[],[]
    wt_contr = params['wt_contr']
    wt_fp = params['wt_fp']
    model_time = str(int(round(time.time() * 1000)))
    stopper = MyEarlyStopping(models_list, model_time, params, output,
        mode='lower', patience = params['early_stopping_patience_contr'], 
        filename = "early_stopping/pred_"+ model_time)

    aug_cand_opt = params['aug_cands']
    in_aug_cands = False
    cnt_aug_runs = 0
    if params['contr_trg']:
        Print("Starting Contrastive Training...", output)
        for epoch in range(params['num_epoch_contr']):
            if aug_cand_opt and not in_aug_cands:
                if epoch < int(params['num_epoch_contr'] * 0.97):
                    params['aug_cands'] = False
                else:
                    params['aug_cands'] = True
                    in_aug_cands = True
                    stopper.best_score = None
                    stopper.counter = 0
                            
            train_contr_loss, train_fp_loss, train_loss, cos_sim_total = 0.0,0.0,0.0,0.0
            train_cand_aug_loss, train_contr_only_loss = 0.0, 0.0
            for model, idx, frz, _, _ in models_list: model.train()
    
            for batch_id, (batch_g, mz_b, int_b, pad, fp_b, lengths, bat_cand_g, sim_l) in enumerate(tqdm(train_dl, total=int(len(train_dl)), leave=False)):
                batch_g = batch_g.to(torch.device(device))
                mz_b = mz_b.to(torch.device(device), dtype=torch.float32)
                int_b = int_b.to(torch.device(device), dtype=torch.float32)
                pad = pad.to(torch.device(device))
                fp_b = fp_b.to(torch.device(device))
                mol_enc = mol_enc_model(batch_g, batch_g.ndata['h'])
                spec_enc = spec_enc_model(mz_b, int_b, pad, lengths)
                
                #fp_pred = fp_model(spec_enc)
                
                loss = contrastive_loss(mol_enc, spec_enc, params['contr_temp'])
                #loss_fp = fp_predict_loss(fp_pred, fp_b)
                if params['aug_cands']:
                    cnt_aug_runs += 1
                    sim_l = [sim.to(torch.device(device)) for sim in sim_l]
                    # cand_mol_enc_l = []
                    # for idx in range(mol_enc.shape[0]): #iterate as many times as mol_enc because of last batch
                    #     cand_g = bat_cand_g[idx]
                    #     cand_mol_enc = mol_enc_model(cand_g, cand_g.ndata['h'])
                    #     cand_mol_enc_l.append(cand_mol_enc)
                    bat_cand_g = bat_cand_g.to(torch.device(device))   
                    cand_mol_enc = mol_enc_model(bat_cand_g, bat_cand_g.ndata['h'])
                    cand_mol_enc = cand_mol_enc.reshape([mol_enc.shape[0], params['batch_size_train_contr_cand'], params['gnn_hidden_dim']])
                    loss_cand_aug = augmented_cand_loss_spec(spec_enc, cand_mol_enc, sim_l)
                    loss_contr = loss
                    loss = (1-params['aug_cands_wt']) * loss + params['aug_cands_wt'] * loss_cand_aug
                    
                #loss = wt_contr * loss_contr + wt_fp * loss_fp
                #cos_sim = fp_cos(fp_pred, fp_b)
                #cos_sim = cos_sim.detach().item()
                #cos_sim_total += cos_sim
                loss.backward()
                
                optimizer.step()
                optimizer.zero_grad()
    
                train_contr_loss += loss.detach().item()
                if params['aug_cands']:
                    train_cand_aug_loss += loss_cand_aug.detach().item()
                    train_contr_only_loss += loss_contr.detach().item()
                #train_fp_loss += loss_fp.detach().item()
                #train_loss += loss.detach().item()
                
            train_contr_loss /= (batch_id + 1)
            if params['aug_cands']:
                train_cand_aug_loss /= (batch_id + 1)
                train_contr_only_loss /= (batch_id + 1)
            del collate_fn.cand_molgraph_dict
            collate_fn.cand_molgraph_dict = {} #reset cand_dict to save memory
            #train_fp_loss /= (batch_id + 1)
            #train_loss /= (batch_id + 1)
            #cos_sim_total /= (batch_id + 1)
            # inline_log = 'Epoch {} / {}, train_contr_loss: {:.4f}, train_fp_loss: {:.4f}, train_loss: {:.4f}, cos_sim: {:.4f}'.format(
            #     epoch + 1, params['num_epoch_contr'], train_contr_loss, train_fp_loss, train_loss, cos_sim_total)
            if params['aug_cands']:
                inline_log = 'Epoch {} / {}, train_contr_loss: {:.4f}, train_contr_only_loss: {:.4f}, train_cand_aug_loss: {:.4f}'.format(
                    epoch + 1, params['num_epoch_contr'], train_contr_loss, train_contr_only_loss, train_cand_aug_loss)
            else:
                inline_log = 'Epoch {} / {}, train_contr_loss: {:.4f}'.format(
                    epoch + 1, params['num_epoch_contr'], train_contr_loss)
            Print(inline_log, output)
            train_contr_losses.append(train_contr_loss)
            # train_fp_losses.append(train_fp_loss)
            # train_losses.append(train_loss)
            early_stop = stopper.step(train_contr_loss)
            if early_stop:
                if aug_cand_opt:
                    params['aug_cands'] = True
                    stopper.best_score = None
                    in_aug_cands = True
                    stopper.counter = 0
                else:
                    saved_model_name = "Saved early stopping model in " + stopper.filename
                    Print(saved_model_name, output)
                    break
        
            if cnt_aug_runs == 15:
                saved_model_name = "Saved early stopping model in " + stopper.filename
                Print(saved_model_name, output)
                break

        save_all_models(params, models_list, model_time + "_last", output)
        set_saved_best_model_names(params, models_list, model_time)
        
        plt.figure()
        plt.plot(train_contr_losses)
        plt.ylim([0, max(train_contr_losses)])
        plt.legend(['train'], loc='upper left')
        plt.title('Loss')
        plt.savefig("logs/train_contr_losses_" + str(model_time) + ".png")
        loss_graph_file = "created loss graph in logs/train_contr_losses_" + str(model_time) + ".png"
        Print(loss_graph_file, output)

    return mol_enc_model, spec_enc_model, models_list
