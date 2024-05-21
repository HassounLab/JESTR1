# -*- coding: utf-8 -*-
"""
Created on Sat Aug 12 20:47:14 2023

@author: apurv
"""
import yaml
from utils import DatasetBuilder, MultiView_data, collate_contr_views, Print, load_models, contrastive_loss
from utils import fp_bce_loss, fp_cos_loss, fp_cos, print_hp, Spectra_data, collate_spectra_data
from utils import MyEarlyStopping, save_all_models, set_seeds
from dataset import load_contrastive_data, load_spectra_wneg_data
import sys
import os
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import pickle
from models import MolEnc, SpecEncMLP_BIN, SpecEncMLP_SIN, SpecEncTFM, INTER_MLP, INTER_MLP2
import matplotlib.pyplot as plt
import time
from train_contr import train_contr
import torch.nn.functional as F
from sklearn.metrics import auc, roc_auc_score, roc_curve, average_precision_score

if __name__ == "__main__":

    #set_seeds(2023)
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

    if os.path.exists(data_path+"cand_dict_train_updated.pkl"):
        with open(os.path.join(data_path, "cand_dict_train_updated.pkl"), 'rb') as f:
            cand_dict_train = pickle.load(f)
    else:
        cand_dict_train = {}
    
    dataset_builder.cand_dict_train = cand_dict_train       

    mol_enc_model, spec_enc_model, models_list = train_contr(dataset_builder, molgraph_dict, params, output, 
                                                             device, data_path, False)
    Print("Loading Final Task Train Data...", output)
    final_train = load_spectra_wneg_data(dataset_builder, params, 'train', device)
    Print("Done Loading Final Task Train Data with {} objects".format(len(final_train)), output)

    Print("Loading Final Task Val Data...", output)
    final_val = load_spectra_wneg_data(dataset_builder, params, 'valid', device)
    Print("Done Loading Final Task Val Data with {} objects".format(len(final_val)), output)

    final_train_ds = Spectra_data(final_train)
    final_val_ds = Spectra_data(final_val)
    collate_fn = collate_spectra_data(molgraph_dict, params)

    with open(os.path.join(data_path, 'molgraph_dict.pkl'), 'wb') as f:
        pickle.dump(molgraph_dict, f)
    
    dl_params = {'batch_size': params['batch_size_train_final'],
                 'shuffle': True}
    train_dl = DataLoader(final_train_ds, collate_fn=collate_fn, **dl_params)

    dl_params = {'batch_size': params['batch_size_val_final'],
                 'shuffle': False}
    val_dl = DataLoader(final_val_ds, collate_fn=collate_fn, **dl_params)

    inter_model = INTER_MLP2(params)
    inter_model = inter_model.to(device)
    models_list.append([inter_model, "inter", False, False, False])
    load_models(params, models_list, device, output)

    #if params['contr_trg'] and params['frz_contr']:
    if params['frz_contr']:
        for model in models_list:
            idx = model[1]
            if idx != "inter":
                model[2] = True

    parameters, pr_params = [], []
    for model, idx, frz, _, _ in models_list:
        if frz: continue
        else:             parameters += [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.Adam([{'params':parameters,    'lr':params['final_lr']}])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60,90], gamma=0.1, verbose=True)
    train_wt = torch.Tensor([1.0, float(1.0)])
    train_wt = train_wt.to(device)
    loss_func = torch.nn.BCELoss()
    val_loss_func = torch.nn.BCELoss()
    train_contr_losses, train_inter_losses, train_losses = [],[],[]

    model_time = str(int(round(time.time() * 1000)))
    stopper = MyEarlyStopping(models_list, model_time, params, output,
        mode='lower', patience = params['early_stopping_patience'], 
        filename = "early_stopping/pred_"+ model_time)

    Print("Starting Final Training...", output)
    for epoch in range(params['num_epoch_final']):
        train_contr_loss, train_inter_loss, train_loss, val_inter_loss = 0.0,0.0,0.0,0.0
        train_ap = 0.0
        for model, idx, frz, _, _ in models_list: model.train()

        for batch_id, (batch_g, mz_b, int_b, pad, fp_b, y, lengths, inchi) in enumerate(tqdm(train_dl, total=int(len(train_dl)), leave=False)):
            batch_g = batch_g.to(torch.device(device))
            mz_b = mz_b.to(torch.device(device), dtype=torch.float32)
            int_b = int_b.to(torch.device(device))
            pad = pad.to(torch.device(device))
            fp_b = fp_b.to(torch.device(device))
            y = y.to(torch.device(device))
            y = y.to(dtype=torch.float32)
            #y = y.unsqueeze(1)
            mol_enc = mol_enc_model(batch_g, batch_g.ndata['h'])
            spec_enc = spec_enc_model(mz_b, int_b, pad, lengths)
            
            prediction = inter_model(mol_enc, spec_enc)
            prediction = prediction.squeeze(1)
            
            loss = loss_func(prediction, y)
            
            loss.backward()
            
            optimizer.step()
            optimizer.zero_grad()
            # logits = F.softmax(prediction, 1)
            # prediction = prediction[:,1]
            prediction = prediction.cpu()
            prediction = prediction.detach()
            y = y.cpu()
            train_ap += average_precision_score(y, prediction)
            train_inter_loss += loss.detach().item()
            
        scheduler.step()
        train_inter_loss /= (batch_id + 1)
        train_ap /= (batch_id + 1)
        predlist = torch.Tensor()
        labellist = torch.Tensor()
        val_loss = 0
        val_ap = 0.0
        
        for model, idx, frz, _, _ in models_list: model.eval()

        for batch_id, (batch_g, mz_b, int_b, pad, fp_b, y, lengths, inchi) in enumerate(val_dl):
            batch_g = batch_g.to(torch.device(device))
            mz_b = mz_b.to(torch.device(device))
            int_b = int_b.to(torch.device(device))
            pad = pad.to(torch.device(device))
            fp_b = fp_b.to(torch.device(device))
            y = y.to(torch.device(device))
            y = y.to(dtype=torch.float32)
            # y = y.unsqueeze(1)
            with torch.no_grad():

                mol_enc = mol_enc_model(batch_g, batch_g.ndata['h'])
                spec_enc = spec_enc_model(mz_b, int_b, pad, lengths)
                
                prediction = inter_model(mol_enc, spec_enc)
                prediction = prediction.squeeze(1)
                
            loss = val_loss_func(prediction, y)
            # logits = F.softmax(prediction, 1)
            # logits = logits[:,1]
            # logits = logits.cpu()
            y = y.cpu()
            # prediction = prediction[:,1]
            prediction = prediction.cpu()
            predlist = torch.cat([predlist, prediction])
            labellist = torch.cat([labellist, y])
            val_loss += loss.detach().item()
        
        val_loss /= (batch_id + 1)
        val_ap = average_precision_score(labellist, predlist)
        inline_log = 'Epoch {} / {}, train_inter_loss: {:.4f}, train_ap: {:.4f}, val_loss: {:.4f}, val_ap: {:.4f}'.format(
            epoch + 1, params['num_epoch_final'], train_inter_loss, train_ap, val_loss, val_ap)
        Print(inline_log, output)
        train_inter_losses.append(train_inter_loss)

        early_stop = stopper.step(val_loss)
        if early_stop:
            saved_model_name = "Saved early stopping model in " + stopper.filename
            Print(saved_model_name, output)
            break

    save_all_models(params, models_list, model_time + "_last", output)

    plt.figure()
    plt.plot(train_inter_losses)
    plt.ylim([0, max(train_inter_losses)])
    plt.legend(['train'], loc='upper left')
    plt.title('Loss')
    plt.savefig("logs/train_inter_losses_" + str(model_time) + ".png")
    loss_graph_file = "created loss graph in logs/train_inter_losses_" + str(model_time) + ".png"
    Print(loss_graph_file, output)