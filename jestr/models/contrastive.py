import typing as T
import torch
import torch.nn as nn
import pandas as pd
from collections import defaultdict
import numpy as np
import os
from massspecgym.models.retrieval.base import RetrievalMassSpecGymModel
from massspecgym.models.base import Stage
from massspecgym import utils

from jestr.utils.loss import contrastive_loss, cand_spec_sim_loss, fp_loss, cons_spec_loss
import jestr.utils.models as model_utils

import torch.nn.functional as F


class ContrastiveModel(RetrievalMassSpecGymModel):
    def __init__(
        self,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.save_hyperparameters()

        self.spec_enc_model = model_utils.get_spec_encoder(self.hparams.spec_enc, self.hparams)
        self.mol_enc_model = model_utils.get_mol_encoder(self.hparams.mol_enc, self.hparams)
            
        self.spec_view = self.hparams.spectra_view
        
        # result storage for testing results
        self.result_dct = defaultdict(lambda: defaultdict(list))
                    
    def forward(self, batch, stage):
        g = batch['cand'] if stage == Stage.TEST else batch['mol']
        
        spec = batch[self.spec_view]
        n_peaks = batch['n_peaks'] if 'n_peaks' in batch else None
        spec_enc = self.spec_enc_model(spec, n_peaks)

        mol_enc = self.mol_enc_model(g)

        return spec_enc, mol_enc

    def compute_loss(self, batch: dict, spec_enc, mol_enc):
        loss = 0
        losses = {}
        contr_loss, _, _ = contrastive_loss(spec_enc, mol_enc, self.hparams.contr_temp)
        
        loss+=contr_loss
        losses['loss'] = loss 

        return losses
    
    def step(
        self, batch: dict, stage= Stage.NONE):
        
        # Compute spectra and mol encoding
        spec_enc, mol_enc = self.forward(batch, stage)

        if stage == Stage.TEST:
            return dict(spec_enc=spec_enc, mol_enc=mol_enc)

        # Calculate loss
        losses = self.compute_loss(batch, spec_enc, mol_enc)

        return losses

    def on_batch_end(self, outputs, batch: dict, batch_idx: int, stage: Stage) -> None:
        # total loss
        self.log(
            f'{stage.to_pref()}loss',
            outputs['loss'],
            batch_size=len(batch['identifier']),
            sync_dist=True,
            prog_bar=True,
            on_epoch=True,
            # on_step=True
        )
    
    def test_step(self, batch, batch_idx):
        # Unpack inputs
        identifiers = batch['identifier']
        cand_smiles = batch['cand_smiles']
        id_to_ct = defaultdict(int)
        for i in identifiers: id_to_ct[i]+=1
        batch_ptr = torch.tensor(list(id_to_ct.values()))

        outputs = self.step(batch, stage=Stage.TEST)
        spec_enc = outputs['spec_enc']
        mol_enc = outputs['mol_enc']

        # Calculate scores
        indexes = utils.batch_ptr_to_batch_idx(batch_ptr)
        
        scores = nn.functional.cosine_similarity(spec_enc, mol_enc)
        scores = torch.split(scores, list(id_to_ct.values()))

        cand_smiles = utils.unbatch_list(batch['cand_smiles'], indexes)
        labels = utils.unbatch_list(batch['label'], indexes)
        
        return dict(identifiers=list(id_to_ct.keys()), scores=scores, cand_smiles=cand_smiles, labels=labels)
    
    def on_test_batch_end(self, outputs, batch: dict, batch_idx: int, stage: Stage = Stage.TEST) -> None:
        
        # save scores
        for i, cands, scores, l in zip(outputs['identifiers'], outputs['cand_smiles'], outputs['scores'], outputs['labels']):
            self.result_dct[i]['candidates'].extend(cands)
            self.result_dct[i]['scores'].extend(scores.cpu().tolist())
            self.result_dct[i]['labels'].extend([x.cpu().item() for x in l])
            
    def _compute_rank(self, scores, labels):
        if not any(labels):
            return -1
        scores = np.array(scores)
        target_score = scores[labels][0]
        rank = np.count_nonzero(scores >=target_score)
        return rank
    
    def _sort_candidates(self, scores, candidates):
        scores = np.array(scores)
        candidates = np.array(candidates)
        sorted_indices = np.argsort(scores)[::-1]
        sorted_candidates = candidates[sorted_indices]
        sorted_scores = scores[sorted_indices]
        return sorted_candidates.tolist(), sorted_scores.tolist()
    
    def on_test_epoch_end(self) -> None:

        self.df_test = pd.DataFrame.from_dict(self.result_dct, orient='index').reset_index().rename(columns={'index': 'identifier'})

        # Compute rank only if smiles is known and provided
        if self.hparams.candidate_file_key == 'smiles':
            self.df_test['rank'] = self.df_test.apply(lambda row: self._compute_rank(row['scores'], row['labels']), axis=1)

        else:

            self.df_test['sorted_candidates'], self.df_test['sorted_scores'] = zip(*self.df_test.apply(lambda row: self._sort_candidates(row['scores'], row['candidates']), axis=1))
            self.df_test = self.df_test[['identifier', 'sorted_candidates', 'sorted_scores']]
        self.df_test.to_pickle(self.df_test_path)

    def get_checkpoint_monitors(self) -> T.List[dict]:
        monitors = [
            {"monitor": f"{Stage.TRAIN.to_pref()}loss", "mode": "min", "early_stopping": False}, # monitor train loss
        ]
        return monitors
    
    def _update_loss_weights(self)-> None:
        if self.hparams.loss_strategy == 'linear':
            for loss in self.loss_wts:
                self.loss_wts[loss] += self.loss_updates[loss]
        elif self.hparams.loss_strategy == 'manual':
            for loss in self.loss_wts:
                if self.current_epoch in self.loss_updates[loss]:
                    self.loss_wts[loss] = self.loss_updates[loss][self.current_epoch]

    def on_train_epoch_end(self) -> None:
        self._update_loss_weights()      