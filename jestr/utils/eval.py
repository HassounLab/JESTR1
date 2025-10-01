from MassSpecGym.massspecgym.utils import MyopicMCES
import numpy as np
import tqdm
from multiprocessing import Pool

import os
import pandas as pd

class Compute_Myopic_MCES:
    mces_compute = MyopicMCES()
    

    def compute_mces(tar_cand):
        target, cand = tar_cand
        
        dist = Compute_Myopic_MCES.mces_compute(target, cand)
        return (tar_cand, dist)
    
    def compute_mces_parallel(target_cand_list, n_processes=25):


        with Pool(processes=n_processes) as pool:
            results = list(tqdm.tqdm(pool.imap(Compute_Myopic_MCES.compute_mces, target_cand_list), total=len(target_cand_list)))
        return results

class Compute_Myopic_MCES_timeout:
    mces_compute = MyopicMCES()

    @staticmethod
    def compute_mces(tar_cand):
        target, cand = tar_cand
        dist = Compute_Myopic_MCES.mces_compute(target, cand)
        return (tar_cand, dist)

    @staticmethod
    def compute_mces_parallel(target_cand_list, n_processes=35, timeout=60):  # timeout in seconds
        results = []

        with Pool(processes=n_processes) as pool:
            async_results = [
                pool.apply_async(Compute_Myopic_MCES.compute_mces, args=(tar_cand,))
                for tar_cand in target_cand_list
            ]
            for async_res in tqdm.tqdm(async_results, total=len(target_cand_list)):
                try:
                    result = async_res.get(timeout=timeout)
                except Exception as e:
                    # You can log the error or return a default value
                    result = (None, f"Timeout or error")
                results.append(result)

        return results

    
def get_result_files(exp_dir, spec_type, views_type):
    files = os.listdir(exp_dir)
    mass_result = ''
    form_result = ''

    for f in files:
        try:
            _, s, views = f.split('_')
        except:
            continue
        
        if s == spec_type and views == views_type:
            print(exp_dir / f)

            files = os.listdir(exp_dir / f)
            for fr in files:
                if 'mass_result' in fr:
                    mass_result = exp_dir / f / fr
                elif 'result' in fr:
                    form_result = exp_dir / f/ fr
            
    return mass_result, form_result

# get target
def get_target(candidates, labels):
    return np.array(candidates)[labels][0]

# get mol rank at 1
def get_top_cand(candidates, scores):
    return candidates[np.argmax(scores)]

# split into hit rates
def convert_rank_to_hit_rates(row, rank_col ,top_k=[1,5,20]):
    top_k_hits ={}
    rank = row[rank_col]
    for k in top_k:
        if rank <= k:
            top_k_hits[f'{rank_col}-hit_rate@{k}'] = 1
        else:
            top_k_hits[f'{rank_col}-hit_rate@{k}'] = 0
    return pd.Series(top_k_hits)

#################### Rank aggregation #######################
from collections import defaultdict
import numpy as np
from scipy.stats import rankdata

def borda_count(candidates, score_lists, target):
    scores = defaultdict(int)
    N = len(candidates)
    for score_list in score_lists:
        ranked_list = sorted(zip(candidates, score_list), key=lambda x: x[1], reverse=True)
        for rank, (mol, _) in enumerate(ranked_list, start=1):
            scores[mol] += N - rank + 1
    ranked_candidates = [mol for mol, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]
    return ranked_candidates.index(target) + 1 if target in ranked_candidates else None

def average_rank(candidates, score_lists, target):
    rank_sums = defaultdict(list)
    for score_list in score_lists:
        ranked_list = sorted(zip(candidates, score_list), key=lambda x: x[1], reverse=True)
        for rank, (mol, _) in enumerate(ranked_list, start=1):
            rank_sums[mol].append(rank)
    avg_ranks = {mol: np.mean(ranks) for mol, ranks in rank_sums.items()}
    ranked_candidates = [mol for mol, _ in sorted(avg_ranks.items(), key=lambda x: x[1])]
    return ranked_candidates.index(target) + 1 if target in ranked_candidates else None

def reciprocal_rank_aggregation(candidates, score_lists, target):
    scores = defaultdict(float)
    for score_list in score_lists:
        ranked_list = sorted(zip(candidates, score_list), key=lambda x: x[1], reverse=True)
        for rank, (mol, _) in enumerate(ranked_list, start=1):
            scores[mol] += 1 / rank
    ranked_candidates = [mol for mol, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]
    return ranked_candidates.index(target) + 1 if target in ranked_candidates else None

def weighted_voting(candidates, score_lists, weights, target):
    scores = defaultdict(float)
    for weight, score_list in zip(weights, score_lists):
        ranked_list = sorted(zip(candidates, score_list), key=lambda x: x[1], reverse=True)
        for rank, (mol, _) in enumerate(ranked_list, start=1):
            scores[mol] += weight / rank
    ranked_candidates = [mol for mol, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]
    return ranked_candidates.index(target) + 1 if target in ranked_candidates else None

def median_rank(candidates, score_lists, target):
    rank_sums = defaultdict(list)
    for score_list in score_lists:
        ranked_list = sorted(zip(candidates, score_list), key=lambda x: x[1], reverse=True)
        for rank, (mol, _) in enumerate(ranked_list, start=1):
            rank_sums[mol].append(rank)
    median_ranks = {mol: np.median(ranks) for mol, ranks in rank_sums.items()}
    ranked_candidates = [mol for mol, _ in sorted(median_ranks.items(), key=lambda x: x[1])]
    return ranked_candidates.index(target) + 1 if target in ranked_candidates else None

def score_based_aggregation(candidates, score_lists, target):
    scores = defaultdict(list)
    for score_list in score_lists:
        for mol, score in zip(candidates, score_list):
            scores[mol].append(score)
    avg_scores = {mol: np.mean(vals) for mol, vals in scores.items()}
    ranked_candidates = [mol for mol, _ in sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)]
    return ranked_candidates.index(target) + 1 if target in ranked_candidates else None
