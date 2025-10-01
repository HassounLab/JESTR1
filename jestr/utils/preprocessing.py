import pandas as pd
import pickle
import numpy as np
import jestr.utils.data as data_utils
import collections
import os
import requests
import tqdm
from multiprocessing import Pool
from urllib.parse import quote
from tqdm import tqdm

class NPClassProcess:
    def process_smiles(smiles):
        try:
            encoded_smiles = quote(smiles)
            url = f"https://npclassifier.gnps2.org/classify?smiles={encoded_smiles}"
            r = requests.get(url)
            return (smiles, r.json())
        except:
            return (smiles, None)
    
    def NPclass_from_smiles(pth, output_dir, n_processes=20):

        data = pd.read_csv(pth, sep='\t')
        unique_smiles = data['smiles'].unique().tolist()

        items = unique_smiles

        with Pool(processes=n_processes) as pool:
            results = list(tqdm(pool.imap(NPClassProcess.process_smiles, items), total=len(items)))

        failed_ct = 0
        smiles_to_class = {}
        for s, out in results:
            if out is None:
                smiles_to_class[s] = 'NA'
                failed_ct+=1
            else:
                smiles_to_class[s] = out
        file_pth = os.path.join(output_dir, 'SMILES_TO_CLASS.pkl')
        with open(file_pth, 'wb') as f:
            pickle.dump(smiles_to_class, f)
        print(f'Failed to process {failed_ct} SMILES')
        print(f'result file saved to {file_pth}')
        return file_pth



def construct_NL_spec(pth, output_dir):
    def _get_spec(row):
        mzs = np.array([float(m) for m in row["mzs"].split(",")], dtype=np.float32)
        intensities = np.array([float(i) for i in row["intensities"].split(",")],dtype=np.float32)
        mzs = float(row['precursor_mz']) - mzs
        valid_idx = np.where(mzs>1.0)
        mzs = mzs[valid_idx]
        intensities = intensities[valid_idx]

        sorted_idx = np.argsort(mzs)
        mzs = np.concatenate((mzs[sorted_idx], [float(row['precursor_mz'])]))
        intensities = np.concatenate((intensities[sorted_idx], [1.0]))

        return mzs, intensities
    
    spec_data = pd.read_csv(pth, sep='\t')
    spec_data[['mzs', 'intensities']] = spec_data.apply(lambda row: _get_spec(row), axis=1, result_type='expand')
    
    file_pth = os.path.join(output_dir, 'NL_spec.pkl')
    with open(file_pth, 'wb') as f:
        pickle.dump(spec_data, f)
    return file_pth

def generate_cons_spec(pth, output_dir):
    spec_data = pd.read_csv(pth, sep='\t')
    data_by_smiles = spec_data[['identifier', 'smiles', 'mzs', 'intensities', 'fold']].groupby('smiles').agg({'identifier':list, 'mzs':lambda x: ','.join(x), 'intensities': lambda x: ','.join(x), 'fold':list})
    smiles_to_fold  = dict(zip(data_by_smiles.index.tolist(), data_by_smiles['fold'].tolist()))

    consensus_spectra = {}
    for idx, row in tqdm(data_by_smiles.iterrows(), total=len(data_by_smiles)):
        mzs = np.array([float(m) for m in row["mzs"].split(",")], dtype=np.float32)
        intensities = np.array([float(i) for i in row["intensities"].split(",")],dtype=np.float32)

        sorted_idx = np.argsort(mzs)
        mzs = mzs[sorted_idx]
        intensities = intensities[sorted_idx]
        smiles = row.name
        
        consensus_spectra[smiles] = {'mzs':mzs, 'intensities':intensities,'precursor_mz': 10000.0,
                    'fold': smiles_to_fold[smiles][0]}
    
    df = pd.DataFrame.from_dict(consensus_spectra, orient='index')
    df = df.rename_axis('smiles').reset_index()

    return df


def generate_cons_spec_formulas(pth, subformula_dir, output_dir=''):
    # load tsv file
    spec_data = pd.read_csv(pth, sep='\t')

    # goup spectra by SMILES
    data_by_smiles = spec_data[['identifier', 'smiles', 'fold', 'precursor_mz', 'formula', 'adduct']].groupby('smiles').agg({'identifier':list, 'fold': list, 'formula': list, 'precursor_mz': "max", 'adduct': list})
    smiles_to_id = dict(zip(data_by_smiles.index.tolist(), data_by_smiles['identifier'].tolist()))
    smiles_to_fold  = dict(zip(data_by_smiles.index.tolist(), data_by_smiles['fold'].tolist()))
    smiles_to_precursorMz = dict(zip(data_by_smiles.index.tolist(), data_by_smiles['precursor_mz'].tolist()))
    smiles_to_precursorFormula = dict(zip(data_by_smiles.index.tolist(), data_by_smiles['formula'].tolist()))
    # load subformulas
    subformulaLoader = data_utils.Subformula_Loader(spectra_view='SpecFormula', dir_path=subformula_dir)
    id_to_spec = subformulaLoader(spec_data['identifier'].tolist())

    # combine spectra
    consensus_spectra = {}
    for smiles, ids in tqdm(smiles_to_id.items(), total=len(data_by_smiles)):
        cons_spec = collections.defaultdict(list)
        for id in ids:
            if id in id_to_spec:
                for k, v in id_to_spec[id].items():
                    cons_spec[k].extend(v)
        cons_spec = pd.DataFrame(cons_spec)

        assert(len(set(smiles_to_fold[smiles]))==1)

        # keep maxed mz and maxed intensity
        try:
            cons_spec = cons_spec.groupby('formulas').agg({'formula_mzs': "max", 'formula_intensities': "max"})
            cons_spec.reset_index(inplace=True)
        except:
            d = {
                'formulas': [smiles_to_precursorFormula[smiles][0]],
                'formula_mzs': [smiles_to_precursorMz[smiles]],
                'formula_intensities': [1.0]
            }
            cons_spec = pd.DataFrame(d)
        
        cons_spec = cons_spec.sort_values(by='formula_mzs').reset_index(drop=True)
        cons_spec = {'formulas': cons_spec['formulas'].tolist(),
                    'formula_mzs': cons_spec['formula_mzs'].tolist(),
                    'formula_intensities': cons_spec['formula_intensities'].tolist(), 
                    'precursor_mz': smiles_to_precursorMz[smiles],
                    'fold': smiles_to_fold[smiles][0],
                    'precursor_formula': smiles_to_precursorFormula[smiles][0]}# formula without adduct...

        consensus_spectra[smiles] = cons_spec

    # save consensus spectra
    df = pd.DataFrame.from_dict(consensus_spectra, orient='index')
    df = df.rename_axis('smiles').reset_index()

    return df