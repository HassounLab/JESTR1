import argparse
import datetime
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rdkit import RDLogger
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from massspecgym.models.base import Stage

from jestr.data.data_module import TestDataModule
from jestr.data.datasets import ContrastiveDataset
from jestr.utils.data import get_spec_featurizer, get_mol_featurizer, get_test_ms_dataset
from jestr.utils.models import get_model

from jestr.definitions import TEST_RESULTS_DIR
import yaml
from functools import partial
# Suppress RDKit warnings and errors
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)

parser = argparse.ArgumentParser()
parser.add_argument("--param_pth", type=str, default="params.yaml")
parser.add_argument('--checkpoint_pth', type=str, default='')
parser.add_argument('--checkpoint_choice', type=str, default='train', choices=['train', 'val'])
parser.add_argument('--df_test_pth', type=str, help='result file name')
parser.add_argument('--exp_dir', type=str)
parser.add_argument('--candidates_pth', type=str)
def main(params):
    # Seed everything
    pl.seed_everything(params['seed'])
        
    # Init paths to data files
    if params['debug']:
        params['dataset_pth'] = "../data/sample/data.tsv"
        params['split_pth']=None
        params['df_test_path'] = os.path.join(params['experiment_dir'], 'debug_result.pkl')

    # Load dataset
    spec_featurizer = get_spec_featurizer(params['spectra_view'], params)
    mol_featurizer = get_mol_featurizer(params['molecule_view'], params)
    dataset = get_test_ms_dataset(params['spectra_view'], params['molecule_view'], spec_featurizer, mol_featurizer, params)

    # Init data module
    collate_fn = partial(ContrastiveDataset.collate_fn, spec_enc=params['spec_enc'], spectra_view=params['spectra_view'], stage=Stage.TEST)
    data_module = TestDataModule(
        dataset=dataset,
        collate_fn=collate_fn,
        batch_size=params['batch_size'],
        num_workers=params['num_workers']
    )

    model = get_model(params['model'], params)
    model.df_test_path = params['df_test_path']
    
    # Init trainer
    trainer = Trainer(
        accelerator=params['accelerator'],
        devices=params['devices'],
        default_root_dir=params['experiment_dir']
    )

    # Prepare data module to test
    data_module.prepare_data()
    data_module.setup(stage="test")
        
    # Test
    trainer.test(model, datamodule=data_module)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)

    # Load
    with open(args.param_pth) as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
    
    # Experiment directory
    exp_dir = ''
    if args.exp_dir:
        exp_dir = args.exp_dir
    else:
        run_name = params['run_name']
        for exp in os.listdir(TEST_RESULTS_DIR): # find exp dir with matching run_name
            if exp.endswith("_"+run_name):
                exp_dir = str(TEST_RESULTS_DIR / exp)
                break
    if not exp_dir: # if no exp dir provided, create a new one
        now = datetime.datetime.now().strftime("%Y%m%d")
        exp_dir = str(TEST_RESULTS_DIR / f"{now}_{params['run_name']}")
        os.makedirs(exp_dir, exist_ok=True)
    print("EXPERIMENT directory: ",exp_dir)
    params['experiment_dir'] = exp_dir
    
    # Look for checkpoint
    if args.checkpoint_pth:
        params['checkpoint_pth'] = args.checkpoint_pth
    elif params['checkpoint_pth'] != '':
        pass
    elif params['checkpoint_pth_spec_enc'] != '' and params['checkpoint_pth_mol_enc'] != '':
        pass
    else:
        print("No checkpoint provided. Using the checkpoint in the experiment directory")
        for f in os.listdir(exp_dir):
            if f.endswith("ckpt") and f.startswith("epoch") and args.checkpoint_choice in f:
                checkpoint_path = os.path.join(exp_dir, f)
                params['checkpoint_pth'] = checkpoint_path
                break
        if not params['checkpoint_pth']:
            raise ValueError("No checkpoint provided. Please provide a checkpoint path.")
    
    if args.candidates_pth:
        params['candidates_pth'] = args.candidates_pth
    if args.df_test_pth:
        params['df_test_path'] = os.path.join(exp_dir, args.df_test_pth)
    if not params['df_test_path']:
        params['df_test_path'] = os.path.join(exp_dir, f"result_{params['candidates_pth'].split('/')[-1].split('.')[0]}.pkl")
        
    main(params)


