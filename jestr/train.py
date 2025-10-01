import argparse
import datetime

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rdkit import RDLogger
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from jestr.data.data_module import ContrastiveDataModule

from jestr.definitions import TEST_RESULTS_DIR
import yaml
from jestr.data.datasets import ContrastiveDataset
from functools import partial

from jestr.utils.data import get_ms_dataset, get_spec_featurizer, get_mol_featurizer
from jestr.utils.models import get_model
# Suppress RDKit warnings and errors
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)

parser = argparse.ArgumentParser()
parser.add_argument("--param_pth", type=str, default="params.yaml")

def main(params):
    # Seed everything
    pl.seed_everything(params['seed'])
        
    # Init paths to data files
    if params['debug']:
        params['dataset_pth'] = "../data/sample/data.tsv"
        params['candidates_pth'] =None
        params['split_pth']=None

    # Load dataset
    spec_featurizer = get_spec_featurizer(params['spectra_view'], params)
    mol_featurizer = get_mol_featurizer(params['molecule_view'], params)
    dataset = get_ms_dataset(params['spectra_view'], params['molecule_view'], spec_featurizer, mol_featurizer, params)
    
    # Init data module
    collate_fn = partial(ContrastiveDataset.collate_fn, spec_enc=params['spec_enc'], spectra_view=params['spectra_view'])
    data_module = ContrastiveDataModule(
        dataset=dataset,
        collate_fn=collate_fn,
        split_pth=params['split_pth'],
        batch_size=params['batch_size'],
        num_workers=params['num_workers'],
    )

    model = get_model(params['model'], params)

    # Init logger
    if params['no_wandb']:
        logger = None
    else:
        logger = pl.loggers.WandbLogger(
            save_dir=params['experiment_dir'],
            dir=params['experiment_dir'],
            log_dir=params['experiment_dir'],
            name=params['run_name'],
            project=params['project_name'],
            log_model=False,
            config=model.hparams
        )

    # Init callbacks for checkpointing and early stopping
    callbacks = [pl.callbacks.ModelCheckpoint(save_last=False) ]
    for i, monitor in enumerate(model.get_checkpoint_monitors()):
        monitor_name = monitor['monitor']
        checkpoint = pl.callbacks.ModelCheckpoint(
            monitor=monitor_name,
            save_top_k=1,
            mode=monitor['mode'],
            dirpath=params['experiment_dir'],
            filename=f'{{epoch}}-{{{monitor_name}:.2f}}',
            # filename='{epoch}-{val_loss:.2f}-{train_loss:.2f}',
            auto_insert_metric_name=True,
            save_last=(i == 0)
        )
        callbacks.append(checkpoint)
        if monitor.get('early_stopping', False):
            early_stopping = EarlyStopping(
                monitor=monitor_name,
                mode=monitor['mode'],
                verbose=True,
                patience=params['early_stopping_patience'],
            )
            callbacks.append(early_stopping)

    # Init trainer
    trainer = Trainer(
        accelerator=params['accelerator'],
        devices=params['devices'],
        max_epochs=params['max_epochs'],
        logger=logger,
        log_every_n_steps=params['log_every_n_steps'],
        val_check_interval=params['val_check_interval'],
        callbacks=callbacks,
        default_root_dir=params['experiment_dir'],
    )

    # Prepare data module to validate or test before training
    data_module.prepare_data()
    data_module.setup()


    # Validate before training
    trainer.validate(model, datamodule=data_module)

    # Train
    trainer.fit(model, datamodule=data_module)
        


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)

    # Get current time
    now = datetime.datetime.now()
    now_formatted = now.strftime("%Y%m%d")

    # Load
    with open(args.param_pth) as f:
        params = yaml.load(f, Loader=yaml.FullLoader)

    experiment_dir = str(TEST_RESULTS_DIR / f"{now_formatted}_{params['run_name']}")
    params['experiment_dir'] = experiment_dir

    if not params['df_test_path']:
        params['df_test_path'] = os.path.join(experiment_dir, "result.pkl")

    main(params)
