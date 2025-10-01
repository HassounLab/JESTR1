# **JESTR: Joint Embedding Space Technique for Ranking Candidate Molecules for the Annotation of Untargeted Metabolomics Data**

V2. This repository contains Python code to train and test the JESTR model and includes the NPLIB1 dataset for demonstration.

---

## **Overview**
JESTR learns a joint embedding of spectra and candidate molecules to rank candidates for untargeted metabolomics annotation. We provide:
- **Pretrained weights** for the NPLIB1 dataset
- **Sample data** and a **demo notebook** for quick evaluation
- **Training scripts** for contrastive pretraining

JESTR uses [PyTorch](https://pytorch.org). The released weights were trained on NVIDIA A100 (CUDA 11.8). Please ensure your environment supports GPU execution.

---

## **Quickstart**

### **1) Prepare the environment**
- Required packages are listed in `jestr_requirements.txt`.
- We recommend a fresh Conda environment.

```bash
# Create and activate environment
conda create -n jestr -y python=3.10
conda activate jestr

# Install dependencies (pip)
pip install -r jestr_requirements.txt
```

If you prefer Conda-only workflows, use the versions pinned in `jestr_requirements.txt` as guidance. See Conda docs: `https://docs.conda.io/en/latest/`.


### **2) Get the data**
- For a quick demo, use the provided sample under `data/sample/`:
  - `data/sample/data.tsv`: spectra data and metadata
  - `data/sample/identifier_to_candidates.json`: Mapping of spectra identifier to a list of candidate SMILES
- For larger experiments, you may also use the MassSpecGym dataset from Hugging Face: [MassSpecGym](https://huggingface.co/datasets/roman-bushuiev/MassSpecGym). 


### **3) Configure `params.yaml`**
Create or edit `params.yaml`. Key fields:
- `run_name`: experiement directory name
- `checkpoint_pth_spec_enc`: spectra encoder checkpoint
- `checkpoint_pth_mol_enc`: molecule encoder checkpoint
- `candidates_pth`: path to your identifier_to_candidate.json
- `dataset_pth`: path to your data.tsv
```

### **4) Evaluate**
```bash
python test.py
```

## **Training**
1. Prepare data in the same format as the [MassSpecGym dataset](https://huggingface.co/datasets/roman-bushuiev/MassSpecGym).
2. Edit `params.yaml`, make sure all checkpoint paths are empty if training from scratch. 

3. Train and evaluate
```bash
# train
python train.py

# test/evaluate
python test.py
```

## **Notes on licensing and datasets**
We release the NPLIB1 dataset and pretrained weights. Other datasets may be subject to licensing restrictions and are not included. If you test other datasets, confirm you have appropriate licenses and access.

---

## **Troubleshooting**
- **GPU/driver errors**: Ensure CUDA toolkit and drivers compatible with your GPU (e.g., CUDA 11.8 for A100) are installed and visible to PyTorch.
- **Dependency conflicts**: Recreate the environment and strictly follow `jestr_requirements.txt`.

---

## **License**
This project is licensed under the MIT license.

