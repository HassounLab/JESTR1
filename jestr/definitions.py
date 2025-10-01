"""Global variables used across the package."""
import pathlib

# Dirs
ROOT_DIR = pathlib.Path(__file__).parent.absolute()
REPO_DIR = ROOT_DIR.parent
DATA_DIR = REPO_DIR / 'data'
TEST_RESULTS_DIR = REPO_DIR / 'experiments'
ASSETS_DIR = REPO_DIR / 'assets'

# C
CHEM_ELEMS_SMALL = ['H', 'C',  'O', 'N', 'P', 'S', 'Cl', 'F', 'Br', 'I', 'B', 'As', 'Si', 'Se']

MSGYM_FORMULA_VECTOR_NORM = [102.0, 59.0, 25.0, 13.0, 3.0, 6.0, 6.0, 17.0, 4.0, 4.0, 1.0, 1.0, 5.0, 2.0]

#MSGYM standardization
MSGYM_STANDARD_MH = {
    'mz_mean': 195.155185,
    'mz_std':127.591549
}
MSGYM_STANDARD_all = { # got these from Yinkai
"mz_mean": 80.88304948022557,
"mz_std" : 197.4588028571758}