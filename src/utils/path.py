from pathlib import Path


# Project and source files
PROJECT_PATH = Path(__file__).parent.parent.parent
CONFIGS_PATH = PROJECT_PATH / 'configs'
DATASETS_PATH = PROJECT_PATH / 'datasets'
RAW_DATA_PATH = PROJECT_PATH / 'raw_data'
RESULTS_PATH = PROJECT_PATH / 'results'
MODELS_PATH = PROJECT_PATH / 'models'

# Synthetic
SYNTHETIC_DOCUMENT_DATASET_PATH = DATASETS_PATH / 'syndoc'
SYNTHETIC_LINE_DATASET_PATH = DATASETS_PATH / 'synline'
SYNTHETIC_RESRC_PATH = PROJECT_PATH / 'synthetic_resource'
