import transformers
from transformers import AutoConfig

import re

class DefaultConfig:
    MODEL_NAME = "huggingface/CodeBERTa-small-v1"
    MODEL_CONFIG = AutoConfig.from_pretrained(MODEL_NAME)
    SEED = 42
    
    OPTION = ""
    MODEL_PATH = "data/{}_{}_model.bin".format(re.sub('[ \/:*?"<>|]',"",MODEL_NAME), OPTION)
    SAVE_SUB_FILE_NAME = "submission_{}_{}".format(re.sub('[ \/:*?"<>|]', '', MODEL_NAME), OPTION)
    
    TRAIN_BATCH = 32
    VALID_BATCH = 128
    TEST_BATCH = 64
    
    TRAIN_LOG_INTERVAL = 1
    VALID_LOG_INTERVAL = 1