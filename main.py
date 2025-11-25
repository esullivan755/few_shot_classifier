from models.wrappers import MetaLearner, Base_Wrapper, PEFT_Wrapper
from models.dataset_utils import few_shot_text
from train.meta_training import inner_loop, meta_loop
from transformers import AutoTokenizer
from peft import LoraConfig
import torch

# Configed Factors:

LR = 1e-4
R = 8
LORA_ALPHA = 16
TARGET_MODULES = ["query_proj", "value_proj"]

if __name__ == "__main__":

    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
    special_tokens = {"additional_special_tokens": ["[HEAD]", "[TAIL]"]}
    tokenizer.add_special_tokens(special_tokens)
    
    # train_data = few_shot_text("data/fewrel_train.json", tokenizer, n_ways=5, k_shot=5, q_shot=10)
    # val_data = few_shot_text("data/fewrel_val.json", tokenizer, n_ways=5, k_shot=5, q_shot=10)
    
    # lora_cfg = LoraConfig(r=8, lora_alpha=16, target_modules=["query_proj", "value_proj"], lora_dropout=0.1)
    # model = PEFT_Wrapper(ModelClass=None, ModelName="microsoft/deberta-v3-base", lora_config=lora_cfg)
    # metalearner = MetaLearner(model, Is_PEFT=True)
    
    # optimizer = torch.optim.Adam(metalearner.parameters(), lr=1e-4)
    
    # meta_loop(train_data, val_data, tokenizer, metalearner, optimizer)



