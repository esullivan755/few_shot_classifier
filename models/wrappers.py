import os
from typing import Optional, Union, List, Dict
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, BitsAndBytesConfig, DebertaV2Config, DebertaV2Model
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.func import functional_call
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model


DEBUG = False

def adjust_debug(set):

    assert isinstance(set,bool), f'adjust_debug input must be boolean, got {type(set)}'

    global DEBUG
    DEBUG = set

    print(f"DEBUG MODE: {set}")

print(f'DEBUG MODE: {DEBUG}')

print(f"Torch Version: {torch.__version__}")


class PEFT_Wrapper(nn.Module):
    """
    This class wraps PEFT with LoRA to initalize and track model parameters
    with a frozen base and updateable LoRA params.
    """

    def __init__(self,
        ModelClass: PreTrainedModel,
        ModelName: str,
        lora_config: LoraConfig,
        gradient_checkpointing = True,
        base_params: Optional[List[str]] = None,
        num_heads: int = 2
    ):



        super().__init__()

        if gradient_checkpointing:
            config = DebertaV2Config.from_pretrained(ModelName)
            config.gradient_checkpointing = True
            model = DebertaV2Model.from_pretrained(ModelName,config=config)
        else:
            model = AutoModel.from_pretrained(ModelName)

        self.model = get_peft_model(model,lora_config)


        hidden_size = self.model.config.hidden_size
        print(f'hidden_size: {hidden_size}')
        #Adding on fully learned feed forward attention pooler & classifier head


        self.attn_pool = nn.Linear(hidden_size, num_heads)

        # TO DO : potentially add outer-learned dropout here
        self.dropout = nn.Dropout(self.model.config.hidden_dropout_prob)

        self.classifier = nn.Linear(hidden_size*num_heads, 2)




    def debug(self):

        inputs = {
    "input_ids": torch.randint(0, 1000, (4, 128)),
    "attention_mask": torch.ones(4, 128, dtype=torch.long)
    #,"labels": torch.tensor([0,1,0,1], dtype=torch.long)
                }

        #print("labels dtype:", inputs['labels'].dtype)
        print("num_labels:", self.model.config.num_labels)
        print("problem_type:", self.model.config.problem_type)


        out = self.model(**inputs)
        print(out.logits)



    def forward(self,params,inputs):

        new_inputs = {k:v for k,v in inputs.items() if k!= 'labels'}

        out = self.model(input_ids = new_inputs['input_ids'], attention_mask = new_inputs['attention_mask'], token_type_ids=new_inputs['token_type_ids'])

        latent = out.last_hidden_state #Latest batch latent outputs

        #Functional pooling and clasifier to adapt to the inner loop
        scores = F.linear(latent,params['attn_pool.weight'],params['attn_pool.bias']) # Batch size, tokens, num heads (per token)

        mask = inputs['attention_mask'].unsqueeze(-1)

        scores = scores.masked_fill(mask == 0, float('-inf')) #Replacing 0 with -inf so softmax ingores it

        attention_weights = torch.softmax(scores, dim=1)

        pooled = torch.einsum('bth,btk->bkh', latent, attention_weights)

        pooled = pooled.reshape(pooled.shape[0],-1)

        pooled = self.dropout(pooled)

        logits = F.linear(pooled, params['classifier.weight'],params['classifier.bias'])

        return logits





class Base_Wrapper(nn.Module):
    """
    This class contains the initialization and configuration of a
    non-PEFT wrapper to be passed through AdaptableWrapper
    """

    def __init__(self,
        ModelClass: PreTrainedModel,
        ModelName: str,
        base_params: Optional[List[str]] = None,
        num_heads: int = 2
    ):

        super().__init__()

        self.model = ModelClass.from_pretrained(ModelName)

        #Freeze Encoder
        for n,p in self.model.named_parameters():
            p.requires_grad_(False)

        hidden_size = self.model.config.hidden_size
        print(f'hidden_size: {hidden_size}')
        #Adding on fully learned feed forward attention pooler & classifier head
        #just classifier for now, LoRA should be training the encoder layers well enough

        #classification so 2 classes

        #if no specified params we just grab all LoRA injected adapters + classifier head to train


        #sanitation check that the specified params exist
        self.attn_pool = nn.Linear(hidden_size, num_heads)
        self.dropout = nn.Dropout(self.model.config.hidden_dropout_prob)
        self.classifier = nn.Linear(hidden_size*num_heads, 2)


    def forward(self, **inputs):

        # This entire pass will be functionalized in MetaLearner so we make sure to detach gradients for the frozen encoder
        if 'token_type_ids' in inputs.keys():

            with torch.no_grad():
                out = self.model(input_ids = inputs['input_ids'], attention_mask = inputs['attention_mask'],token_type_ids=inputs['token_type_ids'])
        else:
            with torch.no_grad():
                out = self.model(input_ids = inputs['input_ids'], attention_mask = inputs['attention_mask'])

            latent = out.last_hidden_state #Latest batch latent outputs

        scores = self.attn_pool(latent) # Batch size, tokens, num heads (per token)

        mask = inputs['attention_mask'].unsqueeze(-1)

        scores = scores.masked_fill(mask == 0, float('-inf')) #Replacing 0 with -inf so softmax ingores it

        attention_weights = torch.softmax(scores, dim=1)

        pooled = torch.einsum('bth,btk->bkh', latent, attention_weights)

        pooled = pooled.reshape(pooled.shape[0],-1)

        pooled = self.dropout(pooled)

        logits = self.classifier(pooled)


        return logits




class MetaLearner(nn.Module):

    """
    This class contains the initialization and forward pass of a meta learning
    model
    """
    def __init__(
        self,
        model_wrapper: Union[PEFT_Wrapper, Base_Wrapper],
        Is_PEFT: bool = False
    ):
    
        super().__init__()
        print('MetaLearner__init__')
        self.Is_PEFT = Is_PEFT
        self.model_wrapper = model_wrapper
        #Only make functional if meta learning, otherwise no point
        #well this is the metalearning class so its gonna be functional
        self.buffers = dict(self.model_wrapper.named_buffers())

        #all params need to be initialized and set to trainable or not trainable by this point at the very latest or inner_lrs will explode

        #This or cross entropy depending on the task
        self.loss_fn = nn.CrossEntropyLoss()

        # when making keys in nn.ParameterDict() pytorch gets cranky if its got periods in it :(

        self.inner_lrs = nn.ParameterDict(
            {
                name.replace('.','_'): nn.Parameter(torch.ones_like(param)*1e-3)
                for name, param in self.get_trainable_params_dict().items()
            })
        if DEBUG:
            print("num_labels: ", self.model.config.num_labels)
            print("problem_type: ", self.model.config.problem_type)
            print(":")



    def compute_loss(self, *args, **kwargs):
        """

        """

        return self.loss_fn(*args, **kwargs)


    def get_all_params_dict(self):
        """
        returns a dict of all trainable params in model
        """


        return dict(self.model_wrapper.named_parameters())

    def print_all_params(self):

        for i in self.get_all_params_dict().keys():
            print(i)


    def state_dict(self):
        # Combine state_dicts
        state = dict()
        # Base model + LoRA adapters
        state['base_model'] = self.model_wrapper.state_dict()
        # Learned learning rates
        state['learned_lrs'] = {k: v.cpu() for k, v in self.inner_lrs.items()}
        # Pooling and classifier layers
        state['attn_pool'] = self.model_wrapper.attn_pool.state_dict()
        state['classifier'] = self.model_wrapper.classifier.state_dict()

        return state

    def load_state_dict(self, state):
        # Load base model
        self.model_wrapper.load_state_dict(state['base_model'])
        # Load learned lrs
        for k, v in state['learned_lrs'].items():
            self.inner_lrs[k].data.copy_(v)
        # Load pooling and classifier layers
        self.model_wrapper.attn_pool.load_state_dict(state['attn_pool'])
        self.model_wrapper.classifier.load_state_dict(state['classifier'])

    def get_trainable_params_dict(self):
        """
        returns a dict of all trainable params in model
        """

        trainable = {}
        for n,p in self.model_wrapper.named_parameters():
            if p.requires_grad:
                trainable[n] = p
        return trainable



    def print_trainable_params(self):
        """
        Prints out all the trainable params
        """

        for name in self.get_trainable_params_dict().keys():
            print(name)


    def forward(self, params: dict, inner: bool = True,  **inputs):
        """
        Forward pass for the metalearner. Updates inner params.
        This is a specifically functional_call
        """


        labels = inputs['labels']



        if self.Is_PEFT:
            logits = self.model_wrapper(params=params,inputs=inputs)

        else:
            #Slower attention due to functional call issues with pytorches newer flash attention for certain transformers
            with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
                logits = functional_call(self.model_wrapper, {**params,**self.buffers}, (), kwargs=inputs, strict=True)


        if DEBUG:
            print(f'Logits are shape {logits.shape}')

        loss = self.compute_loss(logits,labels)




        if inner:

            #calculating all the gradients, purposefully setting allow_unused = True because we pass in all params

            trainable_params = {n:p for n,p in params.items() if p.requires_grad and 'lora' not in n} # Only computing trainable gradients to save computation




            grads = torch.autograd.grad(loss, trainable_params.values(), create_graph=True, allow_unused=True) # allow_unused = True for easy debugging

            if DEBUG:
                for (n,v),g in zip(trainable_params.items(),grads):
                    if g is None:
                        print(f"nonetype gradient associated with parameter: {n}")

            adapted_params = {
                    name: param - self.inner_lrs[name.replace('.', '_')] * grad

                    for (name, param), grad in zip(trainable_params.items(), grads)
                    }


            for n,p in self.get_all_params_dict().items():
                if not p.requires_grad:
                    adapted_params[n] = p


            return adapted_params

        #for query there is no backprop, just loss
        else:
            preds = torch.argmax(logits, dim=1)
            correct = (preds == labels).sum().item()
            accuracy = correct / labels.size(0)

            return loss, accuracy
