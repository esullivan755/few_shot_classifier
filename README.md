Author: Eric Sullivan
Description: The current state of this repo is focues on metalearning for large pretrained transformers. The highest accuracy I have found under my low compute availability is with inner training loop 
updates (inter-task) on the final pooling and classification layer parameters and learning rates, and performing LoRA steps on the outer loop (intra-task). If you would like to experiment, simply import 
these packages in your environment, choose a base model from huggingface (supports BERT, roBERTa, deBERTa), and wrap it in the base or peft wrapper, then call this model instance under the MetaLearner 
wrapper, pick an optimizer, and train! 
