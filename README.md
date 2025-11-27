Author: Eric Sullivan


/n
Description: The current state of this repo is focuses on metalearning for large pretrained transformers. The highest accuracy I have found under my low compute availability is with inner training loop updates (inter-task) on the final pooling and classification layer parameters and learning rates, and performing LoRA steps on the outer loop (intra-task). If you would like to experiment, follow the base script in main.py with your own models. 
