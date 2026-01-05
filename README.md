# BF-m7GPred
 BF-m7GPred is a dual-branch deep learning framework that integrates single-nucleotide level embeddings and motif-level embeddings for m7G modification sites prediction. 
## How to Use
 1. Download DNABERT-2-117M from https://huggingface.co/zhihan1996/DNABERT-2-117M to folder ./dnabert2.
 2. Configuration environment: conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch. Besides, make sure your environment support DNABERT2.
 3. python myModel.py. For multi-GPU training, please use torchrun --nproc_per_node Your_nodes myModel.py.
 4. The best model is saved in ./output/best_model. The evaluation result is saved in ./output/results.
    
