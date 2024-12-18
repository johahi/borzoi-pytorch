# borzoi-pytorch
Implementation of the [Borzoi model](https://www.biorxiv.org/content/10.1101/2023.08.30.555582v1) from Calico in Pytorch! Original implementation and weights are [here](https://github.com/calico/borzoi). 

## Installation
borzoi-pytorch is avaiable on PyPI and can be installed with
`pip install borzoi-pytorch`

## Pretrained Models

### Borzoi

Ported weights (with permission) are uploaded to <a href="https://huggingface.co/johahi"> Huggingface</a>, the model (human or mouse heads) can be loaded with:

```python
from borzoi_pytorch import Borzoi
borzoi = Borzoi.from_pretrained('johahi/borzoi-replicate-0') # 'johahi/borzoi-replicate-[0-3][-mouse]'
````
The Pytorch version produces the same predictions as the original implementation, see for instance in this [notebook](https://github.com/johahi/borzoi-pytorch/blob/main/notebooks/pytorch_borzoi_example_eqtl_chr10_116952944_T_C.ipynb).  

### Flashzoi

After installation of [FlashAttention-2](https://github.com/Dao-AILab/flash-attention#installation-and-features), Flashzoi offers 3x the speed of Borzoi at comparable or slightly better predictive performance and can be loaded with:

<img width="1288" alt="image" src="https://github.com/user-attachments/assets/bda016b9-1cd5-4377-a771-726f0285613a" />

```python
from borzoi_pytorch import Borzoi
borzoi = Borzoi.from_pretrained('johahi/flashzoi-replicate-0') # 'johahi/flashzoi-replicate-[0-3]'
````
Note that this model should/must be run in autocast, and requires a modern Nvidia GPU.

## Misc.
The relative shift operation should be [faster](https://johahi.github.io/blog/2024/fast-relative-shift/) than in enformer_pytorch or other implementations. 

## References
<a id="1">[1]</a> 
Predicting RNA-seq coverage from DNA sequence as a unifying model of gene regulation  
Johannes Linder, Divyanshi Srivastava, Han Yuan, Vikram Agarwal, David R. Kelley  
bioRxiv 2023.08.30.555582; doi: [https://doi.org/10.1101/2023.08.30.555582](https://www.biorxiv.org/content/10.1101/2023.08.30.555582v1)  
<a id="2">[2]</a> 
[enformer-pytorch github](https://github.com/lucidrains/enformer-pytorch/),
Phil Wang *lucidrains*

## Citation
Please cite this repository if you used borzoi-pytorch.
