# borzoi-pytorch
The [Borzoi model](https://www.biorxiv.org/content/10.1101/2023.08.30.555582v1) from Calico, but ported to Pytorch! Original implementation and weights are [here](https://github.com/calico/borzoi).  
We show that the Pytorch version produces the same predictions as the original implementation for all tested notebook examples.  

## Pretrained Model

Ported weights (with permission) are uploaded to [huggingface](https://huggingface.co/johahi), the model (human or mouse heads) can be loaded with:

```python
from borzoi_pytorch import Borzoi
borzoi = Borzoi.from_pretrained('johahi/borzoi-replicate-0') # 'johahi/borzoi-replicate-[0-3][-mouse]'
````


## Installation
`pip install git+https://github.com/johahi/borzoi-pytorch.git`

## Misc.
Documentation coming soon. 
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
