# borzoi-pytorch
Pytorch implementation of the [Borzoi](https://doi.org/10.1038/s41588-024-02053-6) model from Calico, [Flashzoi](https://doi.org/10.1093/bioinformatics/btaf467), an up to 3x faster Borzoi enhancement, and [Borzoi Prime](https://doi.org/10.1101/2025.06.10.658961)! 

## Installation
borzoi-pytorch is available on PyPI and can be installed with
`pip install borzoi-pytorch`

## Pretrained Models

### Borzoi

Ported weights (with permission) are uploaded to <a href="https://huggingface.co/johahi"> Huggingface</a>, the model (human or mouse heads) can be loaded with:

```python
from borzoi_pytorch import Borzoi
borzoi = Borzoi.from_pretrained('johahi/borzoi-replicate-0') # 'johahi/borzoi-replicate-[0-3][-mouse]'
````
The Pytorch version produces the same predictions as the original implementation, see for instance in this [notebook](https://github.com/johahi/borzoi-pytorch/blob/main/notebooks/pytorch_borzoi_example.ipynb).  

### Flashzoi

After installation of [FlashAttention-2](https://github.com/Dao-AILab/flash-attention#installation-and-features), Flashzoi offers 3x the speed of Borzoi at comparable or slightly better predictive performance and can be loaded with:
```python
from borzoi_pytorch import Borzoi
borzoi = Borzoi.from_pretrained('johahi/flashzoi-replicate-0') # 'johahi/flashzoi-replicate-[0-3]'
````
Note that this model should/must be run in autocast, and requires a modern Nvidia GPU.

<img width="1288" alt="image" src="https://github.com/user-attachments/assets/bda016b9-1cd5-4377-a771-726f0285613a" />

### Borzoi Prime

Ported weights (with permission) are uploaded to <a href="https://huggingface.co/johahi"> Huggingface</a>, the model (human head) can be loaded with:

```python
from borzoi_pytorch import Prime
borzoi = Prime.from_pretrained('johahi/borzoi-prime-replicate-0') # 'johahi/borzoi-prime-replicate-[0-3]'
````
The Pytorch version produces the same predictions as the original implementation, see for instance in this [notebook](https://github.com/johahi/borzoi-pytorch/blob/main/notebooks/pytorch_prime_example.ipynb).  



## Misc.
The relative shift operation should be [faster](https://johahi.github.io/blog/2024/fast-relative-shift/) than in enformer_pytorch or other implementations. 

## References
Original Borzoi implementation and weights are [here](https://github.com/calico/borzoi).  
<a id="1">[1]</a> 
Linder, Johannes, et al. "Predicting RNA-seq coverage from DNA sequence as a unifying model of gene regulation." Nature Genetics (2025): 1-13; doi: [https://doi.org/10.1101/2023.08.30.555582](https://doi.org/10.1038/s41588-024-02053-6)  
<a id="2">[2]</a> 
Hingerl, Johannes et al. "Flashzoi: an enhanced Borzoi for accelerated genomic analysis." Bioinformatics, Volume 41, Issue 9, September 2025, btaf467; doi: [https://doi.org/10.1093/bioinformatics/btaf467](https://doi.org/10.1093/bioinformatics/btaf467)  
<a id="2">[3]</a> 
Linder, Johannes et al. "Predicting cell type-specific coverage profiles from DNA sequence." biorxiv, 2025.06.10.658961; doi: [https://doi.org/10.1101/2025.06.10.658961](https://doi.org/10.1101/2025.06.10.658961)   
<a id="3">[4]</a>
[enformer-pytorch github](https://github.com/lucidrains/enformer-pytorch/)
Phil Wang

## Citation
Please cite the Borzoi paper [1], along with Flashzoi [2], or Borzoi Prime [3] if you used this repository or the models.
