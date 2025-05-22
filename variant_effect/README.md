# Variant effect prediction scripts - WIP
We provide variant effect prediction scripts similar to the ones used in Borzoi.  
You'll need anndata installed.
## Gene-specific scores
Setup/change `configs/gene_specific_config.yaml` and prepare your input tsv to match the one in `data_examples`.
Run `python BorzoiVariant.py --in_path data_examples/gene_specific.tsv --out_dir outdir --config_path configs/gene_specific_config.yaml`.
## Gene-agnostic scores
Setup/change `configs/gene_agnostic_config.yaml` and prepare your input tsv to match the one in `data_examples`.
Run `python BorzoiVariantCentred.py --in_path data_examples/gene_agnostic.tsv --out_dir outdir --config_path configs/gene_agnostic_config.yaml`.

