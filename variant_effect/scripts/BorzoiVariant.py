import yaml
import argparse

import pandas as pd
import anndata as ad
import torch
import tqdm
from torch.utils.data import DataLoader, Dataset

from enformer_pytorch.data import GenomeIntervalDataset, str_to_one_hot
from borzoi_pytorch import AnnotatedBorzoi, Transcriptome
from borzoi_pytorch.config_borzoi import BorzoiConfig

parser = argparse.ArgumentParser()
parser.add_argument('-in_path', type=str, required="True")
parser.add_argument('-out_dir', type=str, required="True")
parser.add_argument('-config_path', type=str, required="True")
args = parser.parse_args()

in_path = args.in_path
out_dir = args.out_dir
config_path = args.config_path

# load and chunk
print("Load data")
names = list(pd.read_csv(in_path, nrows=2, sep="\t").columns)
dataset = pd.read_csv(in_path, names=names, sep="\t")
print("Loaded data")
print(dataset.iloc[0])

# load config
with open(config_path, "r") as f:
    config = yaml.safe_load(f)
# get config params
data_path = config['data_path']
gtf_file = config['gtf_file']
fasta_file = config['fasta_file']
bed_file = config['bed_file']
bs = config['bs']
span = config['span']
track_subset = config['track_subset']
squashed_scale_mode = config['squashed_scale_mode']
pretrained_path = config['pretrained_path']
return_center_bins_only = config['return_center_bins_only']
disable_autocast = config.get('disable_autocast', False)
insert_ref_allele = config.get('insert_ref_allele', False)

print(pretrained_path)
print(config_path)

cfg = BorzoiConfig.from_pretrained(pretrained_path)
if return_center_bins_only:
    bins = 6144 # how many bins are predicted
    offset = 163840 # how much after sequence start does the prediction start
else:
    bins = 16384 - 32 # how many bins are predicted
    offset = 512 # how much after sequence start does the prediction start
    cfg.return_center_bins_only = False

def get_gene_slice_and_strand(transcriptome, gene, position, span, bins=bins):
    """
    Retrieves the gene slice and strand information from the transcriptome.

    Args:
        transcriptome: The transcriptome object.
        gene (str): The name of the gene.
        position (int): The genomic position.
        span (int): The span of the genomic region.
        sliced (bool, optional): Whether to slice the output. Defaults to True.

    Returns:
        Tuple[torch.Tensor, str]: The gene slice and strand.
    """
    gene_slice = transcriptome.genes[gene].output_slice(
        position, bins * 32, 32, span=span, old_version = True
    )  # select right columns
    strand = transcriptome.genes[gene].strand
    return gene_slice, strand

def geneslice_collate_fn(batch):
    fixed_tensors = []
    variable_tensors = []

    for fixed_tensor, variable_tensor in batch:
        fixed_tensors.append(fixed_tensor)
        variable_tensors.append(torch.from_numpy(variable_tensor))

    # Stack the fixed-size tensors
    stacked_fixed_tensors = torch.stack(fixed_tensors)
    
    return stacked_fixed_tensors, variable_tensors

# VariantDataSet
class VariantDataset(Dataset):

    def __init__(self, snp_df, gtf_file, fasta_file, bed_file,
                context_length = 524288, span=span, offset=offset, bins=bins,
                insert_ref_allele=insert_ref_allele):
        # load gene regions
        gene_regions = pd.read_table(bed_file,names=['Chromosome','Start','End','gene_name','Strand'])
        # extend gene regions (from 200kb to 500kb)
        gene_regions['Start'] = gene_regions['Start'] - 163840
        gene_regions['End'] = gene_regions['End'] + 163840
        # intersect variants with gene regions (of target gene)
        snp_df = gene_regions.merge(snp_df,on=['Chromosome','gene_name'])
        self.snp_df = snp_df.query('Pos >= Start and Pos < End').reset_index(drop=True)
        # build dict to quickly query the gene regions
        self.gene_to_idx = {g:i for i,g in enumerate(gene_regions['gene_name'])}
        # build genome dataset
        self.gene_region_ds = GenomeIntervalDataset(
            bed_file = bed_file,
            fasta_file = fasta_file,
            filter_df_fn = lambda x: x,
            return_seq_indices = False,
            shift_augs = (0,0),
            rc_aug = False,
            return_augs = True,
            context_length = context_length,
            chr_bed_to_fasta_map = {}
        )
        self.transcriptome = Transcriptome(gtf_file)
        self.span = span
        self.offset = offset
        self.bins = bins
        self.insert_ref_allele = insert_ref_allele
        # find genes which the transcriptome does not have
        # or where the associated region offers no bins
        blacklist = []
        for _,rec in self.snp_df.iterrows():
            gene = rec['gene_name']
            if gene not in self.transcriptome.genes.keys():
                blacklist.append(gene)
                continue
            gene_slice, gene_strand = get_gene_slice_and_strand(self.transcriptome, 
                                                            gene, 
                                                            rec['Start'] + self.offset,
                                                            span=self.span)
            if gene_slice.shape[0] == 0:
                blacklist.append(gene)
        # remove these genes
        self.snp_df = self.snp_df.query('gene_name not in @blacklist')

    def __len__(self):
        return len(self.snp_df) * 2

    def __getitem__(self,idx):
        # get variant
        allele = 'Ref' if idx < len(self.snp_df) else 'Alt'
        rec = self.snp_df.iloc[idx % len(self.snp_df)]
        gene = rec['gene_name']
        strand = rec['Strand']
        pos = rec['Pos'] - rec['Start']# - 1
        # get sequence of associated gene
        gene_idx = self.gene_to_idx[gene]
        seq = self.gene_region_ds[gene_idx][0]
        # if ref, compute offset, check nuc is correct
        nuc = str_to_one_hot(rec[allele]).squeeze()
        varlen = len(rec[allele])
        if allele == 'Ref':
            if self.insert_ref_allele:
                seq[pos:pos+varlen] = nuc
            else:
                assert torch.allclose(seq[pos:pos+varlen], nuc), gene + ":" + str(seq[pos:pos+varlen])
        # if alt, compute offset, insert variant
        else:
            seq[pos:pos+varlen] = nuc
        # get bins
        gene_slice, gene_strand = get_gene_slice_and_strand(self.transcriptome, 
                                                            gene, 
                                                            rec['Start'] + self.offset,
                                                            span=self.span)
        assert strand == gene_strand, gene
        assert gene_slice.shape[0] > 0, gene # we do not want an empty span
        # make sequence sense
        if strand == '-':
            seq = seq.flip(dims=(0,1))
            gene_slice = (self.bins - 1) - gene_slice
        return seq, gene_slice

# Make model
device = 'cuda'
borzoi = AnnotatedBorzoi.from_pretrained(pretrained_path, config=cfg)
# subset heads
if track_subset == 'CAGE':
    target_tracks = borzoi.tracks_df.loc[borzoi.tracks_df.description.str.startswith('CAGE')].index
else:
    target_tracks = borzoi.tracks_df.loc[borzoi.tracks_df.description.str.startswith('RNA')].index
borzoi.set_track_subset(torch.tensor(target_tracks))
print(borzoi)
borzoi.eval()
borzoi.to(device)

# Make dl
var_ds = VariantDataset(dataset, gtf_file, fasta_file, bed_file)
var_dl = DataLoader(var_ds, shuffle=False, batch_size=bs, num_workers=1, pin_memory=True, collate_fn=geneslice_collate_fn)

# Run
preds = []

if squashed_scale_mode == 'keep':
    print('Predicting with squashed scale')
    bin_level_transform=lambda z: z 
    log1p=False
elif squashed_scale_mode == 'keep_track_scales':
    print('Do not unscale tracks, but remove squashed scale')
    bin_level_transform=lambda z: borzoi._undo_squashed_scale(z, unscale = False)
    log1p=True
else:
    bin_level_transform = None
    log1p=True

agg_fn = lambda x: torch.sum(x, dim=-1)

pred_fn = lambda x,y: borzoi.predict_gene_count(x,y, bin_level_transform=bin_level_transform, 
                                                agg_fn=agg_fn, log1p=log1p)

with torch.inference_mode():
    with torch.autocast(device, enabled=not disable_autocast):
        for batch in tqdm.tqdm(var_dl, miniters=10):
            seq, bins = batch
            seq = seq.to(device)
            seq = seq.permute(0,2,1)
            # predict sense and antisense and average
            pred = pred_fn(seq, bins)
            # return 
            preds.append(pred.cpu())

preds = torch.concat(preds)
snp_effects = (preds[len(var_ds.snp_df):] - preds[:len(var_ds.snp_df)])

adata = ad.AnnData(snp_effects.numpy(), 
                   obs=var_ds.snp_df.copy(),
                   var=borzoi.output_tracks_df.copy(),
                  )
adata.write(f"{out_dir}/variants.h5ad" , compression="gzip")
