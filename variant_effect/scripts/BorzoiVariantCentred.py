import os
import yaml
import argparse

import pandas as pd
import torch
import tqdm
import anndata as ad

from torch.utils.data import DataLoader, Dataset

from enformer_pytorch.data import GenomeIntervalDataset, str_to_one_hot

from borzoi_pytorch import AnnotatedBorzoi, Transcriptome
from borzoi_pytorch.config_borzoi import BorzoiConfig

parser = argparse.ArgumentParser()
parser.add_argument('-in_path', type=str, required="True")
parser.add_argument('-out_path', type=str, required="True")
parser.add_argument('-config_path', type=str, required="True")
args = parser.parse_args()

in_path = args.in_path
out_path = args.out_path
config_path = args.config_path

# load and chunk
# parse out indices
out_dir = "/"+os.path.join(*out_path.split("/")[:-1])
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
table = pd.read_table(in_path)
bed_cols_req = ['Chromosome','Start','End','Ref','Alt','gene_name','variant']
bed_cols = [x for x in bed_cols_req if x in table.columns]
assert all(x == y for x,y in zip(bed_cols[:5],bed_cols_req[:5]))
table = table[bed_cols]
bed_file = os.path.join(out_dir,f'bed.tsv')
table.to_csv(bed_file,sep="\t",index=None,header=None)
print(bed_file)

# load config
with open(config_path, "r") as f:
    config = yaml.safe_load(f)
# get config params
bs = config['bs']
gtf_file = config['gtf_file']
fasta_file = config['fasta_file']
pretrained_path = config['pretrained_path']
return_center_bins_only = config['return_center_bins_only']
prediction_bin_mode = config['prediction_bin_mode']
track_subset = config['track_subset']
squashed_scale_mode = config['squashed_scale_mode']
metric = config['metric']
disable_autocast = config.get('disable_autocast', False)
eps = 1e-6

print(pretrained_path)
print(config_path)

cfg = BorzoiConfig.from_pretrained(pretrained_path)
if return_center_bins_only:
    bins = 6144
    offset = 163840
else:
    bins = 16384 - 32
    offset = 512
    cfg.return_center_bins_only = True #False


if prediction_bin_mode == 'all':
    use_transcriptome = False
elif isinstance(prediction_bin_mode, int):
    use_transcriptome = False
    bins = prediction_bin_mode
else:
    use_transcriptome = True

def normed_max_abs_diff_metric(ref_preds,alt_preds,eps):
    ref_preds /= ref_preds.sum(axis=-1,keepdims=True) + eps
    alt_preds /= alt_preds.sum(axis=-1,keepdims=True) + eps
    pred = (ref_preds-alt_preds).abs().amax(axis=-1)
    return pred
    
def logL2_metric(ref_preds,alt_preds,eps):
    ref_preds_log = torch.log2(ref_preds+1)
    alt_preds_log = torch.log2(alt_preds+1)
    altref_log_diff = alt_preds_log - ref_preds_log
    altref_log_diff2 = torch.pow(altref_log_diff, 2)
    log_d2 = torch.sqrt(altref_log_diff2.sum(axis=-1))
    return log_d2

def logfc_metric(ref_preds,alt_preds,eps):
    ref_preds_log = torch.log2(ref_preds.sum(axis = -1) + 1)
    alt_preds_log = torch.log2(alt_preds.sum(axis = -1) + 1)
    logfc = alt_preds_log - ref_preds_log
    return logfc

    
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
        position, bins * 32, 32, span=span, 
    )  # select right columns
    strand = transcriptome.genes[gene].strand
    return gene_slice, strand

class VariantCentredDataset(Dataset):

    def __init__(self, bed_file, gtf_file, fasta_file,
                 use_transcriptome = use_transcriptome,
                context_length = 524288, bins = bins):
        self.gene_region_ds = GenomeIntervalDataset(
                                bed_file = bed_file,
                                fasta_file = fasta_file,
                                filter_df_fn = lambda x: x,
                                return_seq_indices = False,
                                shift_augs = (0,0),
                                rc_aug = False,
                                return_augs = True,
                                context_length = context_length,
                                chr_bed_to_fasta_map = {},
                                #has_header=True,
                            )
        # column order ['Chromosome','Start','End','Ref','Alt','gene_name',...]
        # gene_name is only  when transcriptome is used
        self.use_transcriptome = use_transcriptome
        if self.use_transcriptome:
            self.transcriptome = Transcriptome(gtf_file)
        self.context_length = context_length
        self.bins = bins

    def __len__(self):
        return len(self.gene_region_ds.df) * 2

    def __getitem__(self,idx):
        allele = 'column_4' if idx % 2 == 0 else 'column_5' # column_4 is Ref, column_5 is Alt
        rec = self.gene_region_ds.df[idx // 2]
        # variant pos is start in bed file, 
        # interval has len 1 so gets expanded by (context_len - 1)//2 to left
        # Here (context_len - 1)//2 = (context_len)//2 - 1
        # so variant ends up in context_length//2 - 1
        # start in genome thus is start - context_length//2 + 1
        pos = self.context_length//2 - 1
        pred_start = rec['column_2'].item() - ((self.bins * 32) // 2) + 1
        # get sequence
        seq = self.gene_region_ds[idx // 2][0]
        nuc = str_to_one_hot(rec[allele].item()).squeeze()
        varlen = len(rec[allele].item())
        if allele == 'column_4':
            print (rec, nuc, seq[pos:pos+varlen], seq[pos-2:pos+varlen+2])
            assert torch.allclose(seq[pos:pos+varlen], nuc), str(idx) + ":" + str(seq[pos:pos+varlen])
        # if alt, compute offset, insert variant
        else:
            seq[pos:pos+varlen] = nuc 
        if self.use_transcriptome:
            gene = rec['column_6'].item()
            # get gene slice and strand
            gene_slice, gene_strand = get_gene_slice_and_strand(self.transcriptome, 
                                                                gene, 
                                                                pred_start,
                                                                span=True)
            assert gene_slice.shape[0] > 0, gene # we do not want an empty span
            if gene_strand == '-':
                seq = seq.flip(dims=(0,1))
                gene_slice = (self.bins - 1) - gene_slice
            return seq, gene_slice
        else:
            gene_slice = torch.tensor([x for x in range(self.bins)])
            return seq, gene_slice
            

# Make model
device = 'cuda'
borzoi = AnnotatedBorzoi.from_pretrained(pretrained_path, config=cfg)

# subset bins for centre
if isinstance(prediction_bin_mode, int):
    borzoi.crop.target_length = prediction_bin_mode

# subset head
if track_subset == "RNA":
    target_tracks = borzoi.tracks_df.loc[borzoi.tracks_df.description.str.startswith('RNA')].index
    borzoi.set_track_subset(torch.tensor(target_tracks))
elif track_subset == 'DNASE':
    target_tracks = borzoi.tracks_df.loc[borzoi.tracks_df.description.str.startswith('DNASE')].index
    borzoi.set_track_subset(torch.tensor(target_tracks))
elif track_subset == 'DNASE&ATAC':
    target_tracks = borzoi.tracks_df.loc[borzoi.tracks_df.description.str.startswith('DNASE') | borzoi.tracks_df.description.str.startswith('ATAC')].index
    borzoi.set_track_subset(torch.tensor(target_tracks))

print(borzoi)
borzoi.eval()
borzoi.to(device)

var_ds = VariantCentredDataset(bed_file, gtf_file, fasta_file)
var_dl = DataLoader(var_ds,shuffle=False, batch_size=bs, num_workers=1, pin_memory=True)

# Run
preds = []

if squashed_scale_mode == 'keep':
    print('Predicting with squashed scale')
    pred_fn = lambda x,y: borzoi.predict_gene_count(x,y, bin_level_transform=lambda z: z, log1p=False, agg_fn=lambda k:k)
elif squashed_scale_mode == 'keep_track_scales':
    print('Do not unscale tracks, but remove squashed scale')
    transform_fn = lambda z: borzoi._undo_squashed_scale(z, unscale = False)
    pred_fn = lambda x,y: borzoi.predict_gene_count(x,y, bin_level_transform=transform_fn, agg_fn=lambda k:k)
elif squashed_scale_mode == 'undo':
    print('Removing squashed scale')
    pred_fn = lambda x,y: borzoi.predict_gene_count(x,y, log1p=False, agg_fn=lambda k:k)
else:
    pred_fn = lambda x,y: borzoi.predict_gene_count(x,y)

print(f'Metric used: {metric}')
if metric == 'normed_max_abs_diff':
    metric_fn = normed_max_abs_diff_metric
elif metric == 'logL2':
    metric_fn = logL2_metric
elif metric == 'logfc':
    metric_fn = logfc_metric
else:
    metric_fn = lambda ref,alt, # eps: alt - ref

with torch.inference_mode(), torch.autocast(device, enabled=not disable_autocast):
        for batch in tqdm.tqdm(var_dl, miniters=10):
            seq, bins = batch
            seq = seq.to(device)
            seq = seq.permute(0,2,1)
            # predict sense and antisense and average
            pred = pred_fn(seq, bins)
            ref, alt = pred.unbind(dim=0)
            pred = metric_fn(ref,alt,eps)
            # return 
            preds.append(pred.unsqueeze(0).cpu())

preds = torch.concat(preds)

obs_frame = var_ds.gene_region_ds.df.to_pandas()
obs_frame.columns = bed_cols

adata = ad.AnnData(preds.numpy(), 
                   obs=obs_frame,
                   var=borzoi.output_tracks_df.copy(),
                  )
adata.write(out_path, compression="gzip")