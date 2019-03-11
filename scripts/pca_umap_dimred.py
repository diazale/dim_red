#!/usr/bin/env python
# coding: utf-8
import sys
import os
import allel
import numpy as np
from umap import UMAP
from sklearn.decomposition import PCA 
from bokeh.plotting import figure, output_file, show
from bokeh.palettes import Reds,Greys,Greens,Blues,Purples
from collections import defaultdict

old_stdout = sys.stdout
log_file = open("dimred_messages.log", "a")
sys.stdout = log_file

# # Reading data and metadata, Preprocessing
parent_dir = os.path.dirname
vcf_data = '../data/ALL.wgs.nhgri_coriell_affy_6.20140825.genotypes_has_ped.vcf.gz'
sample_pop = '../data/affy_samples.20141118.panel'
pop_metadata = '../data/20131219.populations.tsv'

def mask_dictionary(d, bool_mask):
    for key in d:
        if(d[key].shape[0] == bool_mask.shape[0]):
            d[key] = np.compress(bool_mask, d[key], axis=0)

def get_sampling_mask(chrom_array,frac = 0.1, stratified = False):
    if stratified:
        strat_mask = []
        chrom, chrom_freq = np.unique(chrom_array, return_counts=True)   
        cf_map = dict(zip(chrom.astype(np.int), chrom_freq.astype(np.int)))
        for chrom_no in sorted(cf_map.keys()):
            strat_mask = np.append(strat_mask, np.random.binomial(1, frac, size = cf_map[chrom_no]))
        return strat_mask
    return np.random.binomial(1, frac, size = chrom_array.shape[0])

callset = allel.read_vcf(vcf_data, fields='*', alt_number = 1, types={'calldata/GT': 'i1'},log = log_file)

sample_to_pop = defaultdict(str)
pop_to_samples = defaultdict(list)
for line in open(sample_pop):
    sample, pop = line.split('\t')[:2]
    if sample == 'sample':
        continue
    sample_to_pop[sample] = pop
    pop_to_samples[pop].append(sample)

pops_of_superpop=defaultdict(list)
pop_desc = defaultdict(str)
for line in open(pop_metadata):
    desc, pop, superpop = line.split('\t')[:3]
    if desc == 'Population Description' or desc =='':
        continue
    pops_of_superpop[superpop].append(pop)
    pop_desc[pop] = desc

superpops = ['EAS','SAS','AFR','EUR','AMR']

indices_for_pop = defaultdict(list)
for index, sample in enumerate(callset['samples']):
    indices_for_pop[sample_to_pop[sample]].append(index) #get positions for particular pop, used in plotting

print('Positions of a given population in samples: \n',indices_for_pop)

# The populations within a superpopulation have the same color but different shade
color_dict = dict(zip(pops_of_superpop['EAS'], Reds[9][:5]))
color_dict.update(dict(zip(pops_of_superpop['SAS'], Greys[9][:5])))
color_dict.update(dict(zip(pops_of_superpop['AFR'], Greens[9][:7])))
color_dict.update(dict(zip(pops_of_superpop['EUR'], Blues[9][:5])))
color_dict.update(dict(zip(pops_of_superpop['AMR'], Purples[9][:4])))                  

non_autosomes = ['X','Y','MT']
chromo_mask = np.logical_not(np.isin(callset['variants/CHROM'], non_autosomes))
mask_dictionary(callset, chromo_mask)
sampling_mask = get_sampling_mask(callset['variants/CHROM'],stratified=True)

mask_dictionary(callset, sampling_mask)
genotype_array = np.sum(callset['calldata/GT'],axis=2)
transposed_genotype_matrix = np.array(genotype_array).transpose()
print('Features randomly sampled from all variants (CHROM,POS): \n', list(zip(callset['variants/CHROM'], callset['variants/POS']))) 
# # Dimension Reduction and Visualization
pca_instace = PCA()
pca_cord = pca_instace.fit_transform(transposed_genotype_matrix)
umap_cord = UMAP(n_components=2, min_dist=0.6).fit_transform(transposed_genotype_matrix)
pca_components = 20
pca_umap_cord = UMAP(n_components=2, min_dist=0.6).fit_transform(pca_cord[:,:pca_components])

marker_map = {'EAS' : 'circle','SAS':'triangle','AFR':'cross','EUR':'square','AMR':'asterisk'}
d1,d2 = 0,1 #dimensions to pick

#Only PCA
p = figure(plot_width = 1366, plot_height=768)
for superpop in superpops: 
    for pop in pops_of_superpop[superpop]:
        p.scatter(pca_cord[indices_for_pop[pop], d1], pca_cord[indices_for_pop[pop], d2], 
                 legend=pop_desc[pop], color = color_dict[pop], marker = marker_map[superpop])
output_file("../plots/pca_t2.html", title="PCA top 2 components")
show(p)

#Only UMAP(A non-linear dim red)
p = figure(plot_width = 1366, plot_height=768)
for superpop in superpops: 
    for pop in pops_of_superpop[superpop]:
        p.scatter(umap_cord[indices_for_pop[pop], d1], umap_cord[indices_for_pop[pop], d2], 
                 legend=pop_desc[pop], color = color_dict[pop], marker = marker_map[superpop])
output_file("../plots/only_umap_t2.html", title="Only UMAP")
show(p)

# 10 PC followed by UMAP
p = figure(plot_width = 1366, plot_height=768)
for superpop in superpops: 
    for pop in pops_of_superpop[superpop]:
        p.scatter(pca_umap_cord[indices_for_pop[pop], d1], pca_umap_cord[indices_for_pop[pop], d2], 
                 legend=pop_desc[pop], color = color_dict[pop],marker = marker_map[superpop])
output_file("../plots/pca_umap_t2.html", title="PCA-UMAP top 2 components")
show(p)
