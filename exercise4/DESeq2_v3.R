################################################################################
# Authors:  Markus List, Alexander Dietrich
# Created:  11.10.2017
# Updated:  29.05.2022
# Purpose:  Analyse RNAseq data from overexpressing HepG2 cells (OE vs CT)
################################################################################


### DEPENDENCIES ###

library(DESeq2)
library(tximport)
library(tximportData)
library(biomaRt)

### DESeq2 ###
quant_dir <- "/home/user/Downloads/HepG2/out/" # adjust --> parent directory in which the result folders of kallisto are stored
samples <- list.files(quant_dir)

samples <- data.frame(project = "HepG2", 
                      sample = samples, 
                      condition = c(rep("CT", 3), rep("OE", 3)))


quant_files <- paste0(quant_dir, samples$sample, "/abundance.h5")
names(quant_files) <- samples$sample

dir <- system.file("extdata", package="tximportData")
tx2gene <- read.csv(file.path(dir, "tx2gene.gencode.v27.csv"))

txi <- tximport(quant_files, type = "kallisto", tx2gene = tx2gene)

# create DESeq2 object
dds <- DESeqDataSetFromTximport(txi, colData = samples, design = ~ condition)
dds <- dds[ rowSums(counts(dds)) > 1, ]
dds <- DESeq(dds)


#get results
res <- results(dds, contrast=c("condition", "CT", "OE"))

#lfcShrink
res2 = lfcShrink(dds, contrast=c("condition", "CT", "OE"), type="normal")

#dispersion plot (a)
plotDispEsts(dds)

#highest log-fold change
print(res[res$log2FoldChange == max(res$log2FoldChange),])
# or absolute?
print(res[res$log2FoldChange == max(res$log2FoldChange),])

