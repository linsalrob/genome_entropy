# AMD GPU Slurm Scripts for Genome Entropy Analyses

This repository contains Slurm scripts for running genome entropy analyses on high-performance computing clusters that use the AMD GPUs and has ROCm installed.

## Current scripts, and how to get started with `genome_entropy`

These are the main four scripts that you need to run `genome_entropy`:

1. `install.slurm`: This script loads the rocm module, and then installs the appropriate and required Python packages and dependencies for running the genome entropy pipeline. You should run this first to set up your environment. It should choose the most appropriate rocm/torch version based on your cluster setup.
2. `download.slurm`: Use this script to download the appropriate [ProstT5 models](https://huggingface.co/Rostlab/ProstT5) from huggingface.
3. `estimate_tokens.slurm`: This script estimates the maximum number of amino-acids that your GPU can process into 3Di tokens at once. This is largely dependent on the memory of your GPU and the size of the ProstT5 model you are using (becuase that is also loaded onto the GPU). Its an empirical way of measuring the memory requirements, basically we try and bunch and see what crashes!
4. `pipeline.slurm`: This is the main pipeline script that runs the entire genome entropy analysis, from encoding sequences to calculating entropy scores. If you give this a genbank file or fasta file as input, it will run the full pipeline and output JSON results.

