#!/usr/bin/env python3
"""Example demonstrating multi-GPU encoding usage.

This example shows how to use the multi-GPU encoding feature to
parallelize 3Di encoding across multiple GPUs when available.
"""

import os
from orf_entropy.encode3di.gpu_utils import discover_available_gpus

def main():
    print("=" * 60)
    print("Multi-GPU Encoding Example")
    print("=" * 60)
    print()
    
    # Demonstrate GPU discovery
    print("GPU Discovery:")
    print("-" * 60)
    print(f"SLURM_GPUS: {os.environ.get('SLURM_GPUS', 'not set')}")
    print(f"SLURM_JOB_GPUS: {os.environ.get('SLURM_JOB_GPUS', 'not set')}")
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
    print()
    
    gpu_ids = discover_available_gpus()
    if gpu_ids:
        print(f"✓ Discovered {len(gpu_ids)} GPU(s): {gpu_ids}")
    else:
        print("✗ No GPUs discovered (will use CPU)")
    
    print()
    print("=" * 60)
    print("CLI Usage Examples")
    print("=" * 60)
    print()
    
    print("1. Run pipeline with multi-GPU (auto-discover GPUs):")
    print("   dna23di run --input input.fasta --output output.json --multi-gpu")
    print()
    
    print("2. Run pipeline with specific GPUs:")
    print("   dna23di run --input input.fasta --output output.json --multi-gpu --gpu-ids 0,1,2")
    print()
    
    print("3. Encode proteins with multi-GPU:")
    print("   dna23di encode3di --input proteins.json --output 3di.json --multi-gpu")
    print()
    
    print("4. Run pipeline in SLURM with allocated GPUs:")
    print("   # GPUs automatically discovered from SLURM_JOB_GPUS")
    print("   srun --gres=gpu:4 dna23di run --input input.fasta --output output.json --multi-gpu")
    print()
    
    print("=" * 60)
    print("Python API Usage")
    print("=" * 60)
    print()
    
    print("from orf_entropy.encode3di.prostt5 import ProstT5ThreeDiEncoder")
    print("from orf_entropy.translate.translator import ProteinRecord")
    print()
    print("# Initialize encoder")
    print("encoder = ProstT5ThreeDiEncoder(model_name='Rostlab/ProstT5_fp16')")
    print()
    print("# Encode with multi-GPU (auto-discover)")
    print("three_dis = encoder.encode_proteins(")
    print("    proteins,")
    print("    use_multi_gpu=True,")
    print(")")
    print()
    print("# Encode with specific GPUs")
    print("three_dis = encoder.encode_proteins(")
    print("    proteins,")
    print("    use_multi_gpu=True,")
    print("    gpu_ids=[0, 1, 2],")
    print(")")
    print()
    
    print("=" * 60)
    print("Environment Variable Priority")
    print("=" * 60)
    print()
    print("1. SLURM_JOB_GPUS (highest priority)")
    print("2. SLURM_GPUS")
    print("3. CUDA_VISIBLE_DEVICES")
    print("4. torch.cuda.device_count() (lowest priority)")
    print()


if __name__ == "__main__":
    main()
