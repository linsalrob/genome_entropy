#!/usr/bin/env python3
"""
Example script demonstrating the token size estimation functionality.

This script shows how to use the new token estimator module to find
the optimal encoding size for your GPU when encoding proteins to 3Di.
"""

from orf_entropy.encode3di import (
    ProstT5ThreeDiEncoder,
    estimate_token_size,
    generate_random_protein,
    generate_combined_proteins,
)


def example_random_protein_generation():
    """Demonstrate random protein generation."""
    print("=" * 60)
    print("Example 1: Random Protein Generation")
    print("=" * 60)
    
    # Generate a single random protein
    protein = generate_random_protein(100, seed=42)
    print(f"Generated protein of length {len(protein)}")
    print(f"First 50 characters: {protein[:50]}...")
    print()
    
    # Generate multiple proteins that combine to a target length
    proteins = generate_combined_proteins(500, base_length=100, seed=42)
    print(f"Generated {len(proteins)} proteins totaling {sum(len(p) for p in proteins)} AA")
    for i, p in enumerate(proteins, 1):
        print(f"  Protein {i}: {len(p)} AA")
    print()


def example_token_estimation():
    """Demonstrate token size estimation (requires GPU)."""
    print("=" * 60)
    print("Example 2: Token Size Estimation")
    print("=" * 60)
    print("NOTE: This example requires a GPU and will download the ProstT5 model")
    print("      if not already cached. It may take several minutes to run.")
    print()
    
    try:
        import torch
        
        if not torch.cuda.is_available():
            print("⚠ CUDA not available. Skipping estimation example.")
            print("  This example requires a GPU to run.")
            return
        
        # Initialize encoder
        print("Initializing encoder...")
        encoder = ProstT5ThreeDiEncoder()
        print(f"Using device: {encoder.device}")
        print()
        
        # Run estimation with small range for demonstration
        print("Running token size estimation...")
        print("(Using small range 1000-3000 for quick demo)")
        results = estimate_token_size(
            encoder=encoder,
            start_length=1000,
            end_length=3000,
            step=500,
            num_trials=2,
            base_protein_length=100,
        )
        
        print("\nResults:")
        print(f"  Device: {results['device']}")
        print(f"  Max length: {results['max_length']} AA")
        print(f"  Recommended token size: {results['recommended_token_size']} AA")
        print()
        
    except ImportError:
        print("⚠ PyTorch not installed. Skipping estimation example.")
    except Exception as e:
        print(f"⚠ Error during estimation: {e}")


def example_encoder_usage():
    """Demonstrate basic encoder usage with custom encoding size."""
    print("=" * 60)
    print("Example 3: Using Encoder with Custom Token Size")
    print("=" * 60)
    print("After running token estimation, you can use the recommended")
    print("token size when encoding proteins:")
    print()
    print("  encoder = ProstT5ThreeDiEncoder()")
    print("  proteins = ['ACDEFGHIKLMNPQRSTVWY', 'MKTAYIAKQR']")
    print("  results = encoder.encode(proteins, encoding_size=5000)")
    print()
    print("Or via CLI:")
    print("  dna23di encode3di --input proteins.json --output 3di.json \\")
    print("                    --encoding-size 5000")
    print()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Token Size Estimation Examples")
    print("=" * 60)
    print()
    
    # Example 1: Random protein generation (always runs)
    example_random_protein_generation()
    
    # Example 2: Token estimation (requires GPU)
    # Commented out by default to avoid long run times
    # Uncomment to run:
    # example_token_estimation()
    
    # Example 3: Usage documentation
    example_encoder_usage()
    
    print("=" * 60)
    print("Examples complete!")
    print("=" * 60)
    print()
    print("To run token size estimation via CLI:")
    print("  dna23di estimate-tokens --start 3000 --end 10000 --step 1000")
    print()
