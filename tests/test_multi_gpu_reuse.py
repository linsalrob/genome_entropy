"""Tests to verify multi-GPU encoder is reused across multiple sequences."""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, call
import pytest


@pytest.fixture
def mock_orf_finder(monkeypatch):
    """Mock ORF finder to return synthetic ORFs without needing genome_entropy binary."""
    from genome_entropy.orf.types import OrfRecord
    
    def mock_find_orfs(sequences, table_id=11, min_nt_length=90):
        """Return mock ORFs for each sequence."""
        orfs = []
        for seq_id, dna_seq in sequences.items():
            # Use a consistent simple ORF
            nt_seq = dna_seq[:90] if len(dna_seq) >= 90 else dna_seq
            # Provide placeholder AA sequence - translation will be mocked separately
            aa_seq = "M" + "A" * (len(nt_seq) // 3 - 1)
            
            orf = OrfRecord(
                parent_id=seq_id,
                orf_id=f"{seq_id}_orf_1",
                start=0,
                end=len(nt_seq),
                strand="+",
                frame=0,
                nt_sequence=nt_seq,
                aa_sequence=aa_seq,
                table_id=table_id,
                has_start_codon=True,
                has_stop_codon=False,
                in_genbank=False,
            )
            orfs.append(orf)
        return orfs
    
    # Patch where find_orfs is used, not where it's defined
    from genome_entropy.pipeline import runner
    monkeypatch.setattr(runner, "find_orfs", mock_find_orfs)


@pytest.fixture
def mock_translator(monkeypatch):
    """Mock translator to avoid translation validation errors."""
    from genome_entropy.translate.translator import ProteinRecord
    
    def mock_translate_orfs(orfs, table_id=11):
        """Return protein records without translation validation."""
        proteins = []
        for orf in orfs:
            protein = ProteinRecord(
                orf=orf,
                aa_sequence=orf.aa_sequence,  # Just use what was provided
                aa_length=len(orf.aa_sequence),
            )
            proteins.append(protein)
        return proteins
    
    # Patch where translate_orfs is used, not where it's defined
    from genome_entropy.pipeline import runner
    monkeypatch.setattr(runner, "translate_orfs", mock_translate_orfs)


def test_multi_gpu_encoder_created_once_for_multiple_sequences(mock_prostt5_encoder, mock_orf_finder, mock_translator):
    """Test that MultiGPUEncoder is instantiated only once when use_multi_gpu=True.
    
    This verifies the fix for the issue where models were being loaded for each
    sequence in a GenBank file instead of once at the start.
    """
    from genome_entropy.pipeline.runner import run_pipeline
    from genome_entropy.encode3di.multi_gpu import MultiGPUEncoder
    
    # Create a test FASTA file with multiple sequences
    fasta_content = """>seq1
ATGGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAG
CTAGCTAGCTAGCTAGCTAGCTGA
>seq2
ATGAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAATAA
>seq3
ATGCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCTAA
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False) as tmp:
        tmp.write(fasta_content)
        tmp_path = tmp.name
    
    try:
        # Track MultiGPUEncoder instantiation
        original_init = MultiGPUEncoder.__init__
        init_call_count = [0]
        
        def mock_init(self, model_name, encoder_class, gpu_ids=None):
            init_call_count[0] += 1
            # Mock the encoder to avoid actual GPU operations
            self.model_name = model_name
            self.encoder_class = encoder_class
            self.gpu_ids = []
            self.encoders = [Mock()]
            self.executors = []
        
        with patch.object(MultiGPUEncoder, '__init__', mock_init):
            # Run pipeline with multi-GPU enabled
            results = run_pipeline(
                input_fasta=tmp_path,
                table_id=11,
                min_aa_len=10,
                compute_entropy=True,
                device="cpu",
                use_multi_gpu=True,
                gpu_ids=[0, 1],  # Simulate 2 GPUs
            )
            
            # Verify we got results for all sequences
            assert len(results) == 3
            
            # CRITICAL: Verify MultiGPUEncoder was instantiated only ONCE
            # (not once per sequence)
            assert init_call_count[0] == 1, (
                f"MultiGPUEncoder should be instantiated once, but was instantiated "
                f"{init_call_count[0]} times for {len(results)} sequences"
            )
    
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def test_multi_gpu_encoder_models_loaded_once(mock_prostt5_encoder, mock_orf_finder, mock_translator):
    """Test that models are loaded only once when multi-GPU encoder is reused.
    
    This verifies that the _load_model method is not called repeatedly for each
    sequence when the encoder is properly reused.
    """
    from genome_entropy.pipeline.runner import run_pipeline
    from genome_entropy.encode3di.multi_gpu import MultiGPUEncoder
    from genome_entropy.encode3di.prostt5 import ProstT5ThreeDiEncoder
    
    # Create a test FASTA file with multiple sequences
    fasta_content = """>seq1
ATGGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAG
CTAGCTAGCTAGCTAGCTAGCTGA
>seq2
ATGAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAATAA
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False) as tmp:
        tmp.write(fasta_content)
        tmp_path = tmp.name
    
    try:
        # Track _load_model calls
        load_model_call_count = [0]
        original_load_model = ProstT5ThreeDiEncoder._load_model
        
        def mock_load_model(self):
            load_model_call_count[0] += 1
            # Don't actually load the model in tests
            self.model = Mock()
            self.tokenizer = Mock()
        
        # Create a mock MultiGPUEncoder that has mock encoders
        def mock_multi_gpu_init(self, model_name, encoder_class, gpu_ids=None):
            self.model_name = model_name
            self.encoder_class = encoder_class
            self.gpu_ids = gpu_ids or [0, 1]
            # Create mock encoders with mocked _load_model
            self.encoders = []
            for gpu_id in self.gpu_ids:
                encoder = Mock()
                encoder._load_model = Mock(side_effect=lambda: load_model_call_count.__setitem__(0, load_model_call_count[0] + 1))
                self.encoders.append(encoder)
            self.executors = []
        
        with patch.object(ProstT5ThreeDiEncoder, '_load_model', mock_load_model):
            with patch.object(MultiGPUEncoder, '__init__', mock_multi_gpu_init):
                # Run pipeline with multi-GPU enabled
                results = run_pipeline(
                    input_fasta=tmp_path,
                    table_id=11,
                    min_aa_len=10,
                    compute_entropy=True,
                    device="cpu",
                    use_multi_gpu=True,
                    gpu_ids=[0, 1],
                )
                
                # Verify we processed both sequences
                assert len(results) == 2
                
                # Verify _load_model was called only once per GPU (2 GPUs = 2 calls total)
                # NOT once per sequence per GPU (which would be 2 sequences * 2 GPUs = 4 calls)
                assert load_model_call_count[0] == 2, (
                    f"_load_model should be called 2 times (once per GPU), "
                    f"but was called {load_model_call_count[0]} times for {len(results)} sequences"
                )
    
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def test_multi_gpu_encoder_reused_genbank(mock_prostt5_encoder, mock_orf_finder, mock_translator):
    """Test that multi-GPU encoder is reused for GenBank files with multiple entries.
    
    This is the specific use case from the issue: processing a GenBank file with
    multiple records should reuse the same multi-GPU encoder and not reload models.
    """
    from genome_entropy.pipeline.runner import run_pipeline
    from genome_entropy.encode3di.multi_gpu import MultiGPUEncoder
    
    # Create a GenBank file with multiple entries
    genbank_content = """LOCUS       SEQ1                    102 bp    DNA     linear   BCT 01-JAN-2024
DEFINITION  Test sequence 1.
ACCESSION   SEQ1
VERSION     SEQ1.1
FEATURES             Location/Qualifiers
     CDS             1..99
                     /translation="MASSSSSSSSSSSSSSSSSSSSSSSSSSSSSS"
ORIGIN
        1 atggctagct agctagctag ctagctagct agctagctag ctagctagct agctagctag
       61 ctagctagct agctagctag ctagctagct agctagctga nn
//
LOCUS       SEQ2                     48 bp    DNA     linear   BCT 01-JAN-2024
DEFINITION  Test sequence 2.
ACCESSION   SEQ2
VERSION     SEQ2.1
FEATURES             Location/Qualifiers
     CDS             1..45
                     /translation="MKKKKKKKKKKKKKK"
ORIGIN
        1 atgaaaaaa aaaaaaaa aaaaaaaa aaaaaaaa aaaaaaaa aaataa
//
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.gb', delete=False) as tmp:
        tmp.write(genbank_content)
        tmp_path = tmp.name
    
    try:
        # Track MultiGPUEncoder instantiation
        original_init = MultiGPUEncoder.__init__
        init_call_count = [0]
        
        def mock_init(self, model_name, encoder_class, gpu_ids=None):
            init_call_count[0] += 1
            # Mock the encoder to avoid actual GPU operations
            self.model_name = model_name
            self.encoder_class = encoder_class
            self.gpu_ids = []
            self.encoders = [Mock()]
            self.executors = []
        
        with patch.object(MultiGPUEncoder, '__init__', mock_init):
            # Run pipeline with GenBank file and multi-GPU enabled
            results = run_pipeline(
                genbank_file=tmp_path,
                table_id=11,
                min_aa_len=10,
                compute_entropy=True,
                use_multi_gpu=True,
                gpu_ids=[0, 1],
            )
            
            # Verify we processed both GenBank entries
            assert len(results) == 2
            
            # CRITICAL: Verify MultiGPUEncoder was instantiated only ONCE for both entries
            assert init_call_count[0] == 1, (
                f"MultiGPUEncoder should be instantiated once for GenBank with "
                f"{len(results)} entries, but was instantiated {init_call_count[0]} times"
            )
    
    finally:
        Path(tmp_path).unlink(missing_ok=True)
