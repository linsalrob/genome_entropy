"""Test encode3di command with different input formats (FASTA and JSON)."""

import json
from pathlib import Path

import pytest

# Skip all tests if typer is not installed
pytest.importorskip("typer")

from typer.testing import CliRunner

from genome_entropy.cli.main import app
from genome_entropy.io.fasta import write_fasta
from genome_entropy.io.jsonio import write_json
from genome_entropy.orf.types import OrfRecord
from genome_entropy.translate.translator import ProteinRecord

runner = CliRunner()


@pytest.fixture
def sample_protein_fasta(tmp_path):
    """Create a sample protein FASTA file for testing."""
    fasta_file = tmp_path / "proteins.fasta"
    sequences = {
        "protein1": "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQF",
        "protein2": "MTEITAAMVKELRESTGAGMMDCKNALSETNGDFDKAVQLLREKGLGKAAKKADRLAAEG",
    }
    write_fasta(sequences, fasta_file)
    return fasta_file


@pytest.fixture
def sample_protein_json(tmp_path):
    """Create a sample protein JSON file for testing."""
    json_file = tmp_path / "proteins.json"
    
    # Create ProteinRecord objects with minimal OrfRecord data
    proteins = []
    for i, (seq_id, aa_seq) in enumerate([
        ("protein1", "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQF"),
        ("protein2", "MTEITAAMVKELRESTGAGMMDCKNALSETNGDFDKAVQLLREKGLGKAAKKADRLAAEG"),
    ]):
        orf = OrfRecord(
            parent_id=seq_id,
            orf_id=seq_id,
            start=0,
            end=len(aa_seq) * 3,
            strand="+",
            frame=0,
            nt_sequence="",
            aa_sequence=aa_seq,
            table_id=11,
            has_start_codon=False,
            has_stop_codon=False,
            in_genbank=False,
        )
        protein = ProteinRecord(
            orf=orf,
            aa_sequence=aa_seq,
            aa_length=len(aa_seq),
        )
        proteins.append(protein)
    
    write_json(proteins, json_file)
    return json_file


def test_encode3di_with_fasta_input(tmp_path, sample_protein_fasta, mock_prostt5_encoder):
    """Test encode3di command with protein FASTA file as input."""
    output_file = tmp_path / "output.json"
    
    result = runner.invoke(
        app,
        [
            "encode3di",
            "--input", str(sample_protein_fasta),
            "--output", str(output_file),
            "--model", "Rostlab/ProstT5_fp16",
        ],
    )
    
    # Command should succeed
    assert result.exit_code == 0, f"Command failed: {result.stdout}"
    assert "Detected FASTA format" in result.stdout
    assert "Loaded 2 protein(s)" in result.stdout
    assert "3Di encoding complete" in result.stdout
    
    # Output file should exist
    assert output_file.exists()
    
    # Verify output format
    with open(output_file) as f:
        data = json.load(f)
    
    assert isinstance(data, list)
    assert len(data) == 2
    
    # Verify each record has the expected structure
    for record in data:
        assert "protein" in record
        assert "three_di" in record
        assert "method" in record
        assert "model_name" in record


def test_encode3di_with_json_input(tmp_path, sample_protein_json, mock_prostt5_encoder):
    """Test encode3di command with protein JSON file as input."""
    output_file = tmp_path / "output.json"
    
    result = runner.invoke(
        app,
        [
            "encode3di",
            "--input", str(sample_protein_json),
            "--output", str(output_file),
            "--model", "Rostlab/ProstT5_fp16",
        ],
    )
    
    # Command should succeed
    assert result.exit_code == 0, f"Command failed: {result.stdout}"
    assert "Detected JSON format" in result.stdout
    assert "Loaded 2 protein(s)" in result.stdout
    assert "3Di encoding complete" in result.stdout
    
    # Output file should exist
    assert output_file.exists()
    
    # Verify output format
    with open(output_file) as f:
        data = json.load(f)
    
    assert isinstance(data, list)
    assert len(data) == 2
    
    # Verify each record has the expected structure
    for record in data:
        assert "protein" in record
        assert "three_di" in record
        assert "method" in record
        assert "model_name" in record


def test_encode3di_fasta_and_json_produce_same_output(
    tmp_path, sample_protein_fasta, sample_protein_json, mock_prostt5_encoder
):
    """Test that FASTA and JSON inputs produce equivalent 3Di encodings."""
    output_fasta = tmp_path / "output_fasta.json"
    output_json = tmp_path / "output_json.json"
    
    # Run with FASTA input
    result_fasta = runner.invoke(
        app,
        [
            "encode3di",
            "--input", str(sample_protein_fasta),
            "--output", str(output_fasta),
            "--model", "Rostlab/ProstT5_fp16",
        ],
    )
    assert result_fasta.exit_code == 0
    
    # Run with JSON input
    result_json = runner.invoke(
        app,
        [
            "encode3di",
            "--input", str(sample_protein_json),
            "--output", str(output_json),
            "--model", "Rostlab/ProstT5_fp16",
        ],
    )
    assert result_json.exit_code == 0
    
    # Load both outputs
    with open(output_fasta) as f:
        data_fasta = json.load(f)
    with open(output_json) as f:
        data_json = json.load(f)
    
    # Both should have same number of records
    assert len(data_fasta) == len(data_json)
    
    # Verify 3Di encodings are the same
    for i in range(len(data_fasta)):
        # The 3Di encoding should be identical
        assert data_fasta[i]["three_di"] == data_json[i]["three_di"]
        # The protein sequences should be identical
        assert data_fasta[i]["protein"]["aa_sequence"] == data_json[i]["protein"]["aa_sequence"]


def test_encode3di_with_different_fasta_extensions(tmp_path, mock_prostt5_encoder):
    """Test that encode3di recognizes different FASTA file extensions."""
    extensions = [".fasta", ".fa", ".faa"]
    
    for ext in extensions:
        fasta_file = tmp_path / f"proteins{ext}"
        sequences = {"test_protein": "MKTAYIAKQRQISFVKSHFSRQLE"}
        write_fasta(sequences, fasta_file)
        
        output_file = tmp_path / f"output{ext}.json"
        
        result = runner.invoke(
            app,
            [
                "encode3di",
                "--input", str(fasta_file),
                "--output", str(output_file),
                "--model", "Rostlab/ProstT5_fp16",
            ],
        )
        
        assert result.exit_code == 0, f"Failed for extension {ext}: {result.stdout}"
        assert "Detected FASTA format" in result.stdout
        assert output_file.exists()


def test_encode3di_help_shows_both_formats(tmp_path):
    """Test that encode3di help text mentions both FASTA and JSON formats."""
    result = runner.invoke(app, ["encode3di", "--help"])
    
    assert result.exit_code == 0
    # Help should mention both input formats
    assert "FASTA" in result.stdout or "fasta" in result.stdout.lower()
    assert "JSON" in result.stdout or "json" in result.stdout.lower()
