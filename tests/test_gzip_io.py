"""Tests for gzip support in all I/O operations."""

import gzip

import pytest

from genome_entropy.io.fasta import read_fasta, write_fasta, read_fasta_iter
from genome_entropy.io.jsonio import read_json, write_json
from genome_entropy.io.genbank import read_genbank, extract_cds_features


@pytest.fixture
def temp_dir(tmp_path):
    """Provide a temporary directory for test files."""
    return tmp_path


@pytest.fixture
def sample_sequences():
    """Sample sequences for FASTA tests."""
    return {
        "seq1": "ATGGCTAGCTAGCTAGCTAGCTGA",
        "seq2": "ATGAAAAAAAAAAAAAAATAA",
        "seq3": "ATGGGGGGGGGGGGGGGGGGGTAG",
    }


@pytest.fixture
def sample_json_data():
    """Sample JSON data for tests."""
    return {
        "test_key": "test_value",
        "nested": {"data": [1, 2, 3]},
        "array": ["a", "b", "c"],
    }


class TestFastaGzipSupport:
    """Test gzip support for FASTA files."""

    def test_write_read_fasta_plain(self, temp_dir, sample_sequences):
        """Test writing and reading plain FASTA files."""
        fasta_path = temp_dir / "test.fasta"

        # Write plain FASTA
        write_fasta(sample_sequences, fasta_path)

        # Read it back
        sequences = read_fasta(fasta_path)

        assert sequences == sample_sequences
        assert fasta_path.exists()

    def test_write_read_fasta_gzipped(self, temp_dir, sample_sequences):
        """Test writing and reading gzipped FASTA files."""
        fasta_gz_path = temp_dir / "test.fasta.gz"

        # Write gzipped FASTA
        write_fasta(sample_sequences, fasta_gz_path)

        # Read it back
        sequences = read_fasta(fasta_gz_path)

        assert sequences == sample_sequences
        assert fasta_gz_path.exists()

        # Verify it's actually gzipped (file should be smaller or have gzip magic bytes)
        with open(fasta_gz_path, "rb") as f:
            magic = f.read(2)
            assert magic == b"\x1f\x8b"  # Gzip magic bytes

    def test_read_fasta_iter_plain(self, temp_dir, sample_sequences):
        """Test reading plain FASTA with iterator."""
        fasta_path = temp_dir / "test.fasta"
        write_fasta(sample_sequences, fasta_path)

        # Read with iterator
        sequences = dict(read_fasta_iter(fasta_path))

        assert sequences == sample_sequences

    def test_read_fasta_iter_gzipped(self, temp_dir, sample_sequences):
        """Test reading gzipped FASTA with iterator."""
        fasta_gz_path = temp_dir / "test.fasta.gz"
        write_fasta(sample_sequences, fasta_gz_path)

        # Read with iterator
        sequences = dict(read_fasta_iter(fasta_gz_path))

        assert sequences == sample_sequences

    def test_read_manually_gzipped_fasta(self, temp_dir, sample_sequences):
        """Test reading a FASTA file that was manually gzipped."""
        fasta_path = temp_dir / "test.fasta"
        fasta_gz_path = temp_dir / "test.fasta.gz"

        # Write plain FASTA first
        write_fasta(sample_sequences, fasta_path)

        # Manually gzip it
        with open(fasta_path, "rb") as f_in:
            with gzip.open(fasta_gz_path, "wb") as f_out:
                f_out.write(f_in.read())

        # Should be able to read the manually gzipped file
        sequences = read_fasta(fasta_gz_path)
        assert sequences == sample_sequences


class TestJsonGzipSupport:
    """Test gzip support for JSON files."""

    def test_write_read_json_plain(self, temp_dir, sample_json_data):
        """Test writing and reading plain JSON files."""
        json_path = temp_dir / "test.json"

        # Write plain JSON
        write_json(sample_json_data, json_path)

        # Read it back
        data = read_json(json_path)

        assert data == sample_json_data
        assert json_path.exists()

    def test_write_read_json_gzipped(self, temp_dir, sample_json_data):
        """Test writing and reading gzipped JSON files."""
        json_gz_path = temp_dir / "test.json.gz"

        # Write gzipped JSON
        write_json(sample_json_data, json_gz_path)

        # Read it back
        data = read_json(json_gz_path)

        assert data == sample_json_data
        assert json_gz_path.exists()

        # Verify it's actually gzipped
        with open(json_gz_path, "rb") as f:
            magic = f.read(2)
            assert magic == b"\x1f\x8b"  # Gzip magic bytes

    def test_read_manually_gzipped_json(self, temp_dir, sample_json_data):
        """Test reading a JSON file that was manually gzipped."""
        json_path = temp_dir / "test.json"
        json_gz_path = temp_dir / "test.json.gz"

        # Write plain JSON first
        write_json(sample_json_data, json_path)

        # Manually gzip it
        with open(json_path, "rb") as f_in:
            with gzip.open(json_gz_path, "wb") as f_out:
                f_out.write(f_in.read())

        # Should be able to read the manually gzipped file
        data = read_json(json_gz_path)
        assert data == sample_json_data


class TestGenbankGzipSupport:
    """Test gzip support for GenBank files."""

    @pytest.fixture
    def sample_genbank_content(self):
        """Sample GenBank file content."""
        return """LOCUS       TEST_SEQ                  45 bp    DNA     linear
DEFINITION  Test sequence for gzip support.
ACCESSION   TEST_SEQ
VERSION     TEST_SEQ.1
FEATURES             Location/Qualifiers
     CDS             1..45
                     /translation="MAAAAAAAAAAAA"
ORIGIN
        1 atggcagcag cagcagcagc agcagcagca gcagcagcag cagca
//
"""

    def test_read_genbank_plain(self, temp_dir, sample_genbank_content):
        """Test reading plain GenBank files."""
        gb_path = temp_dir / "test.gb"
        gb_path.write_text(sample_genbank_content)

        # Read GenBank
        sequences = read_genbank(gb_path)

        assert len(sequences) == 1
        # BioPython uses the VERSION field, which includes ".1"
        assert "TEST_SEQ.1" in sequences
        # Sequence should be uppercase
        assert sequences["TEST_SEQ.1"].startswith("ATGGCAGCAG")

    def test_read_genbank_gzipped(self, temp_dir, sample_genbank_content):
        """Test reading gzipped GenBank files."""
        gb_gz_path = temp_dir / "test.gb.gz"

        # Write gzipped GenBank
        with gzip.open(gb_gz_path, "wt") as f:
            f.write(sample_genbank_content)

        # Read GenBank
        sequences = read_genbank(gb_gz_path)

        assert len(sequences) == 1
        assert "TEST_SEQ.1" in sequences
        assert sequences["TEST_SEQ.1"].startswith("ATGGCAGCAG")

    def test_extract_cds_plain(self, temp_dir, sample_genbank_content):
        """Test extracting CDS from plain GenBank files."""
        gb_path = temp_dir / "test.gb"
        gb_path.write_text(sample_genbank_content)

        # Extract CDS features
        cds_features = extract_cds_features(gb_path)

        assert len(cds_features) == 1
        assert cds_features[0].parent_id == "TEST_SEQ.1"
        assert cds_features[0].protein_sequence == "MAAAAAAAAAAAA"

    def test_extract_cds_gzipped(self, temp_dir, sample_genbank_content):
        """Test extracting CDS from gzipped GenBank files."""
        gb_gz_path = temp_dir / "test.gb.gz"

        # Write gzipped GenBank
        with gzip.open(gb_gz_path, "wt") as f:
            f.write(sample_genbank_content)

        # Extract CDS features
        cds_features = extract_cds_features(gb_gz_path)

        assert len(cds_features) == 1
        assert cds_features[0].parent_id == "TEST_SEQ.1"
        assert cds_features[0].protein_sequence == "MAAAAAAAAAAAA"


class TestMixedGzipUsage:
    """Test mixed usage of gzipped and plain files."""

    def test_mixed_fasta_files(self, temp_dir, sample_sequences):
        """Test handling both plain and gzipped FASTA files."""
        plain_path = temp_dir / "plain.fasta"
        gz_path = temp_dir / "compressed.fasta.gz"

        # Write both formats
        write_fasta(sample_sequences, plain_path)
        write_fasta(sample_sequences, gz_path)

        # Read both
        plain_seqs = read_fasta(plain_path)
        gz_seqs = read_fasta(gz_path)

        # Should get same content
        assert plain_seqs == gz_seqs == sample_sequences

    def test_mixed_json_files(self, temp_dir, sample_json_data):
        """Test handling both plain and gzipped JSON files."""
        plain_path = temp_dir / "plain.json"
        gz_path = temp_dir / "compressed.json.gz"

        # Write both formats
        write_json(sample_json_data, plain_path)
        write_json(sample_json_data, gz_path)

        # Read both
        plain_data = read_json(plain_path)
        gz_data = read_json(gz_path)

        # Should get same content
        assert plain_data == gz_data == sample_json_data


class TestCompressionEfficiency:
    """Test that compression actually reduces file size."""

    def test_fasta_compression_reduces_size(self, temp_dir):
        """Test that gzipped FASTA is smaller than plain."""
        # Create large repetitive sequences (compress well)
        large_sequences = {f"seq{i}": "ATGC" * 1000 for i in range(10)}

        plain_path = temp_dir / "large.fasta"
        gz_path = temp_dir / "large.fasta.gz"

        write_fasta(large_sequences, plain_path)
        write_fasta(large_sequences, gz_path)

        plain_size = plain_path.stat().st_size
        gz_size = gz_path.stat().st_size

        # Gzipped should be significantly smaller
        assert gz_size < plain_size
        # For repetitive data, should be at least 10x smaller
        assert gz_size < plain_size / 10

    def test_json_compression_reduces_size(self, temp_dir):
        """Test that gzipped JSON is smaller than plain."""
        # Create large JSON with repetitive data
        large_data = {
            "features": [
                {"id": f"feat_{i}", "value": "repeated_value", "data": [1, 2, 3] * 100}
                for i in range(100)
            ]
        }

        plain_path = temp_dir / "large.json"
        gz_path = temp_dir / "large.json.gz"

        write_json(large_data, plain_path)
        write_json(large_data, gz_path)

        plain_size = plain_path.stat().st_size
        gz_size = gz_path.stat().st_size

        # Gzipped should be significantly smaller
        assert gz_size < plain_size
        # For repetitive JSON, should be much smaller
        assert gz_size < plain_size / 5
