"""Tests for the --max-aa guard against over-long ORFs."""

import re

import pytest
from typer.testing import CliRunner

from genome_entropy.cli.main import app
from genome_entropy.config import DEFAULT_MAX_AA_LENGTH
from genome_entropy.orf.types import OrfRecord
from genome_entropy.pipeline import runner
from genome_entropy.translate.translator import ProteinRecord

LENGTHS_AA = (20, 60)


@pytest.fixture
def fasta(tmp_path):
    path = tmp_path / "in.fasta"
    path.write_text(">seq1\n" + "ATG" * 100 + "\n")
    return path


@pytest.fixture
def two_orfs(monkeypatch):
    """One ORF per length in LENGTHS_AA, translated to exactly that length."""

    def find_orfs(sequences, table_id=11, min_nt_length=90):
        orfs = []
        for seq_id in sequences:
            for i, n in enumerate(LENGTHS_AA, 1):
                orfs.append(
                    OrfRecord(
                        parent_id=seq_id,
                        orf_id=f"{seq_id}_orf{i}",
                        start=1,
                        end=3 * n,
                        strand="+",
                        frame=1,
                        nt_sequence="ATG" * n,
                        aa_sequence="M" * n,
                        table_id=table_id,
                        has_start_codon=True,
                        has_stop_codon=False,
                    )
                )
        return orfs

    def translate_orfs(orfs, table_id=11):
        return [
            ProteinRecord(
                orf=orf, aa_sequence=orf.aa_sequence, aa_length=len(orf.aa_sequence)
            )
            for orf in orfs
        ]

    monkeypatch.setattr(runner, "find_orfs", find_orfs)
    monkeypatch.setattr(runner, "translate_orfs", translate_orfs)


def _run(fasta, **kwargs):
    return runner.run_pipeline(
        input_fasta=fasta,
        model_name="Rostlab/ProstT5_fp16",
        device="cpu",
        **kwargs,
    )


def test_orfs_over_limit_are_skipped_and_counted(
    fasta, two_orfs, mock_prostt5_encoder, caplog
):
    (result,) = _run(fasta, max_aa_len=50)

    assert [o.orf_id for o in result.orfs] == ["seq1_orf1"]
    assert [p.orf.orf_id for p in result.proteins] == ["seq1_orf1"]
    assert [t.protein.orf.orf_id for t in result.three_dis] == ["seq1_orf1"]
    assert set(result.entropy.three_di_entropy) == {"seq1_orf1"}
    assert result.skipped_over_max_aa == 1
    assert "seq1_orf2" in caplog.text and "longest 60 aa" in caplog.text


def test_limit_is_inclusive(fasta, two_orfs, mock_prostt5_encoder):
    (result,) = _run(fasta, max_aa_len=60)
    assert len(result.orfs) == 2
    assert result.skipped_over_max_aa == 0


@pytest.mark.parametrize("disabled", [0, None])
def test_zero_or_none_disables_the_limit(
    fasta, two_orfs, mock_prostt5_encoder, disabled
):
    (result,) = _run(fasta, max_aa_len=disabled)
    assert len(result.orfs) == 2
    assert result.skipped_over_max_aa == 0


def test_every_orf_skipped_still_yields_a_record(
    fasta, two_orfs, mock_prostt5_encoder, monkeypatch
):
    from genome_entropy.encode3di.prostt5 import ProstT5ThreeDiEncoder

    def must_not_encode(self, *args, **kwargs):
        raise AssertionError("encoder called with nothing to encode")

    monkeypatch.setattr(ProstT5ThreeDiEncoder, "encode_proteins", must_not_encode)
    (result,) = _run(fasta, max_aa_len=10)
    assert result.input_id == "seq1"
    assert result.orfs == [] and result.three_dis == []
    assert result.skipped_over_max_aa == 2


def test_default_limit_applies(fasta, two_orfs, mock_prostt5_encoder):
    assert DEFAULT_MAX_AA_LENGTH > max(LENGTHS_AA)
    (result,) = _run(fasta)
    assert len(result.orfs) == 2


def test_negative_limit_is_rejected(fasta, two_orfs, mock_prostt5_encoder):
    with pytest.raises(ValueError, match="max_aa_len"):
        _run(fasta, max_aa_len=-1)


def test_cli_documents_max_aa():
    result = CliRunner().invoke(app, ["run", "--help"])
    assert result.exit_code == 0
    # Remove ANSI codes; rich splits option names with them when colour is on
    clean_output = re.sub(r"\x1b\[[0-9;]*m", "", result.output)
    assert "--max-aa" in clean_output
