"""Focused tests for ModernProst model capabilities and dual-head decoding."""

import math
import os
import json
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest

from genome_entropy.config import (
    DEFAULT_PROSTT5_MODEL,
    MODEL_REGISTRY,
    MODERNPROST_1B_MODEL,
    MODERNPROST_50M_MODEL,
    MODERNPROST_BASE_DEPRECATED_MODEL,
    MODERNPROST_PROFILES_DEPRECATED_MODEL,
    PROSTT5_FP16_MODEL,
    PROSTT5_MODEL,
    THREEDDI_ALPHABET_ORDERED,
    TWELVE_STATE_ALPHABET,
    TWELVE_STATE_ALPHABET_ORDERED,
    get_model_capabilities,
    resolve_model_name,
)
from genome_entropy.encode3di.types import IndexedSeq, StructuralEncoding
from genome_entropy.encode3di.types import ThreeDiRecord
from genome_entropy.entropy.shannon import shannon_entropy
from genome_entropy.errors import ModelError
from genome_entropy.io.jsonio import read_json, to_json_dict
from genome_entropy.ml.classifier import extract_features
from genome_entropy.orf.types import OrfRecord
from genome_entropy.pipeline.runner import calculate_pipeline_entropy
from genome_entropy.translate.translator import ProteinRecord


class FakeTensor:
    """Small NumPy-backed subset of the tensor API used by the decoder."""

    def __init__(self, values):
        self.values = np.asarray(values)

    @property
    def ndim(self):
        return self.values.ndim

    @property
    def shape(self):
        return self.values.shape

    def argmax(self, dim=-1):
        return FakeTensor(self.values.argmax(axis=dim))

    def detach(self):
        return self

    def cpu(self):
        return self

    def bool(self):
        return FakeTensor(self.values.astype(bool))

    def tolist(self):
        return self.values.tolist()

    def __iter__(self):
        return (FakeTensor(value) for value in self.values)

    def __getitem__(self, key):
        if isinstance(key, FakeTensor):
            key = key.values
        return FakeTensor(self.values[key])


class FakeTorch:
    class cuda:
        @staticmethod
        def is_available():
            return False

    class backends:
        pass

    @staticmethod
    def no_grad():
        return nullcontext()


class FakeBatchEncoding(dict):
    def __init__(self, mask):
        input_ids = FakeTensor(np.zeros_like(mask))
        attention_mask = FakeTensor(mask)
        super().__init__(input_ids=input_ids, attention_mask=attention_mask)
        self.input_ids = input_ids
        self.attention_mask = attention_mask

    def to(self, _device):
        return self


def logits_for_classes(class_ids, classes):
    logits = np.full((*np.asarray(class_ids).shape, classes), -1.0)
    for batch_index, row in enumerate(class_ids):
        for residue_index, class_id in enumerate(row):
            logits[batch_index, residue_index, class_id] = 1.0
    return FakeTensor(logits)


def make_encoder(monkeypatch, model_name, logits):
    from genome_entropy.encode3di import modernprost as module

    monkeypatch.setattr(module, "torch", FakeTorch)
    monkeypatch.setattr(module, "AutoModel", object())
    encoder = module.ModernProstThreeDiEncoder(model_name=model_name, device="cpu")
    encoder.tokenizer = lambda sequences, **kwargs: FakeBatchEncoding(
        [
            [1] * len(sequence) + [0] * (max(map(len, sequences)) - len(sequence))
            for sequence in sequences
        ]
    )

    class Model:
        def __init__(self):
            self.kwargs = None

        def __call__(self, **kwargs):
            self.kwargs = kwargs
            return SimpleNamespace(logits=logits)

    encoder.model = Model()
    return encoder


def test_model_registry_and_default() -> None:
    assert DEFAULT_PROSTT5_MODEL == MODERNPROST_50M_MODEL
    assert MODERNPROST_1B_MODEL == "gbouras13/modernprost-base"
    assert get_model_capabilities(MODERNPROST_1B_MODEL).supports_12st
    for name in (
        MODERNPROST_BASE_DEPRECATED_MODEL,
        MODERNPROST_PROFILES_DEPRECATED_MODEL,
        PROSTT5_MODEL,
        PROSTT5_FP16_MODEL,
    ):
        assert name in MODEL_REGISTRY
        assert MODEL_REGISTRY[name].supports_3di
        assert not MODEL_REGISTRY[name].supports_12st


def test_old_profiles_alias_resolves_with_warning() -> None:
    with pytest.warns(FutureWarning, match="was renamed"):
        assert (
            resolve_model_name("gbouras13/modernprost-profiles")
            == MODERNPROST_PROFILES_DEPRECATED_MODEL
        )


def test_multitask_output_decodes_both_heads_and_padding(monkeypatch) -> None:
    logits = {
        "3di": logits_for_classes([[0, 19, 1], [2, 3, 4]], 20),
        "12st": logits_for_classes([[0, 11, 5], [2, 3, 4]], 12),
    }
    encoder = make_encoder(monkeypatch, MODERNPROST_50M_MODEL, logits)

    result = encoder._encode_batch(["AAA", "GG"])

    assert result == [
        StructuralEncoding("AYC", "ALF"),
        StructuralEncoding("DE", "CD"),
    ]
    assert set(encoder.model.kwargs) == {"input_ids", "attention_mask"}
    assert [len(item.three_di) for item in result] == [3, 2]
    assert [len(item.twelve_state or "") for item in result] == [3, 2]


@pytest.mark.parametrize(
    ("logits", "message"),
    [
        ({"3di": logits_for_classes([[0]], 20)}, "12st"),
        (
            {
                "3di": logits_for_classes([[0]], 19),
                "12st": logits_for_classes([[0]], 12),
            },
            "expected 20",
        ),
        (
            {
                "3di": logits_for_classes([[0]], 20),
                "12st": logits_for_classes([[0]], 11),
            },
            "expected 12",
        ),
    ],
)
def test_multitask_output_validation(monkeypatch, logits, message) -> None:
    encoder = make_encoder(monkeypatch, MODERNPROST_50M_MODEL, logits)
    with pytest.raises(ModelError, match=message):
        encoder._encode_batch(["A"])


def test_multitask_model_rejects_tensor_logits(monkeypatch) -> None:
    encoder = make_encoder(
        monkeypatch,
        MODERNPROST_50M_MODEL,
        logits_for_classes([[0]], 20),
    )
    with pytest.raises(ModelError, match="not a mapping"):
        encoder._encode_batch(["A"])


def test_legacy_modernprost_returns_no_twelve_state(monkeypatch) -> None:
    encoder = make_encoder(
        monkeypatch,
        MODERNPROST_BASE_DEPRECATED_MODEL,
        logits_for_classes([[0, 1]], 20),
    )
    assert encoder._encode_batch(["AA"]) == [StructuralEncoding("AC", None)]


def test_ordered_alphabets_cover_every_class() -> None:
    assert len(THREEDDI_ALPHABET_ORDERED) == 20
    assert len(set(THREEDDI_ALPHABET_ORDERED)) == 20
    assert TWELVE_STATE_ALPHABET_ORDERED == "ABCDEFGHIJKL"
    assert len(TWELVE_STATE_ALPHABET) == 12


def test_twelve_state_entropy_extremes() -> None:
    assert shannon_entropy("AAAA") == 0.0
    assert shannon_entropy(TWELVE_STATE_ALPHABET_ORDERED) == pytest.approx(
        math.log2(12)
    )
    assert shannon_entropy(
        TWELVE_STATE_ALPHABET_ORDERED,
        alphabet=TWELVE_STATE_ALPHABET,
        normalize=True,
    ) == pytest.approx(1.0)


def make_structural_record(twelve_state):
    orf = OrfRecord(
        parent_id="seq",
        orf_id="orf1",
        start=0,
        end=12,
        strand="+",
        frame=0,
        nt_sequence="ATGGCCGCCGCC",
        aa_sequence="MAAA",
        table_id=11,
        has_start_codon=True,
        has_stop_codon=False,
    )
    protein = ProteinRecord(orf=orf, aa_sequence="MAAA", aa_length=4)
    return (
        orf,
        protein,
        ThreeDiRecord(
            protein=protein,
            three_di="ACDE",
            twelve_state=twelve_state,
            method="modernprost_multitask",
            model_name=MODERNPROST_50M_MODEL,
            inference_device="cpu",
        ),
    )


def test_new_and_legacy_records_serialize_stable_twelve_state_fields() -> None:
    _orf, _protein, new_record = make_structural_record("ABCD")
    _orf, _protein, legacy_record = make_structural_record(None)
    assert to_json_dict(new_record)["twelve_state"] == "ABCD"
    assert to_json_dict(legacy_record)["twelve_state"] is None


def test_pipeline_entropy_reports_twelve_state_or_none() -> None:
    orf, protein, new_record = make_structural_record("AABC")
    report = calculate_pipeline_entropy(
        orf.nt_sequence,
        [orf],
        [protein],
        [new_record],
    )
    assert report.twelve_state_entropy == {
        "orf1": pytest.approx(shannon_entropy("AABC"))
    }
    assert report.alphabet_sizes["twelve_state"] == 12

    _orf, _protein, legacy_record = make_structural_record(None)
    legacy_report = calculate_pipeline_entropy(
        orf.nt_sequence,
        [orf],
        [protein],
        [legacy_record],
    )
    assert legacy_report.twelve_state_entropy is None


def test_read_json_backfills_older_twelve_state_fields(tmp_path) -> None:
    path = tmp_path / "old.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "2.0.0",
                "features": {"orf1": {"entropy": {"three_di_entropy": 0.0}}},
            }
        ),
        encoding="utf-8",
    )
    loaded = read_json(path)
    assert loaded["features"]["orf1"]["twelve_state"] is None
    assert loaded["features"]["orf1"]["entropy"]["twelve_state_entropy"] is None


def test_ml_features_include_twelve_state_values() -> None:
    feature = {
        "location": {"start": 0, "end": 12, "strand": "+", "frame": 0},
        "dna": {"length": 12},
        "protein": {"length": 4},
        "three_di": {"length": 4},
        "twelve_state": {"encoding": "ABCD", "length": 4},
        "metadata": {
            "has_start_codon": True,
            "has_stop_codon": False,
            "in_genbank": True,
        },
        "entropy": {
            "dna_entropy": 1.0,
            "protein_entropy": 0.8,
            "three_di_entropy": 2.0,
            "twelve_state_entropy": 1.5,
        },
    }
    matrix, _labels, names, _metadata = extract_features(
        [[{"schema_version": "2.1.0", "features": {"orf1": feature}}]]
    )
    assert matrix[0, names.index("twelve_state_entropy")] == pytest.approx(1.5)
    assert matrix[0, names.index("twelve_state_length")] == pytest.approx(4)


def test_multi_gpu_reordering_keeps_structural_encodings_together() -> None:
    from genome_entropy.encode3di.multi_gpu import MultiGPUEncoder

    expected = {
        "first": StructuralEncoding("AAA", "BBB"),
        "second": StructuralEncoding("CCC", "DDD"),
    }

    class Encoder:
        def _encode_batch(self, sequences):
            return [expected[sequence] for sequence in sequences]

    manager = object.__new__(MultiGPUEncoder)
    manager.encoders = [Encoder()]
    result = manager._encode_single_gpu_sequential(
        [[IndexedSeq(1, "second"), IndexedSeq(0, "first")]],
        2,
    )
    assert result == [expected["first"], expected["second"]]


@pytest.mark.integration
@pytest.mark.skipif(
    os.environ.get("RUN_HUGGINGFACE_INTEGRATION_TESTS") != "1",
    reason="set RUN_HUGGINGFACE_INTEGRATION_TESTS=1 to download the 50M model",
)
def test_modernprost_50m_integration() -> None:
    from genome_entropy.encode3di.modernprost import ModernProstThreeDiEncoder

    sequence = "MLKRSLLFLTVLLLLFSFSSITNEVSASSSFDKGKY"
    encoding = ModernProstThreeDiEncoder(
        MODERNPROST_50M_MODEL,
        device="cpu",
    ).encode(
        [sequence]
    )[0]
    assert len(encoding.three_di) == len(sequence)
    assert encoding.twelve_state is not None
    assert len(encoding.twelve_state) == len(sequence)
    assert math.isfinite(shannon_entropy(encoding.three_di))
    assert math.isfinite(shannon_entropy(encoding.twelve_state))
