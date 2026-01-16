"""ProstT5 encoder for amino acid to 3Di structural token conversion."""

from dataclasses import dataclass
from typing import List, Literal, Optional, Iterable, Iterator, Sequence, Tuple
import re
import math
import logging
import time

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

try:
    import torch
    from transformers import AutoModel, T5Tokenizer, AutoModelForSeq2SeqLM
except ImportError:
    torch = None
    AutoModel = None

from ..config import (
    AUTO_DEVICE,
    CPU_DEVICE,
    CUDA_DEVICE,
    DEFAULT_BATCH_SIZE,
    DEFAULT_PROSTT5_MODEL,
    MPS_DEVICE,
)
from ..errors import DeviceError, EncodingError, ModelError
from ..translate.translator import ProteinRecord


@dataclass
class ThreeDiRecord:
    """Represents a 3Di structural encoding of a protein.

    Attributes:
        protein: The ProteinRecord that was encoded
        three_di: The 3Di token sequence
        method: Method used for encoding (always "prostt5_aa2fold")
        model_name: Name of the ProstT5 model used
        inference_device: Device used for inference ("cuda", "mps", or "cpu")
    """

    protein: ProteinRecord
    three_di: str
    method: Literal["prostt5_aa2fold"]
    model_name: str
    inference_device: str

@dataclass(frozen=True)
class IndexedSeq:
    """A sequence paired with its original position in the input list."""
    idx: int
    seq: str

class ProstT5ThreeDiEncoder:
    """Encoder for converting amino acid sequences to 3Di structural tokens.

    Uses the ProstT5 model from HuggingFace to predict 3Di tokens directly
    from protein sequences without requiring 3D structures.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_PROSTT5_MODEL,
        device: Optional[str] = None,
    ):
        """Initialize the ProstT5 encoder.

        Args:
            model_name: HuggingFace model identifier
            device: Device to use ("cuda", "mps", "cpu", or None for auto-detect)

        Raises:
            ModelError: If PyTorch or Transformers are not installed
            DeviceError: If specified device is not available
        """
        if torch is None or AutoModel is None:
            raise ModelError(
                "PyTorch and Transformers are required for 3Di encoding. "
                "Install with: pip install torch transformers"
            )

        self.model_name = model_name
        self.device = self._select_device(device)
        self.model = None
        self.tokenizer = None

    def _select_device(self, device_hint: Optional[str]) -> str:
        """Select the best available device for inference.

        Args:
            device_hint: User-specified device or None for auto-detection

        Returns:
            Device string ("cuda", "mps", or "cpu")

        Raises:
            DeviceError: If specified device is not available
        """
        # If user specified a device other than "auto", use it
        if device_hint and device_hint != AUTO_DEVICE:
            if device_hint == CUDA_DEVICE and not torch.cuda.is_available():
                raise DeviceError("CUDA requested but not available")
            if device_hint == MPS_DEVICE and not (
                hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
            ):
                raise DeviceError("MPS requested but not available")
            return device_hint

        # Auto-detect best device
        if torch.cuda.is_available():
            return CUDA_DEVICE
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return MPS_DEVICE
        return CPU_DEVICE

    def _load_model(self) -> None:
        """Load the ProstT5 model and tokenizer.

        Raises:
            ModelError: If model loading fails
        """
        if self.model is not None:
            return  # Already loaded

        try:
            self.tokenizer = T5Tokenizer.from_pretrained(self.model_name, do_lower_case=False)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name).to(self.device)

            logging.info("Loaded model %s on device %s", self.model_name, self.device)
            logging.info("Model config:\n%s", self.model.config)
        except Exception as e:
            raise ModelError(f"Failed to load ProstT5 model {self.model_name}: {e}") from e

        # only GPUs support half-precision currently; if you want to run on
        # CPU use full-precision (not recommended, much slower)
        _ = self.model.full() if self.device=='cpu' else self.model.half()


    def token_budget_batches(
        self,
        aa_sequences: Sequence[str],
        token_budget: int,
        max_batch_size: int,
    ) -> Iterator[List[IndexedSeq]]:
        """
        Yield batches of sequences (with original indices) under an approximate token budget.

        The approximation assumes per-batch padding to the maximum length in the batch:
            estimated_tokens ~  batch_size * max_len_in_batch

        Strategy:
            1) Keep original indices.
            2) Sort by length to minimize padding within each batch.
            3) Greedily pack adjacent sequences into a batch while staying under token_budget
            and max_batch_size.

        Parameters
        ----------
            aa_sequences : Sequence[str] Unordered amino acid sequences.
            token_budget : int Maximum approximate "tokens" per batch
                               (= batch_size * padded_length).
            max_batch_size : int Maximum number of sequences per batch.

        Yields
        ------
            List[IndexedSeq] A batch of (original_index, sequence) records.
        """

        if token_budget <= 0:
            raise ValueError("token_budget must be > 0")
        if max_batch_size <= 0:
            raise ValueError("max_batch_size must be > 0")

        indexed: List[IndexedSeq] = [IndexedSeq(i, s) for i, s in enumerate(aa_sequences)]
        indexed.sort(key=lambda x: len(x.seq))  # length-sorted for tight padding

        batch: List[IndexedSeq] = []
        batch_max_len = 0

        for item in indexed:
            L = len(item.seq)

            # If a single sequence exceeds the budget, yield it alone
            if L > token_budget:
                if batch:
                    yield batch
                    batch = []
                    batch_max_len = 0
                    yield [item]
                    continue

            new_max_len = max(batch_max_len, L)
            new_size = len(batch) + 1
            est_tokens = new_size * new_max_len

            if batch and (est_tokens > token_budget or new_size > max_batch_size):
                yield batch
                batch = [item]
                batch_max_len = L
            else:
                batch.append(item)
                batch_max_len = new_max_len

        if batch:
            yield batch

    def _encode_batch(self, aa_sequences: List[str]) -> List[str]:
        """
        Encode a smaller batch of sequences - we don't have enough
        memory to encode the whole thing.

        Args:
            aa_sequences: List of amino acid sequences.
                note: Amino acid sequences are expected to be upper-case,
                      while 3Di-sequences need to be lower-case.
        Returns:
            List of 3Di token sequences (one per input sequence)

        Raises:
            EncodingError: If encoding fails
        """

        min_len = min((len(s) for s in aa_sequences), default=0)
        max_len = max((len(s) for s in aa_sequences), default=1000)

        # tokenize sequences and pad up to the longest sequence in the batch
        ids = self.tokenizer.batch_encode_plus(aa_sequences,
                                          add_special_tokens=True,
                                          padding="longest",
                                          return_tensors='pt').to(self.device)

        # Generation configuration for "folding" (AA-->3Di)
        gen_kwargs_aa2fold = {
                          "do_sample": True,
                          "num_beams": 3,
                          "top_p" : 0.95,
                          "temperature" : 1.2,
                          "top_k" : 6,
                          "repetition_penalty" : 1.2,
        }

        # translate from AA to 3Di (AA-->3Di)
        with torch.no_grad():
            translations = self.model.generate(
                ids.input_ids,
                attention_mask=ids.attention_mask,
                max_length=max_len,
                min_length=min_len,
                early_stopping=True, # stop early if end-of-text token is generated
                num_return_sequences=1, # return only a single sequence
                **gen_kwargs_aa2fold
            )
            # Decode and remove white-spaces between tokens
        decoded_translations = self.tokenizer.batch_decode(translations, skip_special_tokens=True)
        # predicted 3Di strings
        structure_sequences = ["".join(ts.split(" ")) for ts in decoded_translations]

        return structure_sequences

    def _fmt_seconds(self, seconds: float) -> str:
        """
        Format seconds as H:MM:SS (or M:SS for < 1 hour).
        """
        seconds = max(0, int(round(seconds)))
        h, rem = divmod(seconds, 3600)
        m, s = divmod(rem, 60)
        return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"

    def encode(
        self,
        aa_sequences: List[str],
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> List[str]:
        """Encode amino acid sequences to 3Di tokens.

        Args:
            aa_sequences: List of amino acid sequences.
                note: Amino acid sequences are expected to be upper-case,
                      while 3Di-sequences need to be lower-case.
            batch_size: Batch size for processing
        Returns:
            List of 3Di token sequences (one per input sequence)

        Raises:
            EncodingError: If encoding fails
        """
        self._load_model()

        # replace all rare/ambiguous amino acids by X (3Di sequences does not
        # have those) and introduce white-space between all sequences (AAs and 3Di)
        aa_sequences = [" ".join(list(re.sub(r"[UZOB]", "X", sequence)))
                             for sequence in aa_sequences]

        # add pre-fixes accordingly.
        # For the translation from AAs to 3Di, you need to prepend "<AA2fold>"
        # and we convert to uppercase to fix tha they are proteins not 3Dis
        aa_sequences = [ "<AA2fold>" + " " + s.upper() for s in aa_sequences]

        three_di_sequences: List[str] = [None] * len(aa_sequences)  # type: ignore[list-item]


        total_batches = math.ceil(len(aa_sequences) / batch_size)
        t0 = time.perf_counter()
        avg_batch_sec: float | None = None

        token_budget = batch_size * 1024  # approx 1k tokens per sequence
        max_batch_size = batch_size * 5  # allow some flexibility in batch size

        try:
            # Process in batches
            for idx, batch in enumerate(
                self.token_budget_batches(aa_sequences, token_budget, max_batch_size),
                start=1
            ):
                logging.info("Preparing batch %d with %d sequences", idx, len(batch))
                batch_seqs = [x.seq for x in batch]
                batch_idxs = [x.idx for x in batch]

                remaining = total_batches - (idx - 1)
                if avg_batch_sec is None:
                    eta_str = "--"
                else:
                    eta_str = self._fmt_seconds(avg_batch_sec * remaining)

                allocated = 0
                reserved = 0
                if torch.cuda.is_available():
                    allocated = torch.cuda.memory_allocated() / 1023**3
                    reserved = torch.cuda.memory_reserved() / 1023**3

                logging.info("3Di encoding batch %d of %d batches. Estimated %s remaining. Cuda memory allocated: %d GB reserved: %d GB",
                             idx,
                             total_batches,
                             eta_str,
                             allocated,
                             reserved
                            )
                batch_start = time.perf_counter()
                batch_results = self._encode_batch(batch_seqs)

                if len(batch_results) != len(batch_seqs):
                    raise ValueError(
                        f"encoder_fn returned {len(enc)} results for a batch of {len(batch_seqs)} sequences"
                    )
                # Reorder results to match original input order
                for bi, br in zip(batch_idxs, batch_results):
                    three_di_sequences[bi] = br

                batch_elapsed = time.perf_counter() - batch_start
                if idx == 1:
                    avg_batch_sec = batch_elapsed
                else:
                    # Running average over completed batches
                    elapsed_total = time.perf_counter() - t0
                    avg_batch_sec = elapsed_total / idx

        except Exception as e:
            raise EncodingError(f"Failed to encode sequences: {e}") from e

        # check all sequences encoded
        missing = [i for i, v in enumerate(three_di_sequences) if v is None]
        if missing:
            raise RuntimeError(f"Missing encodings for {len(missing)} sequences (e.g., indices {missing[:10]})")

        return three_di_sequences

    def encode_proteins(
        self,
        proteins: List[ProteinRecord],
        batch_size: int = DEFAULT_BATCH_SIZE
    ) -> List[ThreeDiRecord]:
        """Encode protein records to 3Di records.

        Args:
            proteins: List of ProteinRecord objects

        Returns:
            List of ThreeDiRecord objects
        """
        # Extract sequences
        aa_sequences = [p.aa_sequence for p in proteins]

        # Encode
        three_di_sequences = self.encode(aa_sequences, batch_size)

        # Create records
        records = []
        for protein, three_di in zip(proteins, three_di_sequences):
            record = ThreeDiRecord(
                protein=protein,
                three_di=three_di,
                method="prostt5_aa2fold",
                model_name=self.model_name,
                inference_device=self.device,
            )
            records.append(record)

        return records
