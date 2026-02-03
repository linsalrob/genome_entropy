"""ProstT5-based encoder for amino acid to 3Di structural token conversion."""

from typing import Any, Iterator, List, Optional, Sequence

try:
    import torch
    from transformers import AutoModelForSeq2SeqLM, T5Tokenizer
except ImportError:
    torch = None  # type: ignore[assignment]
    AutoModelForSeq2SeqLM = None  # type: ignore[assignment,misc]
    T5Tokenizer = None  # type: ignore[assignment,misc]

from ..config import (
    AUTO_DEVICE,
    CPU_DEVICE,
    CUDA_DEVICE,
    DEFAULT_ENCODING_SIZE,
    DEFAULT_PROSTT5_MODEL,
    MPS_DEVICE,
)
from ..errors import DeviceError, ModelError
from ..logging_config import get_logger
from ..translate.translator import ProteinRecord
from .encoding import encode
from .types import IndexedSeq, ThreeDiRecord

logger = get_logger(__name__)


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
        if torch is None or AutoModelForSeq2SeqLM is None:
            raise ModelError(
                "PyTorch and Transformers are required for 3Di encoding. "
                "Install with: pip install torch transformers"
            )

        self.model_name = model_name
        self.device = self._select_device(device)
        self.model: Any = None
        self.tokenizer: Any = None

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
            logger.info("Loading ProstT5 model: %s", self.model_name)
            self.tokenizer = T5Tokenizer.from_pretrained(
                self.model_name, do_lower_case=False
            )
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name).to(
                self.device
            )

            logger.info("Loaded model %s on device %s", self.model_name, self.device)
            logger.debug("Model config:\n%s", self.model.config)
        except Exception as e:
            logger.error("Failed to load ProstT5 model %s: %s", self.model_name, e)
            raise ModelError(
                f"Failed to load ProstT5 model {self.model_name}: {e}"
            ) from e

        # only GPUs support half-precision currently; if you want to run on
        # CPU use full-precision (not recommended, much slower)
        _ = self.model.float() if self.device == "cpu" else self.model.half()

    def token_budget_batches(
        self,
        aa_sequences: Sequence[str],
        token_budget: int,
    ) -> Iterator[List[IndexedSeq]]:
        """
        Yield batches of sequences (with original indices) under an approximate token budget.

        Optimized strategy to address the problem of isolated long sequences:
            1) Keep original indices.
            2) Sort by length to minimize padding within each batch.
            3) For each batch:
               - Start with long sequences from the end (largest first)
               - Add long sequences until adding another would exceed budget
               - Fill remaining budget with short sequences from the beginning
            4) This approach avoids ending up with long proteins that can't be combined,
               resulting in better token budget utilization and fewer iterations.

        Parameters
        ----------
            aa_sequences : Sequence[str] Unordered amino acid sequences.
            token_budget : int Maximum approximate "tokens" per batch

        Yields
        ------
            List[IndexedSeq] A batch of (original_index, sequence) records.
        """

        if token_budget <= 0:
            raise ValueError("token_budget must be > 0")

        indexed: List[IndexedSeq] = [
            IndexedSeq(i, s) for i, s in enumerate(aa_sequences)
        ]
        indexed.sort(key=lambda x: len(x.seq))  # length-sorted for tight padding

        start_idx = 0  # Points to shortest remaining sequence
        end_idx = len(indexed) - 1  # Points to longest remaining sequence

        while start_idx <= end_idx:
            batch: List[IndexedSeq] = []
            batch_max_len = 0

            # Phase 1: Add long sequences from the end
            while end_idx >= start_idx:
                item = indexed[end_idx]
                L = len(item.seq)

                # If a single sequence exceeds the budget, yield it alone
                if L > token_budget:
                    if batch:
                        yield batch
                        batch = []
                        batch_max_len = 0
                    yield [item]
                    end_idx -= 1
                    continue

                new_max_len = max(batch_max_len, L)
                new_size = len(batch) + 1
                est_tokens = new_size * new_max_len

                if est_tokens <= token_budget:
                    # Add this long sequence to the batch
                    batch.append(item)
                    batch_max_len = new_max_len
                    end_idx -= 1
                else:
                    # Can't add more long sequences, move to phase 2
                    break

            # Phase 2: Fill remaining budget with short sequences from the start
            while start_idx <= end_idx:
                item = indexed[start_idx]
                L = len(item.seq)

                new_max_len = max(batch_max_len, L)
                new_size = len(batch) + 1
                est_tokens = new_size * new_max_len

                if est_tokens <= token_budget:
                    # Add this short sequence to fill the gap
                    batch.append(item)
                    batch_max_len = new_max_len
                    start_idx += 1
                else:
                    # Can't fit more sequences, yield this batch
                    break

            if batch:
                yield batch

    def _encode_batch(self, aa_sequences: List[str]) -> List[str]:
        """
        Encode a smaller batch of sequences - we don't have enough
        memory to encode the whole thing.

        Args:
            aa_sequences: List of preprocessed amino acid sequences.
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
        ids = self.tokenizer(
            aa_sequences,
            add_special_tokens=True,
            padding="longest",
            return_tensors="pt",
        ).to(self.device)


        # Generation configuration for "folding" (AA-->3Di)
        gen_kwargs_aa2fold = {
            "do_sample": True,
            "num_beams": 3,
            "top_p": 0.95,
            "temperature": 1.2,
            "top_k": 6,
            "repetition_penalty": 1.2,
        }

        # ProstT5 appends special tokens at the end of each sequence after tokenization.
        # These extra tokens should be masked out during inference to avoid including
        # them in the residue embeddings. The position len(seq) + 1 accounts for the
        # sequence length plus the prefix token added by preprocessing (<AA2fold>).
        for idx, seq in enumerate(aa_sequences):
            mask_position = (len(seq.replace(" ", "")) - 9) + 1 # Note: len(<AA2fold>) == 9 so this accounts for that tag
            try:
                ids.attention_mask[idx, mask_position] = 0
            except Exception as e:
                err_msg = f"Tried to mask at {mask_position} but attention is only {len(ids.attention_mask[idx])}";
                err_msg += f"\nSequences: {seq.replace(" ", "")}\n"
                err_msg += f"\nMask: {ids.attention_mask[idx]}\n"
                logger.exception(err_msg)

        

        # translate from AA to 3Di (AA-->3Di)
        with torch.no_grad():
            translations = self.model.generate(
                ids.input_ids,
                attention_mask=ids.attention_mask,
                max_length=max_len,
                min_length=min_len,
                early_stopping=True,  # stop early if end-of-text token is generated
                num_return_sequences=1,  # return only a single sequence
                **gen_kwargs_aa2fold,
            )
        
        # Decode and remove white-spaces between tokens
        decoded_translations = self.tokenizer.batch_decode(
            translations, skip_special_tokens=True
        )
        # predicted 3Di strings
        structure_sequences = ["".join(ts.split(" ")) for ts in decoded_translations]

        return structure_sequences

    def encode(
        self,
        aa_sequences: List[str],
        encoding_size: int = DEFAULT_ENCODING_SIZE,
        use_multi_gpu: bool = False,
        gpu_ids: Optional[List[int]] = None,
        multi_gpu_encoder: Optional[Any] = None,
    ) -> List[str]:
        """Encode amino acid sequences to 3Di tokens.

        Args:
            aa_sequences: List of amino acid sequences.
                note: Amino acid sequences are expected to be upper-case,
                      while 3Di-sequences need to be lower-case.
            encoding_size: Maximum size (approx. amino acids) to encode per gpu
            use_multi_gpu: If True, use multi-GPU parallel encoding when available
            gpu_ids: Optional list of GPU IDs to use for multi-GPU encoding.
                    If None and use_multi_gpu=True, auto-discover available GPUs.
            multi_gpu_encoder: Optional pre-initialized MultiGPUEncoder instance.
                    If provided, this encoder will be reused instead of creating
                    a new one. This is important for efficiency when processing
                    multiple sequences.
        Returns:
            List of 3Di token sequences (one per input sequence)

        Raises:
            EncodingError: If encoding fails
        """
        if use_multi_gpu:
            # Use multi-GPU encoding
            from .multi_gpu import MultiGPUEncoder

            # Use pre-initialized encoder if provided, otherwise create new one
            if multi_gpu_encoder is None:
                logger.info("Initializing multi-GPU encoding")
                multi_gpu_encoder = MultiGPUEncoder(
                    model_name=self.model_name,
                    encoder_class=ProstT5ThreeDiEncoder,
                    gpu_ids=gpu_ids,
                )
                # Load models if not already loaded
                logger.info("Loading models on all GPUs...")
                for gpu_encoder in multi_gpu_encoder.encoders:
                    gpu_encoder._load_model()

            # Preprocess sequences
            from .encoding import preprocess_sequences

            processed_seqs = preprocess_sequences(aa_sequences)

            # Encode using multi-GPU (models already loaded)
            return multi_gpu_encoder.encode_multi_gpu(
                processed_seqs,
                self.token_budget_batches,
                encoding_size,
                skip_model_loading=True,  # Models already loaded
            )
        else:
            # Use single-GPU encoding (original behavior)
            self._load_model()

            return encode(
                aa_sequences,
                self._encode_batch,
                self.token_budget_batches,
                encoding_size,
            )

    def encode_proteins(
        self,
        proteins: List[ProteinRecord],
        encoding_size: int = DEFAULT_ENCODING_SIZE,
        use_multi_gpu: bool = False,
        gpu_ids: Optional[List[int]] = None,
        multi_gpu_encoder: Optional[Any] = None,
    ) -> List[ThreeDiRecord]:
        """Encode protein records to 3Di records.

        Args:
            proteins: List of ProteinRecord objects
            encoding_size: Maximum size (approx. amino acids) to encode per batch
            use_multi_gpu: If True, use multi-GPU parallel encoding when available
            gpu_ids: Optional list of GPU IDs to use for multi-GPU encoding
            multi_gpu_encoder: Optional pre-initialized MultiGPUEncoder instance.
                    If provided, this encoder will be reused instead of creating
                    a new one. This is important for efficiency when processing
                    multiple sequences.

        Returns:
            List of ThreeDiRecord objects
        """
        # Extract sequences
        aa_sequences = [p.aa_sequence for p in proteins]

        # Encode
        three_di_sequences = self.encode(
            aa_sequences,
            encoding_size,
            use_multi_gpu=use_multi_gpu,
            gpu_ids=gpu_ids,
            multi_gpu_encoder=multi_gpu_encoder,
        )

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
