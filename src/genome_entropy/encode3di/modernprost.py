"""ModernProst encoder for amino acid to 3Di structural token conversion.

This module implements an encoder for gbouras13/modernprost models,
adapted from the phold implementation.

Note: ModernProst models require transformers >= 4.47.0 for ModernBert support.
"""

from typing import Any, Iterator, List, Optional, Sequence

try:
    import torch
    import torch.nn.functional as F
    from transformers import AutoModel, AutoTokenizer
    
    # Check transformers version for ModernBert support
    import transformers
    transformers_version = tuple(map(int, transformers.__version__.split('.')[:2]))
    if transformers_version < (4, 47):
        import warnings
        warnings.warn(
            f"ModernProst models require transformers >= 4.47.0 (current: {transformers.__version__}). "
            "Please upgrade: pip install --upgrade 'transformers>=4.47.0'",
            UserWarning
        )
except ImportError:
    torch = None  # type: ignore[assignment]
    AutoModel = None  # type: ignore[assignment,misc]
    AutoTokenizer = None  # type: ignore[assignment,misc]
    F = None  # type: ignore[assignment,misc]

import numpy as np

from ..config import (
    AUTO_DEVICE,
    CPU_DEVICE,
    CUDA_DEVICE,
    DEFAULT_ENCODING_SIZE,
    MPS_DEVICE,
    MODERNPROST_PROFILES_MODEL,
)
from ..errors import DeviceError, ModelError
from ..logging_config import get_logger
from ..translate.translator import ProteinRecord
from .encoding import encode
from .types import IndexedSeq, ThreeDiRecord

logger = get_logger(__name__)


class ModernProstThreeDiEncoder:
    """Encoder for converting amino acid sequences to 3Di structural tokens.

    Uses ModernProst models (gbouras13/modernprost-base or modernprost-profiles)
    from HuggingFace to predict 3Di tokens directly from protein sequences.
    
    Based on implementation from phold:
    https://github.com/gbouras13/phold/blob/main/src/phold/features/predict_3Di.py
    """

    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None,
    ):
        """Initialize the ModernProst encoder.

        Args:
            model_name: HuggingFace model identifier (gbouras13/modernprost-base or modernprost-profiles)
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
        """Load the ModernProst model and tokenizer.

        Raises:
            ModelError: If model loading fails
        """
        if self.model is not None:
            return  # Already loaded

        try:
            logger.info("Loading ModernProst model: %s", self.model_name)
            
            # Load tokenizer with trust_remote_code for custom models
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
            )
            
            # Load model with trust_remote_code
            self.model = AutoModel.from_pretrained(
                self.model_name,
                trust_remote_code=True,
            ).to(self.device)
            
            # ModernProst models use half precision only on CUDA
            # CPU and MPS may not support half precision properly
            if self.device == CUDA_DEVICE:
                self.model = self.model.half()
            
            self.model = self.model.eval()

            logger.info("Loaded model %s on device %s", self.model_name, self.device)
            logger.debug("Model config:\n%s", self.model.config)
        except Exception as e:
            logger.error("Failed to load ModernProst model %s: %s", self.model_name, e)
            raise ModelError(
                f"Failed to load ModernProst model {self.model_name}: {e}"
            ) from e

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
        Encode a batch of sequences using ModernProst.

        Args:
            aa_sequences: List of amino acid sequences (upper-case).

        Returns:
            List of 3Di token sequences (one per input sequence)

        Raises:
            EncodingError: If encoding fails
        """
        # Replace non-standard amino acids
        processed_seqs = []
        seq_lens = []
        for seq in aa_sequences:
            seq = seq.replace("U", "X").replace("Z", "X").replace("O", "X")
            processed_seqs.append(seq)
            seq_lens.append(len(seq))

        # Tokenize sequences (no special tokens, no prefix for modernprost)
        token_encoding = self.tokenizer(
            processed_seqs,
            padding="longest",
            truncation=False,
            return_tensors="pt",
            add_special_tokens=False,
        ).to(self.device)

        # Run inference
        with torch.no_grad():
            outputs = self.model(
                token_encoding.input_ids,
                attention_mask=token_encoding.attention_mask,
            )

        # Extract logits and compute predictions
        logits = outputs.logits  # [B, L, C]
        
        # Get predictions (argmax over classes)
        preds = torch.argmax(logits, dim=-1)  # [B, L]
        preds_cpu = preds.cpu().numpy()

        # Map predictions to 3Di alphabet
        ss_mapping = {
            0: "A", 1: "C", 2: "D", 3: "E", 4: "F",
            5: "G", 6: "H", 7: "I", 8: "K", 9: "L",
            10: "M", 11: "N", 12: "P", 13: "Q", 14: "R",
            15: "S", 16: "T", 17: "V", 18: "W", 19: "Y",
        }

        # Convert predictions to 3Di strings
        structure_sequences = []
        for batch_idx, s_len in enumerate(seq_lens):
            pred = preds_cpu[batch_idx, :s_len]
            three_di = "".join([ss_mapping[int(p)] for p in pred])
            structure_sequences.append(three_di)

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
            aa_sequences: List of amino acid sequences (upper-case).
            encoding_size: Maximum size (approx. amino acids) to encode per batch
            use_multi_gpu: If True, use multi-GPU parallel encoding when available
            gpu_ids: Optional list of GPU IDs to use for multi-GPU encoding.
            multi_gpu_encoder: Optional pre-initialized MultiGPUEncoder instance.

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
                    encoder_class=ModernProstThreeDiEncoder,
                    gpu_ids=gpu_ids,
                )
                # Load models if not already loaded
                logger.info("Loading models on all GPUs...")
                for gpu_encoder in multi_gpu_encoder.encoders:
                    gpu_encoder._load_model()

            # Preprocess sequences (ModernProst-specific, no ProstT5 prefix)
            processed_seqs = []
            for seq in aa_sequences:
                # Replace rare/ambiguous amino acids
                seq = seq.replace("U", "X").replace("Z", "X").replace("O", "X").replace("B", "X")
                processed_seqs.append(seq)

            # Encode using multi-GPU (models already loaded)
            return multi_gpu_encoder.encode_multi_gpu(
                processed_seqs,
                self.token_budget_batches,
                encoding_size,
                skip_model_loading=True,  # Models already loaded
            )
        else:
            # Use single-GPU encoding
            self._load_model()

            # ModernProst does not use ProstT5 preprocessing
            # Just replace non-standard amino acids
            processed_seqs = []
            for seq in aa_sequences:
                # Replace rare/ambiguous amino acids
                seq = seq.replace("U", "X").replace("Z", "X").replace("O", "X").replace("B", "X")
                processed_seqs.append(seq)

            # Calculate batch info
            import math
            total_sequences = len(processed_seqs)
            total_batches = math.ceil(sum(map(len, processed_seqs)) / encoding_size)

            # Create batches iterator
            batches = self.token_budget_batches(processed_seqs, encoding_size)

            # Process all batches
            from .encoding import process_batches
            return process_batches(
                batches,
                self._encode_batch,
                total_sequences,
                total_batches,
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
        method = "modernprost_profiles" if self.model_name == MODERNPROST_PROFILES_MODEL else "modernprost_base"
        for protein, three_di in zip(proteins, three_di_sequences):
            record = ThreeDiRecord(
                protein=protein,
                three_di=three_di,
                method=method,
                model_name=self.model_name,
                inference_device=self.device,
            )
            records.append(record)

        return records
