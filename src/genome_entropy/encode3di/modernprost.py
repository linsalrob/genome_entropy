"""ModernProst encoder for amino acid to 3Di structural token conversion.

This module implements an encoder for gbouras13/modernprost models,
adapted from the phold implementation.

Note: The multitask ModernProst models require transformers >= 5.14.1.
Multi-GPU support uses HuggingFace accelerate library.
"""

import inspect
import sys
import time
from collections.abc import Mapping
from typing import Any, Iterator, List, Optional, Sequence

try:
    import torch
    import torch.nn.functional as F
    from transformers import AutoModel, AutoTokenizer
    from accelerate import Accelerator

    # Check transformers version for ModernBert support
    import transformers

    transformers_version = tuple(map(int, transformers.__version__.split(".")[:2]))
    if transformers_version < (5, 14):
        import warnings

        warnings.warn(
            f"ModernProst models require transformers >= 5.14.1 (current: {transformers.__version__}). "
            "Please upgrade: pip install --upgrade 'transformers>=5.14.1'",
            UserWarning,
        )
except ImportError:
    torch = None  # type: ignore[assignment]
    AutoModel = None  # type: ignore[assignment,misc]
    AutoTokenizer = None  # type: ignore[assignment,misc]
    F = None  # type: ignore[assignment,misc]
    Accelerator = None  # type: ignore[assignment,misc]

from pathlib import Path

from ..config import (
    AUTO_DEVICE,
    CPU_DEVICE,
    CUDA_DEVICE,
    DEFAULT_ENCODING_SIZE,
    MPS_DEVICE,
    THREEDDI_ALPHABET_ORDERED,
    TWELVE_STATE_ALPHABET_ORDERED,
    get_model_capabilities,
    resolve_model_name,
)
from ..errors import DeviceError, ModelError
from ..logging_config import get_logger
from ..translate.translator import ProteinRecord
from .types import IndexedSeq, StructuralEncoding, ThreeDiRecord

logger = get_logger(__name__)


def _enable_remote_code_sibling_imports(config: Any) -> None:
    """Expose the trusted remote-code directory for an absolute sibling import.

    The multitask repository currently imports its configuration module without
    a relative prefix. Transformers caches trusted remote code in a package, so
    that import otherwise cannot resolve even though the file was downloaded.
    """
    module_path = Path(inspect.getfile(config.__class__)).parent
    module_path_string = str(module_path)
    if module_path_string not in sys.path:
        sys.path.append(module_path_string)


def _is_model_cached(model_name: str) -> bool:
    """Check if a model is already cached locally.

    Args:
        model_name: HuggingFace model identifier

    Returns:
        True if model is cached, False otherwise
    """
    try:
        from huggingface_hub import try_to_load_from_cache

        # Try to find the model in the cache
        cache_path = try_to_load_from_cache(
            repo_id=model_name,
            filename="config.json",
        )

        # If we get a path (not None or _CACHED_NO_EXIST), model is cached
        if cache_path is not None and isinstance(cache_path, (str, Path)):
            logger.debug(f"Model {model_name} found in cache at {cache_path}")
            return True

        logger.debug(f"Model {model_name} not found in cache")
        return False
    except Exception as e:
        # If we can't check the cache, assume it's not cached
        logger.debug(f"Could not check cache for {model_name}: {e}")
        return False


class ModernProstThreeDiEncoder:
    """Encoder for converting proteins to structural-state tokens.

    Multitask models predict paired 3Di and 12-state sequences, while deprecated
    models retain the tensor-only 3Di output API.

    Based on implementation from phold:
    https://github.com/gbouras13/phold/blob/main/src/phold/features/predict_3Di.py
    """

    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None,
        use_accelerate: bool = False,
    ):
        """Initialize the ModernProst encoder.

        Args:
            model_name: A supported Hugging Face model identifier.
            device: Device to use ("cuda", "mps", "cpu", or None for auto-detect)
            use_accelerate: If True, use HuggingFace accelerate for multi-GPU support

        Raises:
            ModelError: If PyTorch or Transformers are not installed
            DeviceError: If specified device is not available
        """
        if torch is None or AutoModel is None:
            raise ModelError(
                "PyTorch and Transformers are required for 3Di encoding. "
                "Install with: pip install torch transformers"
            )

        self.model_name = resolve_model_name(model_name)
        self.capabilities = get_model_capabilities(self.model_name, warn=False)
        if self.capabilities.family == "prostt5":
            raise ModelError(f"{self.model_name} is not a ModernProst model")
        if self.capabilities.deprecated:
            logger.warning(
                "The selected model is deprecated and produces only 3Di output. "
                "twelve_state and twelve_state_entropy will be null."
            )
        self.use_accelerate = use_accelerate
        self.accelerator: Any = None

        if use_accelerate:
            if Accelerator is None:
                raise ModelError(
                    "accelerate library is required for multi-GPU support. "
                    "Install with: pip install accelerate"
                )
            # Initialize accelerator for multi-GPU support
            self.accelerator = Accelerator()
            self.device = str(self.accelerator.device)
            logger.info(f"Using accelerate with device: {self.device}")
        else:
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

            # Check if model is already cached
            is_cached = _is_model_cached(self.model_name)
            if is_cached:
                logger.info("Model found in cache, using local files only")
            else:
                logger.info("Model not in cache, will download from HuggingFace")

            if not self.use_accelerate:
                # Disable torch.compile/dynamo globally for multi-GPU compatibility
                # ModernBert uses compiled_mlp which conflicts with multi-threading
                import torch._dynamo

                torch._dynamo.config.suppress_errors = True
                torch._dynamo.reset()

                # Disable compilation by setting environment
                import os

                os.environ["PYTORCH_JIT"] = "0"
                os.environ["TORCH_COMPILE_DISABLE"] = "1"

                logger.info("Disabled torch.compile/dynamo for multi-GPU compatibility")

            # Load tokenizer with trust_remote_code for custom models
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                local_files_only=is_cached,
                force_download=not is_cached,
            )

            # Load model config first
            import transformers

            config = transformers.AutoConfig.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                local_files_only=is_cached,
                force_download=not is_cached,
            )
            if self.capabilities.supports_12st:
                _enable_remote_code_sibling_imports(config)

            # Disable torch.compile if supported (ModernBert has this option)
            if hasattr(config, "reference_compile"):
                config.reference_compile = False
                logger.debug("Set reference_compile=False in model config")

            # Load model with modified config
            if self.use_accelerate:
                # Use accelerate for multi-GPU - don't move to device yet
                self.model = AutoModel.from_pretrained(
                    self.model_name,
                    config=config,
                    trust_remote_code=True,
                    local_files_only=is_cached,
                    force_download=not is_cached,
                )

                # Prepare model with accelerate for distributed inference
                self.model = self.accelerator.prepare(self.model)
                logger.info("Prepared model with accelerate for multi-GPU")
            else:
                # Standard single-GPU/CPU loading
                self.model = AutoModel.from_pretrained(
                    self.model_name,
                    config=config,
                    trust_remote_code=True,
                    local_files_only=is_cached,
                    force_download=not is_cached,
                ).to(self.device)

                # Additional safety: Remove torch.compile from model if it was applied
                self._disable_torch_compile_in_model()

            # ModernProst models use half precision only on CUDA
            # CPU and MPS may not support half precision properly
            if self.device.startswith("cuda") or self.device == CUDA_DEVICE:
                self.model = self.model.half()

            self.model = self.model.eval()

            logger.info(
                "Loaded %s: 3Di=%s, 12-state=%s (device=%s)",
                self.model_name,
                "yes" if self.capabilities.supports_3di else "no",
                "yes" if self.capabilities.supports_12st else "no",
                self.device,
            )
            logger.debug("Model config:\n%s", self.model.config)
        except Exception as e:
            logger.error("Failed to load ModernProst model %s: %s", self.model_name, e)
            raise ModelError(
                f"Failed to load ModernProst model {self.model_name}: {e}"
            ) from e

    def _disable_torch_compile_in_model(self) -> None:
        """Disable torch.compile in the model for multi-GPU compatibility.

        ModernBert models use compiled_mlp which causes issues with multi-threading.
        This method walks through the model and replaces any compiled components
        with their original uncompiled versions.
        """
        try:
            # Walk through all modules in the model
            for name, module in self.model.named_modules():
                # Check if this is a compiled module
                if hasattr(module, "_orig_mod"):
                    # This is a compiled module, replace it with the original
                    logger.debug(f"Removing torch.compile from module: {name}")
                    # Get the parent module and attribute name
                    parent_name = ".".join(name.split(".")[:-1])
                    attr_name = name.split(".")[-1]

                    if parent_name:
                        parent = dict(self.model.named_modules())[parent_name]
                        setattr(parent, attr_name, module._orig_mod)
                    else:
                        # This is a top-level module
                        self.model = module._orig_mod

            logger.info(
                "Removed torch.compile optimizations for multi-GPU compatibility"
            )
        except Exception as e:
            # If removal fails, just log a warning - the config approach might have worked
            logger.debug(f"Could not remove torch.compile (this may be okay): {e}")

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

    def _decode_head(
        self,
        logits: Any,
        attention_mask: Any,
        lengths: Sequence[int],
        alphabet: str,
        expected_classes: int,
        head_name: str,
    ) -> List[str]:
        """Validate and decode a residue-level classifier head."""
        if getattr(logits, "ndim", None) != 3:
            raise ModelError(
                f"{self.model_name} {head_name} logits must have shape B x L x "
                f"{expected_classes}; got {getattr(logits, 'shape', None)}"
            )
        if logits.shape[-1] != expected_classes:
            raise ModelError(
                f"{self.model_name} {head_name} head has {logits.shape[-1]} classes; "
                f"expected {expected_classes}"
            )
        if logits.shape[:2] != attention_mask.shape:
            raise ModelError(
                f"{self.model_name} {head_name} logits shape {tuple(logits.shape[:2])} "
                f"does not match attention mask {tuple(attention_mask.shape)}"
            )
        if logits.shape[0] != len(lengths):
            raise ModelError(
                f"{self.model_name} returned {logits.shape[0]} {head_name} rows for "
                f"{len(lengths)} proteins"
            )

        predictions = logits.argmax(dim=-1).detach().cpu()
        masks = attention_mask.detach().cpu().bool()
        decoded = []
        for row, mask, expected_length in zip(predictions, masks, lengths):
            class_ids = row[mask].tolist()
            if len(class_ids) != expected_length:
                raise ModelError(
                    f"{self.model_name} tokenizer produced {len(class_ids)} residue "
                    f"positions for a protein of length {expected_length}"
                )
            decoded.append("".join(alphabet[int(index)] for index in class_ids))
        return decoded

    def _encode_batch(self, aa_sequences: List[str]) -> List[StructuralEncoding]:
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
            outputs = self.model(**token_encoding)

        logits = outputs.logits
        if self.capabilities.supports_12st:
            if not isinstance(logits, Mapping):
                raise ModelError(
                    f"{self.model_name} is declared dual-head but outputs.logits "
                    "is not a mapping"
                )
            missing = {"3di", "12st"} - set(logits)
            if missing:
                raise ModelError(
                    f"{self.model_name} output is missing required head(s): "
                    f"{', '.join(sorted(missing))}"
                )
            logits_3di = logits["3di"]
            logits_12st = logits["12st"]
        else:
            if isinstance(logits, Mapping):
                raise ModelError(
                    f"Legacy model {self.model_name} unexpectedly returned mapped logits"
                )
            logits_3di = logits
            logits_12st = None

        three_di = self._decode_head(
            logits_3di,
            token_encoding.attention_mask,
            seq_lens,
            THREEDDI_ALPHABET_ORDERED,
            20,
            "3di",
        )
        twelve_state: List[str | None] = []
        if logits_12st is not None:
            decoded_twelve_state = self._decode_head(
                logits_12st,
                token_encoding.attention_mask,
                seq_lens,
                TWELVE_STATE_ALPHABET_ORDERED,
                12,
                "12st",
            )
            twelve_state.extend(decoded_twelve_state)
        else:
            twelve_state = [None] * len(three_di)

        return [
            StructuralEncoding(three_di=three, twelve_state=twelve)
            for three, twelve in zip(three_di, twelve_state)
        ]

    def encode(
        self,
        aa_sequences: List[str],
        encoding_size: int = DEFAULT_ENCODING_SIZE,
        use_multi_gpu: bool = False,
        gpu_ids: Optional[List[int]] = None,
        multi_gpu_encoder: Optional[Any] = None,
    ) -> List[StructuralEncoding]:
        """Encode amino acid sequences to 3Di tokens.

        Args:
            aa_sequences: List of amino acid sequences (upper-case).
            encoding_size: Maximum size (approx. amino acids) to encode per batch
            use_multi_gpu: If True, use accelerate for multi-GPU parallel encoding
            gpu_ids: Optional list of GPU IDs (currently unused with accelerate)
            multi_gpu_encoder: Optional pre-initialized encoder (for backward compatibility)

        Returns:
            List of 3Di token sequences (one per input sequence)

        Raises:
            EncodingError: If encoding fails
        """
        if use_multi_gpu:
            # Use accelerate for multi-GPU encoding
            if not self.use_accelerate:
                # Need to re-initialize with accelerate support
                logger.info(
                    "Re-initializing encoder with accelerate for multi-GPU support"
                )
                self.__init__(
                    model_name=self.model_name,
                    device=None,
                    use_accelerate=True,
                )

            self._load_model()

            # Preprocess sequences (ModernProst-specific, no ProstT5 prefix)
            processed_seqs = []
            for seq in aa_sequences:
                # Replace rare/ambiguous amino acids
                seq = (
                    seq.replace("U", "X")
                    .replace("Z", "X")
                    .replace("O", "X")
                    .replace("B", "X")
                )
                processed_seqs.append(seq)

            # Process in batches using accelerate
            import math

            total_sequences = len(processed_seqs)
            total_batches = math.ceil(sum(map(len, processed_seqs)) / encoding_size)

            # Create batches
            batches = list(self.token_budget_batches(processed_seqs, encoding_size))

            # Process all batches with accelerate
            encodings: List[StructuralEncoding | None] = [None] * total_sequences

            from .encoding import format_seconds, get_memory_info

            t0 = time.perf_counter()
            avg_batch_sec: float | None = None

            for idx, batch in enumerate(batches, start=1):
                batch_seqs = [x.seq for x in batch]
                batch_idxs = [x.idx for x in batch]

                # Calculate ETA
                remaining = total_batches - (idx - 1)
                eta_str = (
                    "--"
                    if avg_batch_sec is None
                    else format_seconds(avg_batch_sec * remaining)
                )

                # Get memory info
                allocated, reserved = get_memory_info()

                logger.info(
                    "3Di encoding batch %d of %d batches. "
                    "Estimated %s remaining. Cuda memory allocated: %.1f GB reserved: %.1f GB",
                    idx,
                    total_batches,
                    eta_str,
                    allocated,
                    reserved,
                )

                batch_start = time.perf_counter()
                batch_results = self._encode_batch(batch_seqs)

                # Store results in original order
                for bi, br in zip(batch_idxs, batch_results):
                    encodings[bi] = br

                # Update timing
                batch_elapsed = time.perf_counter() - batch_start
                if idx == 1:
                    avg_batch_sec = batch_elapsed
                else:
                    elapsed_total = time.perf_counter() - t0
                    avg_batch_sec = elapsed_total / idx

            if any(value is None for value in encodings):
                raise ModelError("ModernProst failed to encode every input sequence")
            return [value for value in encodings if value is not None]
        else:
            # Use single-GPU encoding
            self._load_model()

            # ModernProst does not use ProstT5 preprocessing
            # Just replace non-standard amino acids
            processed_seqs = []
            for seq in aa_sequences:
                # Replace rare/ambiguous amino acids
                seq = (
                    seq.replace("U", "X")
                    .replace("Z", "X")
                    .replace("O", "X")
                    .replace("B", "X")
                )
                processed_seqs.append(seq)

            # Calculate batch info
            import math

            total_sequences = len(processed_seqs)
            total_batches = math.ceil(sum(map(len, processed_seqs)) / encoding_size)

            # Create batches iterator
            batch_iterator = self.token_budget_batches(processed_seqs, encoding_size)

            # Process all batches
            from .encoding import process_batches

            return process_batches(
                batch_iterator,
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
        structural_encodings = self.encode(
            aa_sequences,
            encoding_size,
            use_multi_gpu=use_multi_gpu,
            gpu_ids=gpu_ids,
            multi_gpu_encoder=multi_gpu_encoder,
        )

        # Create records
        records = []
        method = self.capabilities.family
        for protein, structural in zip(proteins, structural_encodings):
            expected = len(protein.aa_sequence)
            observed_3di = len(structural.three_di)
            observed_12st = (
                None
                if structural.twelve_state is None
                else len(structural.twelve_state)
            )
            if observed_3di != expected or (
                self.capabilities.supports_12st and observed_12st != expected
            ):
                raise ModelError(
                    f"Structural encoding length mismatch for {protein.orf.orf_id}: "
                    f"expected={expected}, three_di={observed_3di}, "
                    f"twelve_state={observed_12st}, model={self.model_name}"
                )
            record = ThreeDiRecord(
                protein=protein,
                three_di=structural.three_di,
                method=method,
                model_name=self.model_name,
                inference_device=self.device,
                twelve_state=structural.twelve_state,
            )
            records.append(record)

        return records
