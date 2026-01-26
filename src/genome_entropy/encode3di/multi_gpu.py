"""Multi-GPU asynchronous encoding for protein to 3Di conversion."""

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Iterator, List, Optional, Tuple

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]

from ..errors import EncodingError
from ..logging_config import get_logger
from .gpu_utils import discover_available_gpus, select_device_for_gpu, validate_gpu_availability
from .types import IndexedSeq

logger = get_logger(__name__)


class MultiGPUEncoder:
    """Manages multi-GPU encoding of amino acid sequences to 3Di tokens.
    
    This class distributes encoding batches across multiple GPUs using asyncio
    for parallel processing. It handles GPU allocation, load balancing, and
    error recovery.
    """
    
    def __init__(
        self,
        model_name: str,
        encoder_class: type,
        gpu_ids: Optional[List[int]] = None,
    ):
        """Initialize multi-GPU encoder.
        
        Args:
            model_name: HuggingFace model identifier
            encoder_class: Encoder class to instantiate (e.g., ProstT5ThreeDiEncoder)
            gpu_ids: List of GPU IDs to use. If None, auto-discover available GPUs.
                    If empty list or None after discovery, falls back to single GPU.
        """
        self.model_name = model_name
        self.encoder_class = encoder_class
        
        # Discover and validate GPUs
        if gpu_ids is None:
            gpu_ids = discover_available_gpus()
        
        # Validate the GPU IDs
        self.gpu_ids = validate_gpu_availability(gpu_ids) if gpu_ids else []
        
        # Create encoders for each GPU
        self.encoders: List[Any] = []
        self.executors: List[Any] = []
        if self.gpu_ids:
            logger.info("Initializing multi-GPU encoding with %d GPU(s): %s", 
                       len(self.gpu_ids), self.gpu_ids)
            for gpu_id in self.gpu_ids:
                device = select_device_for_gpu(gpu_id)
                encoder = encoder_class(model_name=model_name, device=device)
                self.encoders.append(encoder)
                self.executors.append(ThreadPoolExecutor(max_workers=1))
                logger.info("Created encoder for %s", device)
        else:
            # Fallback to single GPU/CPU encoder
            logger.info("No GPUs available, falling back to single device encoder")
            encoder = encoder_class(model_name=model_name, device=None)
            self.encoders.append(encoder)
    
    @property
    def num_gpus(self) -> int:
        """Number of GPUs being used."""
        return len(self.encoders)
    
    def is_multi_gpu(self) -> bool:
        """Check if using multiple GPUs."""
        return len(self.encoders) > 1
    
    async def encode_batch_async(
        self,
        encoder_idx: int,
        batch: List[IndexedSeq],
    ) -> Tuple[List[int], List[str]]:
        """Encode a single batch on a specific GPU asynchronously.
        
        Args:
            encoder_idx: Index of encoder/GPU to use
            batch: List of IndexedSeq objects to encode
            
        Returns:
            Tuple of (original_indices, encoded_3di_sequences)
        """
        encoder = self.encoders[encoder_idx]
        executor = self.executors[encoder_idx]
        gpu_id = self.gpu_ids[encoder_idx] if self.gpu_ids else None
        
        batch_seqs = [x.seq for x in batch]
        batch_idxs = [x.idx for x in batch]
        
        logger.info(
            "GPU %s: Encoding batch with %d sequences (total len: %d)",
            gpu_id if gpu_id is not None else "default",
            len(batch_seqs),
            sum(len(s) for s in batch_seqs)
        )
        
        # Run encoding in thread pool to avoid blocking asyncio
        loop = asyncio.get_running_loop()
        # Use encoder's _encode_batch method directly
        results = await loop.run_in_executor(
            executor,
            encoder._encode_batch,
            batch_seqs
        )
        
        return batch_idxs, results
    
    async def encode_all_batches_async(
        self,
        batches: List[List[IndexedSeq]],
        total_sequences: int,
    ) -> List[str]:
        """Encode all batches across multiple GPUs asynchronously.
        
        Args:
            batches: List of batches to encode
            total_sequences: Total number of sequences
            
        Returns:
            List of encoded 3Di sequences in original input order
            
        Raises:
            EncodingError: If encoding fails
        """
        three_di_sequences: List[str] = [None] * total_sequences  # type: ignore[list-item]
        
        t0 = time.perf_counter()
        total_batches = len(batches)
        
        logger.info(
            "Starting multi-GPU encoding of %d sequences in %d batches across %d GPU(s)",
            total_sequences, total_batches, len(self.encoders)
        )
        
        # Create a shared queue for all batches
        batch_queue: asyncio.Queue[Tuple[int, List[IndexedSeq]]] = asyncio.Queue()
        
        # Enqueue all batches with their indices
        for batch_idx, batch in enumerate(batches):
            await batch_queue.put((batch_idx, batch))
        
        # Track completed batches and errors
        completed = 0
        completed_lock = asyncio.Lock()
        first_error: Optional[Exception] = None
        
        async def gpu_worker(gpu_idx: int) -> None:
            """Worker coroutine that processes batches for a specific GPU."""
            nonlocal completed, first_error
            
            while True:
                try:
                    # Get next batch from queue (non-blocking check)
                    batch_idx, batch = await asyncio.wait_for(
                        batch_queue.get(), timeout=0.1
                    )
                except asyncio.TimeoutError:
                    # Queue is empty, exit worker
                    break
                
                try:
                    # Encode the batch on this GPU
                    batch_idxs, batch_results = await self.encode_batch_async(
                        gpu_idx, batch
                    )
                    
                    # Store results in original order
                    for bi, br in zip(batch_idxs, batch_results):
                        three_di_sequences[bi] = br
                    
                    # Update progress
                    async with completed_lock:
                        completed += 1
                        elapsed = time.perf_counter() - t0
                        avg_batch_time = elapsed / completed
                        eta_remaining = avg_batch_time * (total_batches - completed)
                        
                        logger.info(
                            "Completed batch %d/%d (%.1f%%) - Elapsed: %.1fs, ETA: %.1fs",
                            completed, total_batches,
                            100.0 * completed / total_batches,
                            elapsed, eta_remaining
                        )
                    
                    # Mark task as done
                    batch_queue.task_done()
                    
                except Exception as e:
                    # Store first error and stop processing
                    if first_error is None:
                        first_error = e
                        logger.error(
                            "GPU %d failed encoding batch %d: %s",
                            gpu_idx, batch_idx, e, exc_info=True
                        )
                    batch_queue.task_done()
                    break
        
        # Create one worker per GPU
        workers = [
            asyncio.create_task(gpu_worker(gpu_idx))
            for gpu_idx in range(len(self.encoders))
        ]
        
        # Wait for all workers to complete
        try:
            await asyncio.gather(*workers, return_exceptions=True)
        except Exception as e:
            logger.error("Multi-GPU encoding failed during worker execution: %s", e, exc_info=True)
            if first_error is None:
                first_error = e
        
        # If any error occurred, raise it
        if first_error is not None:
            raise EncodingError(
                f"Multi-GPU encoding failed: {first_error}"
            ) from first_error
        
        # Check all sequences encoded
        missing = [i for i, v in enumerate(three_di_sequences) if v is None]
        if missing:
            raise RuntimeError(
                f"Missing encodings for {len(missing)} sequences "
                f"(e.g., indices {missing[:10]})"
            )
        
        elapsed_total = time.perf_counter() - t0
        logger.info(
            "Multi-GPU encoding complete! Encoded %d sequences in %.1fs (%.2f seqs/sec)",
            total_sequences, elapsed_total,
            total_sequences / elapsed_total if elapsed_total > 0 else 0
        )
        
        return three_di_sequences
    
    def encode_multi_gpu(
        self,
        aa_sequences: List[str],
        token_budget_batches_fn: Callable[[List[str], int], Iterator[Any]],
        encoding_size: int,
        skip_model_loading: bool = False,
    ) -> List[str]:
        """Encode sequences using multiple GPUs.
        
        This is a synchronous wrapper around the async encoding method.
        
        Args:
            aa_sequences: List of preprocessed amino acid sequences
            token_budget_batches_fn: Function to create batches under token budget
            encoding_size: Maximum size (approx. amino acids) per batch
            skip_model_loading: If True, skip model loading (assumes models already loaded).
                    This is useful when the encoder is being reused across multiple calls.
            
        Returns:
            List of 3Di token sequences (one per input sequence)
        """
        # Create batches
        batches_iter = token_budget_batches_fn(aa_sequences, encoding_size)
        batches = list(batches_iter)
        
        total_sequences = len(aa_sequences)
        
        # Load models for all encoders (unless already loaded)
        if not skip_model_loading:
            logger.info("Loading models on all GPUs...")
            for encoder in self.encoders:
                encoder._load_model()
        
        # Run async encoding
        if self.is_multi_gpu():
            logger.info("Using multi-GPU parallel encoding")
            result = asyncio.run(
                self.encode_all_batches_async(batches, total_sequences)
            )
        else:
            logger.info("Using single-GPU sequential encoding")
            # Fall back to sequential processing for single GPU
            result = self._encode_single_gpu_sequential(batches, total_sequences)
        
        return result
    
    def _encode_single_gpu_sequential(
        self,
        batches: List[List[IndexedSeq]],
        total_sequences: int,
    ) -> List[str]:
        """Fallback to sequential single-GPU encoding.
        
        Args:
            batches: List of batches to encode
            total_sequences: Total number of sequences
            
        Returns:
            List of encoded 3Di sequences in original input order
        """
        three_di_sequences: List[str] = [None] * total_sequences  # type: ignore[list-item]
        encoder = self.encoders[0]
        
        logger.info("Encoding %d sequences in %d batches (single GPU)", 
                   total_sequences, len(batches))
        
        t0 = time.perf_counter()
        for batch_idx, batch in enumerate(batches, 1):
            batch_seqs = [x.seq for x in batch]
            batch_idxs = [x.idx for x in batch]
            
            results = encoder._encode_batch(batch_seqs)
            
            for bi, br in zip(batch_idxs, results):
                three_di_sequences[bi] = br
            
            elapsed = time.perf_counter() - t0
            avg_time = elapsed / batch_idx
            eta = avg_time * (len(batches) - batch_idx)
            
            logger.info(
                "Batch %d/%d complete - Elapsed: %.1fs, ETA: %.1fs",
                batch_idx, len(batches), elapsed, eta
            )
        
        # Check all sequences encoded
        missing = [i for i, v in enumerate(three_di_sequences) if v is None]
        if missing:
            raise RuntimeError(
                f"Missing encodings for {len(missing)} sequences"
            )
        
        return three_di_sequences
