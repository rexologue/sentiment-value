from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import torch
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

LOGGER = logging.getLogger(__name__)


@dataclass
class OptimizationInfo:
    device: torch.device
    dtype: torch.dtype
    attn_implementation: str
    compiled: bool


class OptimizedSequenceClassificationModel:
    """Wrapper around AutoModelForSequenceClassification with best-effort optimizations.

    Features:
    - Chooses the best device (CUDA if available).
    - On CUDA prefers bfloat16 if supported, otherwise float16.
    - Tries attention implementations in order:
        flash_attention_2 -> sdpa -> eager -> default.
    - Optionally wraps the model with torch.compile (if available).
    - Prints a summary of what optimizations were successfully enabled.
    """

    def __init__(
        self,
        model_dir: Union[str, Path],
        *,
        prefer_cuda: bool = True,
        enable_compile: bool = True,
    ) -> None:
        self.model_dir = str(model_dir)
        self.device = self._select_best_device(prefer_cuda=prefer_cuda)
        self.dtype = self._select_best_dtype(self.device)

        LOGGER.info("Selected device: %s", self.device)
        LOGGER.info("Selected dtype: %s", self.dtype)

        self.config = AutoConfig.from_pretrained(self.model_dir)
        self.tokenizer = self._load_tokenizer(self.model_dir)

        self.model, attn_impl = self._load_with_best_attention_impl(
            self.model_dir,
            self.config,
            self.device,
            self.dtype,
        )

        self.model.eval()

        compiled = False
        if enable_compile:
            self.model, compiled = self._maybe_compile(self.model)

        self.info = OptimizationInfo(
            device=self.device,
            dtype=self.dtype,
            attn_implementation=attn_impl,
            compiled=compiled,
        )

        self._print_summary()

    @property
    def num_labels(self) -> int:
        return int(self.config.num_labels)

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def logits(
        self,
        texts: Union[str, Sequence[str]],
        *,
        max_length: Optional[int] = None,
        batch_size: int = 32,
        **tokenizer_kwargs: Any,
    ) -> torch.Tensor:
        """Compute logits for one or many texts."""
        if isinstance(texts, str):
            texts = [texts]

        if max_length is None:
            max_length = (
                self.tokenizer.model_max_length
                if self.tokenizer.model_max_length
                and self.tokenizer.model_max_length < 10000
                else 128
            )

        all_logits: List[torch.Tensor] = []
        self.model.eval()

        with torch.no_grad():
            for start in range(0, len(texts), batch_size):
                batch = texts[start : start + batch_size]
                enc = self.tokenizer(
                    list(batch),
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    **tokenizer_kwargs,
                )
                enc = {k: v.to(self.device) for k, v in enc.items()}

                outputs = self.model(**enc)
                batch_logits = outputs.logits.detach().to("cpu")
                all_logits.append(batch_logits)

        return torch.cat(all_logits, dim=0)

    def probabilities(
        self,
        texts: Union[str, Sequence[str]],
        *,
        max_length: Optional[int] = None,
        batch_size: int = 32,
        **tokenizer_kwargs: Any,
    ) -> torch.Tensor:
        """Softmax over logits for one or many texts."""
        logits = self.logits(
            texts,
            max_length=max_length,
            batch_size=batch_size,
            **tokenizer_kwargs,
        )
        return torch.softmax(logits, dim=-1)

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _select_best_device(prefer_cuda: bool = True) -> torch.device:
        if prefer_cuda and torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    @staticmethod
    def _select_best_dtype(device: torch.device) -> torch.dtype:
        if device.type != "cuda":
            return torch.float32

        # On CUDA: prefer bfloat16 if supported, otherwise float16.
        try:
            if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
                return torch.bfloat16
        except Exception:  # pragma: no cover - very defensive
            pass

        return torch.float16

    @staticmethod
    def _load_tokenizer(model_dir: str):
        # Try to fix Mistral-style regex if transformers supports it.
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_dir,
                use_fast=True,
                fix_mistral_regex=True,
            )
            LOGGER.info("Loaded tokenizer with fix_mistral_regex=True")
            return tokenizer
        except TypeError:
            tokenizer = AutoTokenizer.from_pretrained(
                model_dir,
                use_fast=True,
            )
            LOGGER.info("Loaded tokenizer without fix_mistral_regex (not supported)")
            return tokenizer

    @staticmethod
    def _load_with_best_attention_impl(
        model_dir: str,
        config: Any,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.nn.Module, str]:
        """Try several attention implementations, from fastest to safest."""
        # Order matters: flash_attention_2 -> sdpa -> eager -> default
        candidates: List[Optional[str]] = [
            "flash_attention_2",
            "sdpa",
            "eager",
            None,
        ]

        last_exc: Optional[BaseException] = None

        for impl in candidates:
            kwargs: Dict[str, Any] = {}
            if impl is not None:
                kwargs["attn_implementation"] = impl

            try:
                LOGGER.info("Trying to load model with attn_implementation=%r", impl)
                model = AutoModelForSequenceClassification.from_pretrained(
                    model_dir,
                    config=config,
                    torch_dtype=dtype if device.type == "cuda" else torch.float32,
                    **kwargs,
                )
                model.to(device)
                LOGGER.info("Successfully loaded model with attn_implementation=%r", impl)
                return model, impl or "default"
            except TypeError as exc:
                # attn_implementation is unknown for this arch/version
                LOGGER.warning(
                    "attn_implementation=%r not supported by model: %s",
                    impl,
                    exc,
                )
                last_exc = exc
            except Exception as exc:  # pragma: no cover - runtime issues
                LOGGER.warning(
                    "Failed to load model with attn_implementation=%r: %s",
                    impl,
                    exc,
                )
                last_exc = exc

        # If everything failed, surface the last error.
        raise RuntimeError(
            f"Failed to load model from {model_dir} with any attention implementation"
        ) from last_exc

    @staticmethod
    def _maybe_compile(model: torch.nn.Module) -> tuple[torch.nn.Module, bool]:
        if not hasattr(torch, "compile"):
            LOGGER.info("torch.compile is not available in this PyTorch version")
            return model, False

        try:
            LOGGER.info("Trying to compile model with torch.compile(...)")
            compiled = torch.compile(model, mode="max-autotune")
            LOGGER.info("Successfully compiled model with torch.compile")
            return compiled, True
        except Exception as exc:  # pragma: no cover - backend dependent
            LOGGER.warning("torch.compile failed, using original model: %s", exc)
            return model, False

    def _print_summary(self) -> None:
        info = self.info
        device_str = str(info.device)
        dtype_str = str(info.dtype).replace("torch.", "")
        attn_str = info.attn_implementation
        compile_str = "enabled" if info.compiled else "disabled"

        summary = (
            f"Loaded model from {self.model_dir}\n"
            f"  device           : {device_str}\n"
            f"  dtype            : {dtype_str}\n"
            f"  attention impl   : {attn_str}\n"
            f"  torch.compile    : {compile_str}"
        )

        LOGGER.info(summary.replace("\n", " | "))
        print(summary)


class ModelWrapper:
    """High-level model interface used by the FastAPI application.

    This wrapper keeps the model and tokenizer in memory and exposes a simple
    batch `predict` API that returns softmax probabilities on the CPU.
    """

    def __init__(
        self,
        model_path: Union[str, Path],
        *,
        prefer_cuda: bool = True,
        enable_compile: bool = True,
    ) -> None:
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model path does not exist: {model_path}")

        self.model = OptimizedSequenceClassificationModel(
            model_path,
            prefer_cuda=prefer_cuda,
            enable_compile=enable_compile,
        )

    @property
    def info(self) -> OptimizationInfo:
        return self.model.info

    @property
    def device(self) -> torch.device:
        return self.model.info.device

    @property
    def num_labels(self) -> int:
        return self.model.num_labels

    def predict(
        self,
        texts: Sequence[str],
        *,
        batch_size: int = 32,
        max_length: Optional[int] = None,
    ) -> torch.Tensor:
        """Return softmax probabilities for a batch of texts on the CPU."""

        probs = self.model.probabilities(
            list(texts),
            batch_size=batch_size,
            max_length=max_length,
        )

        return probs.to("cpu")


def load_model(
    model_path: Union[str, Path],
    *,
    prefer_cuda: bool = True,
    enable_compile: bool = True,
) -> ModelWrapper:
    """Convenience helper to load the optimized model wrapper."""

    LOGGER.info("Loading model from %s", model_path)
    return ModelWrapper(
        model_path,
        prefer_cuda=prefer_cuda,
        enable_compile=enable_compile,
    )
