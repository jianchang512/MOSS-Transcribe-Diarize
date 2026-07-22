from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoProcessor
from transformers.audio_utils import load_audio
from transformers.generation.streamers import BaseStreamer


DEFAULT_PROMPT = (
    "请将音频转写为文本，每一段需以起始时间戳和说话人编号"
    "（[S01]、[S02]、[S03]…）开头，正文为对应的语音内容，"
    "并在段末标注结束时间戳，以清晰标明该段语音范围。"
)
VIDEO_EXTENSIONS = {".mp4", ".m4v", ".mov", ".mkv", ".webm", ".avi", ".flv", ".wmv"}
TokenCallback = Callable[[int], None]
MAX_META_PARAMETER_PREVIEW = 10
AUDIO_PAD_TOKEN = "<|audio_pad|>"


class ProgressStreamer(BaseStreamer):
    """Count generated tokens from ``generate(streamer=...)`` without decoding text."""

    def __init__(self, callback: TokenCallback):
        self.callback = callback
        self.generated_tokens = 0
        self._seen_prompt = False

    def put(self, value):
        token_count = _token_count(value)
        if not self._seen_prompt:
            self._seen_prompt = True
            return
        self.generated_tokens += token_count
        self.callback(self.generated_tokens)

    def end(self):
        return None


def _token_count(value) -> int:
    if hasattr(value, "numel"):
        return int(value.numel())
    if isinstance(value, (list, tuple)):
        return sum(_token_count(item) for item in value)
    return 1


def _is_mistral_regex_conflict(exc: Exception) -> bool:
    message = exc.args[0] if getattr(exc, "args", None) else str(exc)
    is_type_error = isinstance(exc, TypeError)
    has_flag_name = "fix_mistral_regex" in message
    has_duplicate_keyword_message = "multiple values for keyword argument" in message
    return is_type_error and has_flag_name and has_duplicate_keyword_message


def _iter_meta_parameters(model):
    for name, parameter in model.named_parameters():
        if getattr(parameter, "is_meta", False):
            yield name


def ensure_materialized_model(model) -> None:
    """Validate that model parameters are fully materialized (no ``meta`` tensors)."""
    meta_parameters = list(_iter_meta_parameters(model))
    if meta_parameters:
        preview = ", ".join(meta_parameters[:MAX_META_PARAMETER_PREVIEW])
        raise RuntimeError(f"Model contains meta parameters and cannot be moved safely: {preview}")


def load_model_for_inference(
    model_name_or_path: str | Path,
    *,
    device: torch.device,
    dtype: torch.dtype,
    trust_remote_code: bool = True,
):
    """Load a generation model with a safe Transformers 5.3-compatible flow.

    Args:
        model_name_or_path: Hugging Face model id or local model directory.
        device: Target torch device for inference.
        dtype: Explicit dtype used during weight loading.
        trust_remote_code: Whether to allow remote-code model classes.
    """
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
        dtype=dtype,
        low_cpu_mem_usage=False,
    )
    ensure_materialized_model(model)
    return model.to(device).eval()


def load_processor_for_inference(
    model_name_or_path: str | Path,
    *,
    trust_remote_code: bool = True,
):
    """Load processor with fallback for the Transformers 5.3 tokenizer bug.

    Args:
        model_name_or_path: Hugging Face model id or local model directory.
        trust_remote_code: Whether to allow remote-code processor classes.
    """
    try:
        return AutoProcessor.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
        )
    except TypeError as exc:
        if not _is_mistral_regex_conflict(exc):
            raise
        return AutoProcessor.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
            use_fast=False,
        )


def _resolve_processor_audio_token_id(processor) -> int:
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        raise ValueError("Processor is missing tokenizer.")
    audio_token = getattr(processor, "audio_token", None) or getattr(tokenizer, "audio_token", AUDIO_PAD_TOKEN)
    audio_token_id = tokenizer.convert_tokens_to_ids(audio_token)
    if audio_token_id is None:
        raise ValueError(f"Tokenizer is missing required audio token {audio_token!r}.")
    if int(audio_token_id) < 0:
        raise ValueError(f"Tokenizer resolved invalid id {audio_token_id} for required audio token {audio_token!r}.")
    return int(audio_token_id)


def validate_audio_token_alignment(model, processor) -> int:
    processor_audio_token_id = _resolve_processor_audio_token_id(processor)
    expected_audio_token_id = int(model.config.audio_token_id)
    if processor_audio_token_id != expected_audio_token_id:
        raise ValueError(
            "Tokenizer/model audio token mismatch: "
            f"tokenizer resolved {processor_audio_token_id}, model config expects {expected_audio_token_id}. "
            "Please ensure you are using the matching processor/tokenizer for this model revision."
        )
    return processor_audio_token_id


def dtype_from_name(name: str) -> torch.dtype:
    table = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    try:
        return table[name.lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported dtype: {name}") from exc


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return resolved


def _is_likely_video_path(path: str) -> bool:
    return Path(path.split("?", 1)[0]).suffix.lower() in VIDEO_EXTENSIONS


def load_audio_av(audio: str, sampling_rate: int) -> np.ndarray:
    """Decode an audio stream from a media container with PyAV."""
    try:
        import av
    except ImportError as exc:
        raise ImportError("Install `av` to decode audio from video containers.") from exc

    chunks: list[np.ndarray] = []
    with av.open(audio) as container:
        stream = next((stream for stream in container.streams if stream.type == "audio"), None)
        if stream is None:
            raise ValueError(f"No audio stream found in {audio!r}.")

        resampler = av.audio.resampler.AudioResampler(format="s16", layout="mono", rate=sampling_rate)
        for frame in container.decode(stream):
            frames = resampler.resample(frame)
            if frames is None:
                continue
            if not isinstance(frames, list):
                frames = [frames]
            for resampled in frames:
                chunks.append(resampled.to_ndarray().reshape(-1))

        frames = resampler.resample(None)
        if frames is not None:
            if not isinstance(frames, list):
                frames = [frames]
            for resampled in frames:
                chunks.append(resampled.to_ndarray().reshape(-1))

    if not chunks:
        raise ValueError(f"No decodable audio samples found in {audio!r}.")
    return (np.concatenate(chunks).astype(np.float32) / 32768.0).astype(np.float32, copy=False)


def load_audio_item(audio: str | np.ndarray, sampling_rate: int) -> np.ndarray:
    """Load audio with Transformers' loader, using PyAV for media containers."""
    if isinstance(audio, str) and _is_likely_video_path(audio):
        return load_audio_av(audio, sampling_rate=sampling_rate)
    try:
        return load_audio(audio, sampling_rate=sampling_rate)
    except Exception as exc:
        if not isinstance(audio, str):
            raise
        try:
            return load_audio_av(audio, sampling_rate=sampling_rate)
        except Exception as av_exc:
            raise RuntimeError(
                f"Failed to load audio {audio!r} with transformers.audio_utils.load_audio or PyAV."
            ) from av_exc


def process_audio_info(messages: list[dict[str, Any]], sampling_rate: int):
    """Load audio items from chat messages in the same order as the template."""
    audios = []
    for message in messages:
        content = message["content"]
        if isinstance(content, str):
            continue
        for item in content:
            if item.get("type") != "audio":
                continue
            audio = item.get("audio") or item.get("audio_url") or item.get("url") or item.get("path")
            if audio is None:
                raise ValueError("Audio content must include audio, audio_url, url, or path.")
            audios.append(load_audio_item(audio, sampling_rate=sampling_rate))
    return audios


def build_transcription_messages(audio_path: str | Path, prompt: str = DEFAULT_PROMPT) -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": str(audio_path)},
                {"type": "text", "text": prompt.strip() or DEFAULT_PROMPT},
            ],
        }
    ]


def prepare_inputs(processor, messages, *, max_length: int = 131072, device: torch.device | None = None):
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    audios = process_audio_info(messages, sampling_rate=processor.feature_extractor.sampling_rate)
    audio_kwargs = {"device": str(device)} if device is not None and device.type == "cuda" else {}
    return processor(
        text=text,
        audio=audios,
        max_length=max_length,
        audio_kwargs=audio_kwargs,
        return_tensors="pt",
    )


def generate_transcription(
    model,
    processor,
    messages,
    *,
    max_length: int = 131072,
    max_new_tokens: int | None = None,
    do_sample: bool = False,
    temperature: float | None = None,
    top_p: float | None = None,
    top_k: int | None = None,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
    input_callback: Callable[[int], None] | None = None,
    token_callback: TokenCallback | None = None,
) -> dict[str, Any]:
    device = device or next(model.parameters()).device
    dtype = dtype or next(model.parameters()).dtype
    context = (
        torch.amp.autocast("cuda", dtype=dtype)
        if device.type == "cuda" and dtype in (torch.float16, torch.bfloat16)
        else torch.no_grad()
    )
    with context:
        inputs = prepare_inputs(processor, messages, max_length=max_length, device=device).to(device)
    audio_token_id = validate_audio_token_alignment(model, processor)
    if inputs["input_features"].numel() == 0:
        raise ValueError(
            "No audio features were produced from input messages. "
            "Ensure each message includes valid audio content in an audio field (audio/audio_url/url/path)."
        )
    has_audio_placeholder = torch.any(inputs["input_ids"] == audio_token_id, dim=1)
    if not torch.all(has_audio_placeholder).item():
        raise ValueError(
            "Prompt is missing audio placeholder tokens after processing. "
            "Expected at least one audio token in each sample."
        )

    prompt_len = int(inputs["attention_mask"][0].sum().item())
    if input_callback is not None:
        input_callback(prompt_len)
    generation_config = copy.deepcopy(model.generation_config)
    if max_new_tokens is not None:
        generation_config.max_new_tokens = max_new_tokens
    generation_config.do_sample = do_sample
    if do_sample and temperature is not None:
        generation_config.temperature = temperature
    if do_sample and top_p is not None:
        generation_config.top_p = top_p
    if do_sample and top_k is not None:
        generation_config.top_k = top_k
    streamer = ProgressStreamer(token_callback) if token_callback is not None else None
    generate_kwargs = {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "input_features": inputs["input_features"],
        "audio_feature_lengths": inputs["audio_feature_lengths"],
        "audio_chunk_mapping": inputs["audio_chunk_mapping"],
        "generation_config": generation_config,
    }
    if streamer is not None:
        generate_kwargs["streamer"] = streamer

    with torch.inference_mode(), (
        torch.amp.autocast("cuda", dtype=dtype)
        if device.type == "cuda" and dtype in (torch.float16, torch.bfloat16)
        else torch.no_grad()
    ):
        try:
            outputs = model.generate(**generate_kwargs)
        except TypeError as exc:
            if streamer is None or "streamer" not in str(exc):
                raise
            generate_kwargs.pop("streamer", None)
            outputs = model.generate(**generate_kwargs)

    generated_ids = outputs[0][prompt_len:]
    text = processor.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    return {
        "text": text,
        "prompt_len": prompt_len,
        "generated_tokens": int(generated_ids.numel()),
    }
