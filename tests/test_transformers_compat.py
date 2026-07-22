import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.models.whisper.configuration_whisper import WhisperConfig

from moss_transcribe_diarize.inference_utils import (
    generate_transcription,
    load_model_for_inference,
    load_processor_for_inference,
    validate_audio_token_alignment,
)
from moss_transcribe_diarize.configuration_moss_transcribe_diarize import MossTranscribeDiarizeConfig
from moss_transcribe_diarize.modeling_moss_transcribe_diarize import (
    GenerationMixin,
    MossTranscribeDiarizeForConditionalGeneration,
)


class _FakeModel:
    def __init__(self, *, has_meta: bool):
        self._has_meta = has_meta
        self.moved_to = None
        self.eval_called = False

    def named_parameters(self):
        yield "layer.weight", SimpleNamespace(is_meta=self._has_meta)

    def to(self, device):
        self.moved_to = device
        return self

    def eval(self):
        self.eval_called = True
        return self


class TransformersCompatibilityTest(unittest.TestCase):
    def setUp(self):
        self.base_model_inputs = {"input_ids": torch.tensor([[1]])}

    def test_prepare_inputs_keeps_audio_on_first_step_without_cache(self):
        model = object.__new__(MossTranscribeDiarizeForConditionalGeneration)
        input_features = torch.zeros((1, 80, 3000), dtype=torch.float32)
        audio_feature_lengths = torch.tensor([128], dtype=torch.long)
        audio_chunk_mapping = torch.tensor([0], dtype=torch.long)

        with patch.object(GenerationMixin, "prepare_inputs_for_generation", return_value=dict(self.base_model_inputs)):
            model_inputs = model.prepare_inputs_for_generation(
                input_ids=torch.tensor([[1]]),
                past_key_values=None,
                input_features=input_features,
                audio_feature_lengths=audio_feature_lengths,
                audio_chunk_mapping=audio_chunk_mapping,
                use_cache=True,
            )

        self.assertIs(model_inputs["input_features"], input_features)
        self.assertIs(model_inputs["audio_feature_lengths"], audio_feature_lengths)
        self.assertIs(model_inputs["audio_chunk_mapping"], audio_chunk_mapping)

    def test_prepare_inputs_keeps_audio_on_first_step_with_cache_object(self):
        model = object.__new__(MossTranscribeDiarizeForConditionalGeneration)
        input_features = torch.zeros((1, 80, 3000), dtype=torch.float32)
        audio_feature_lengths = torch.tensor([128], dtype=torch.long)
        audio_chunk_mapping = torch.tensor([0], dtype=torch.long)

        with patch.object(GenerationMixin, "prepare_inputs_for_generation", return_value=dict(self.base_model_inputs)):
            model_inputs = model.prepare_inputs_for_generation(
                input_ids=torch.tensor([[1]]),
                past_key_values=object(),
                cache_position=torch.tensor([0], dtype=torch.long),
                input_features=input_features,
                audio_feature_lengths=audio_feature_lengths,
                audio_chunk_mapping=audio_chunk_mapping,
                use_cache=True,
            )

        self.assertIn("input_features", model_inputs)
        self.assertIn("audio_feature_lengths", model_inputs)
        self.assertIn("audio_chunk_mapping", model_inputs)

    def test_prepare_inputs_skips_audio_on_cached_step(self):
        model = object.__new__(MossTranscribeDiarizeForConditionalGeneration)
        input_features = torch.zeros((1, 80, 3000), dtype=torch.float32)

        with patch.object(GenerationMixin, "prepare_inputs_for_generation", return_value=dict(self.base_model_inputs)):
            model_inputs = model.prepare_inputs_for_generation(
                input_ids=torch.tensor([[1]]),
                past_key_values=object(),
                cache_position=torch.tensor([8], dtype=torch.long),
                input_features=input_features,
                audio_feature_lengths=torch.tensor([128], dtype=torch.long),
                audio_chunk_mapping=torch.tensor([0], dtype=torch.long),
                use_cache=True,
            )

        self.assertNotIn("input_features", model_inputs)
        self.assertNotIn("audio_feature_lengths", model_inputs)
        self.assertNotIn("audio_chunk_mapping", model_inputs)

    def test_generate_loop_injects_audio_only_on_first_step(self):
        config = MossTranscribeDiarizeConfig(
            text_config=Qwen3Config(
                vocab_size=320,
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=1,
                num_attention_heads=4,
                num_key_value_heads=4,
                head_dim=8,
                max_position_embeddings=128,
                tie_word_embeddings=True,
            ),
            audio_config=WhisperConfig(
                num_mel_bins=80,
                d_model=32,
                encoder_layers=1,
                encoder_attention_heads=4,
                encoder_ffn_dim=64,
                max_source_positions=1500,
            ),
            audio_token_id=42,
            audio_merge_size=4,
        )
        model = MossTranscribeDiarizeForConditionalGeneration(config).eval()
        original_get_audio_features = model.model.get_audio_features
        with patch.object(model.model, "get_audio_features", wraps=original_get_audio_features) as mocked_get_audio:
            output_ids = model.generate(
                input_ids=torch.tensor([[1, 42, 2]], dtype=torch.long),
                attention_mask=torch.tensor([[1, 1, 1]], dtype=torch.long),
                input_features=torch.randn(1, 80, 3000),
                audio_feature_lengths=torch.tensor([1], dtype=torch.long),
                audio_chunk_mapping=torch.tensor([0], dtype=torch.long),
                max_new_tokens=2,
                do_sample=False,
                eos_token_id=3,
                pad_token_id=0,
            )

        self.assertEqual(output_ids.shape[0], 1)
        self.assertEqual(int(mocked_get_audio.call_count), 1)

    def test_forward_raises_when_audio_tokens_present_but_features_missing(self):
        config = MossTranscribeDiarizeConfig(
            text_config=Qwen3Config(
                vocab_size=320,
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=1,
                num_attention_heads=4,
                num_key_value_heads=4,
                head_dim=8,
                max_position_embeddings=128,
                tie_word_embeddings=True,
            ),
            audio_config=WhisperConfig(
                num_mel_bins=80,
                d_model=32,
                encoder_layers=1,
                encoder_attention_heads=4,
                encoder_ffn_dim=64,
                max_source_positions=1500,
            ),
            audio_token_id=42,
            audio_merge_size=4,
        )
        model = MossTranscribeDiarizeForConditionalGeneration(config).eval()
        with self.assertRaisesRegex(ValueError, "input_features were not provided"):
            model(
                input_ids=torch.tensor([[1, 42, 2]], dtype=torch.long),
                attention_mask=torch.tensor([[1, 1, 1]], dtype=torch.long),
                use_cache=True,
            )

    def test_load_model_uses_explicit_dtype_and_validates_meta(self):
        fake_model = _FakeModel(has_meta=False)
        device = torch.device("cpu")
        with patch(
            "moss_transcribe_diarize.inference_utils.AutoModelForCausalLM.from_pretrained",
            return_value=fake_model,
        ) as mocked_loader:
            loaded = load_model_for_inference("dummy/model", device=device, dtype=torch.float32)

        self.assertIs(loaded, fake_model)
        self.assertEqual(fake_model.moved_to, device)
        self.assertTrue(fake_model.eval_called)
        self.assertEqual(mocked_loader.call_args.kwargs["dtype"], torch.float32)
        self.assertFalse(mocked_loader.call_args.kwargs["low_cpu_mem_usage"])

    def test_load_model_raises_when_meta_parameters_exist(self):
        fake_model = _FakeModel(has_meta=True)
        with patch(
            "moss_transcribe_diarize.inference_utils.AutoModelForCausalLM.from_pretrained",
            return_value=fake_model,
        ):
            with self.assertRaises(RuntimeError):
                load_model_for_inference("dummy/model", device=torch.device("cpu"), dtype=torch.float32)

    def test_processor_fallback_to_slow_tokenizer_on_mistral_regex_conflict(self):
        first_error = TypeError("got multiple values for keyword argument 'fix_mistral_regex'")
        fallback_processor = object()
        with patch(
            "moss_transcribe_diarize.inference_utils.AutoProcessor.from_pretrained",
            side_effect=[first_error, fallback_processor],
        ) as mocked_loader:
            loaded = load_processor_for_inference("dummy/model")

        self.assertIs(loaded, fallback_processor)
        self.assertEqual(mocked_loader.call_count, 2)
        self.assertIn("use_fast", mocked_loader.call_args_list[1].kwargs)
        self.assertFalse(mocked_loader.call_args_list[1].kwargs["use_fast"])

    def test_validate_audio_token_alignment_detects_mismatch(self):
        class _Tokenizer:
            def __init__(self):
                self.audio_token = "<|audio_pad|>"

            def convert_tokens_to_ids(self, token):
                return 7 if token == self.audio_token else None

        model = SimpleNamespace(config=SimpleNamespace(audio_token_id=42))
        processor = SimpleNamespace(tokenizer=_Tokenizer(), audio_token="<|audio_pad|>")
        with self.assertRaisesRegex(ValueError, "Tokenizer/model audio token mismatch"):
            validate_audio_token_alignment(model, processor)

    def test_generate_transcription_decodes_only_new_tokens(self):
        class _Batch(dict):
            def to(self, device):
                for key, value in list(self.items()):
                    if hasattr(value, "to"):
                        self[key] = value.to(device)
                return self

        class _Tokenizer:
            audio_token = "<|audio_pad|>"

            def convert_tokens_to_ids(self, token):
                return 42 if token == self.audio_token else 0

            def decode(self, token_ids, skip_special_tokens=True):
                return ",".join(str(item) for item in token_ids.tolist())

        class _Processor:
            tokenizer = _Tokenizer()

        class _Model:
            config = SimpleNamespace(audio_token_id=42)
            generation_config = SimpleNamespace()

            def parameters(self):
                return iter([torch.zeros(1)])

            def generate(self, **kwargs):
                prompt = kwargs["input_ids"]
                return torch.cat([prompt, torch.tensor([[99, 100]], dtype=torch.long)], dim=1)

        fake_inputs = _Batch({
            "input_ids": torch.tensor([[1, 42, 2]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
            "input_features": torch.randn(1, 80, 3000),
            "audio_feature_lengths": torch.tensor([1], dtype=torch.long),
            "audio_chunk_mapping": torch.tensor([0], dtype=torch.long),
        })
        with patch("moss_transcribe_diarize.inference_utils.prepare_inputs", return_value=fake_inputs):
            output = generate_transcription(_Model(), _Processor(), messages=[])
        self.assertEqual(output["text"], "99,100")
        self.assertEqual(output["prompt_len"], 3)
        self.assertEqual(output["generated_tokens"], 2)


if __name__ == "__main__":
    unittest.main()
