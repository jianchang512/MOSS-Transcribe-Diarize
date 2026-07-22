import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from moss_transcribe_diarize.inference_utils import (
    load_model_for_inference,
    load_processor_for_inference,
)
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
    def test_prepare_inputs_keeps_audio_on_first_step_without_cache(self):
        model = object.__new__(MossTranscribeDiarizeForConditionalGeneration)
        input_features = torch.zeros((1, 80, 3000), dtype=torch.float32)
        audio_feature_lengths = torch.tensor([128], dtype=torch.long)
        audio_chunk_mapping = torch.tensor([0], dtype=torch.long)

        with patch.object(GenerationMixin, "prepare_inputs_for_generation", return_value={"input_ids": torch.tensor([[1]])}):
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

        with patch.object(GenerationMixin, "prepare_inputs_for_generation", return_value={"input_ids": torch.tensor([[1]])}):
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

        with patch.object(GenerationMixin, "prepare_inputs_for_generation", return_value={"input_ids": torch.tensor([[1]])}):
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
        self.assertFalse(mocked_loader.call_args_list[1].kwargs.get("use_fast", True))


if __name__ == "__main__":
    unittest.main()
