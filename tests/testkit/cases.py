"""Case registries the contract and reproducibility suites parametrize over.

A *case* is a named, fully specified way to build a model or objective plus a
matching input batch. Contract tests are written once against the case
interface and run over every case, so adding a component means adding a case
here, not writing a new test file.

Adding a component
------------------
1. Register it in ``audiossl.modelling.registry``.
2. Add a builder to ``testkit.configs`` and a ``ModelCase``/``ObjectiveCase``
   below that varies *only* that component relative to the base case.
3. Run ``pytest tests/contracts``. ``test_registry_coverage`` fails until the
   new registry key is exercised by some case.
4. If a contract property genuinely does not hold, record it in ``known``
   with the property name, a reason, and whether it is a bug:

   - ``Known(bug=True)`` marks the test ``xfail(strict=True)``. Fixing the
     bug turns the xfail into an XPASS failure, forcing the entry's removal.
   - ``Known(bug=False)`` marks the test ``skip``: the deviation is a
     deliberate modelling choice (e.g. reference-faithful normalization over
     padding) and the reason documents why.

Property names are the ``PROPERTY`` constants declared by each contract test;
``"*"`` applies to every property of the case.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field, replace
from typing import Any, Literal

import pytest
import torch

from audiossl.config import AudioModelConfig, ChannelMaskingConfig, SpanMaskingConfig
from audiossl.config.batch_dataclasses import AudioBatch
from audiossl.modelling.model import AudioModel
from audiossl.modelling.objectives._base import Objective
from audiossl.modelling.objectives.hubert.hubert import HuBERTObjective
from audiossl.modelling.objectives.objective import build_objective

from . import batches, configs
from .stubs import FrameAlignedStubTeacher, SeededGainAugmentor


@dataclass(frozen=True)
class Known:
    """A documented deviation from a contract property for one case."""

    reason: str
    bug: bool
    raises: type[BaseException] | tuple[type[BaseException], ...] | None = None

    def mark(self) -> pytest.MarkDecorator:
        if self.bug:
            return pytest.mark.xfail(reason=self.reason, strict=True, raises=self.raises)
        return pytest.mark.skip(reason=f"by design: {self.reason}")


def _marks(known: Mapping[str, Known], prop: str) -> list[pytest.MarkDecorator]:
    entry = known.get(prop) or known.get("*")
    return [entry.mark()] if entry is not None else []


# ------------------------------------------------------------------ models ---


@dataclass(frozen=True)
class ModelCase:
    id: str
    config: Callable[[], AudioModelConfig]
    lengths: tuple[int, ...] = configs.DEFAULT_LENGTHS
    known: Mapping[str, Known] = field(default_factory=dict)

    def model_config(self, seed: int = 0) -> AudioModelConfig:
        config = self.config()
        return _with_seed(config, seed)

    def build(self, seed: int = 0) -> AudioModel:
        return AudioModel(self.model_config(seed))

    def batch(self, seed: int = 0) -> AudioBatch:
        return batches.waveform_batch(self.lengths, seed=seed)

    def components(self) -> set[tuple[str, str]]:
        return _components(self.config())


def _with_seed(config: AudioModelConfig, seed: int) -> AudioModelConfig:
    return replace(config, runtime=replace(config.runtime, seed=seed))


def _components(config: AudioModelConfig) -> set[tuple[str, str]]:
    return {
        ("processor", config.processor.type),
        ("tokenizer", config.tokenizer.type),
        ("embedding", config.embedding.type),
        ("positional_encoding", config.embedding.positional_encoding.type),
        ("encoder", config.encoder.type),
    }


_FORWARD_PROPS = (
    "output_contract",
    "staged_api",
    "padding_invariance",
    "batch_independence",
    "input_gradient_isolation",
    "parameter_gradients",
    "eval_determinism",
    "bfloat16",
    "state_dict_roundtrip",
    "golden_forward",
)
_PADDING_PROPS = ("padding_invariance", "batch_independence", "input_gradient_isolation")


def _padding_known(
    reason: str,
    *,
    bug: bool,
    props: tuple[str, ...] = _PADDING_PROPS,
) -> dict[str, Known]:
    return {prop: Known(reason, bug=bug) for prop in props}


c = configs

MODEL_CASES: tuple[ModelCase, ...] = (
    ModelCase("base", c.model),
    # Processors.
    ModelCase(
        "processor_conv",
        lambda: c.model(processor=c.conv_processor(), tokenizer=c.noop_tokenizer()),
    ),
    ModelCase(
        "processor_wav2vec2_layer_norm",
        lambda: c.model(processor=c.wav2vec2_processor(), tokenizer=c.noop_tokenizer()),
    ),
    ModelCase(
        "processor_kaldi_fbank",
        lambda: c.model(processor=c.kaldi_fbank()),
        lengths=(3200, 2880, 2400),
    ),
    ModelCase(
        "processor_logmel_batchnorm",
        lambda: c.model(processor=c.melspec(log_mel=True, batch_norm=True)),
    ),
    # Tokenizers.
    ModelCase("tokenizer_noop", lambda: c.model(tokenizer=c.noop_tokenizer())),
    ModelCase(
        "tokenizer_patchify_strided_pad",
        lambda: c.model(
            tokenizer=c.patchify(patch_shape=(3, 4), stride=(2, 4), padding_strategy="pad")
        ),
    ),
    # Embeddings.
    ModelCase(
        "embedding_conv",
        lambda: c.model(tokenizer=c.noop_tokenizer(), embedding=c.conv_embedding()),
    ),
    ModelCase(
        "embedding_conv_subsampled",
        lambda: c.model(
            tokenizer=c.noop_tokenizer(),
            embedding=c.conv_embedding(subsampling=c.subsampling(2)),
        ),
    ),
    ModelCase(
        "embedding_linear_subsampled_normed",
        lambda: c.model(
            tokenizer=c.noop_tokenizer(),
            embedding=c.linear_embedding(
                c.pe_learnable(),
                subsampling=c.subsampling(2),
                input_layer_norm=True,
                post_pos_layer_norm=True,
            ),
        ),
    ),
    # Positional encodings.
    ModelCase("pe_learnable", lambda: c.model(embedding=c.linear_embedding(c.pe_learnable()))),
    ModelCase("pe_sinusoidal", lambda: c.model(embedding=c.linear_embedding(c.pe_sinusoidal()))),
    ModelCase(
        "pe_sinusoidal_2d", lambda: c.model(embedding=c.linear_embedding(c.pe_sinusoidal_2d()))
    ),
    ModelCase(
        "pe_learnable_2d", lambda: c.model(embedding=c.linear_embedding(c.pe_learnable_2d()))
    ),
    ModelCase("pe_conv1d", lambda: c.model(embedding=c.linear_embedding(c.pe_conv1d()))),
    # Encoders.
    ModelCase("encoder_post_norm", lambda: c.model(encoder=c.transformer(pre_norm=False, final_norm=True))),
    ModelCase("encoder_conformer", lambda: c.model(encoder=c.conformer())),
    ModelCase("encoder_branchformer", lambda: c.model(encoder=c.branchformer())),
    ModelCase("encoder_ebranchformer", lambda: c.model(encoder=c.ebranchformer())),
    ModelCase(
        "encoder_alt_transformer",
        lambda: c.model(encoder=c.alt_transformer(pooling="masked_mean")),
    ),
    # Reference presets at tiny scale.
    ModelCase(
        "preset_wav2vec2",
        c.wav2vec2_model,
        known=_padding_known(
            "wav2vec2-base 'default' extractor GroupNorm normalizes over the full padded "
            "time axis, matching fairseq/HF; padded batches are not padding-invariant",
            bug=False,
        ),
    ),
    ModelCase("preset_eat", c.eat_model, lengths=configs.EAT_LENGTHS),
    ModelCase(
        "preset_dasheng_like",
        c.dasheng_like_model,
        known={
            **_padding_known(
                "frame_validity='overlap_any' keeps edge frames that overlap padding, "
                "matching the Dasheng reference framing",
                bug=False,
            ),
            "declared_lengths": Known(
                "MelSpectrogramProcessor.output_lengths always applies the strict-window "
                "formula and disagrees with output_mask under frame_validity='overlap_any'",
                bug=True,
            ),
        },
    ),
)


MASKING_CASES: Mapping[str, Any] = {
    "span": SpanMaskingConfig(mask_prob=0.3, mask_length=3),
    "channel": ChannelMaskingConfig(mask_prob=0.3, mask_length=2),
}


def model_params(prop: str, cases: Iterable[ModelCase] = MODEL_CASES) -> list[Any]:
    return [pytest.param(case, id=case.id, marks=_marks(case.known, prop)) for case in cases]


def model_case(case_id: str) -> ModelCase:
    return _by_id(MODEL_CASES, case_id)


# -------------------------------------------------------------- objectives ---

TeacherKind = Literal["ema", "stop_grad", "frozen", "none"]
Construct = Callable[[AudioModel, Any, Any, "torch.Generator | None"], Objective]


def _construct_hubert(model: AudioModel, config: Any, augmentor: Any, generator: Any) -> Objective:
    teacher = FrameAlignedStubTeacher(
        model.processor.processor.output_mask, num_codes=configs.HUBERT_NUM_CODES
    )
    return HuBERTObjective(model, config, generator=generator, teacher=teacher)


@dataclass(frozen=True)
class ObjectiveCase:
    """A buildable objective plus everything a generic test needs to know about it.

    ``teacher_pairs`` maps EMA teacher submodule attributes to the student
    attributes they track, e.g. ``("teacher_model", "model")``.
    ``frozen_student`` lists student parameters frozen by design (e.g. DINO's
    weight-normalized prototype magnitude).
    """

    id: str
    model: Callable[[], AudioModelConfig]
    objective: Callable[[], Any]
    teacher: TeacherKind
    uses_augmentor: bool = False
    uses_generator: bool = False
    teacher_pairs: tuple[tuple[str, str], ...] = ()
    frozen_student: frozenset[str] = frozenset()
    hooks: frozenset[str] = frozenset({"metrics"})
    global_rng: bool = False
    eval_forward: bool = True
    lengths: tuple[int, ...] = configs.DEFAULT_LENGTHS
    num_classes: int | None = None
    construct: Construct | None = None
    known: Mapping[str, Known] = field(default_factory=dict)

    @property
    def type(self) -> str:
        return self.objective().type

    def objective_config(self) -> Any:
        return self.objective()

    def build(
        self,
        seed: int = 0,
        *,
        generator_seed: int = 0,
        augmentor_seed: int = 0,
        model: AudioModel | None = None,
    ) -> Objective:
        model = model if model is not None else AudioModel(_with_seed(self.model(), seed))
        augmentor = SeededGainAugmentor(augmentor_seed) if self.uses_augmentor else None
        generator = (
            torch.Generator().manual_seed(generator_seed) if self.uses_generator else None
        )
        config = self.objective()
        if self.construct is not None:
            return self.construct(model, config, augmentor, generator)
        return build_objective(model, config, augmentor=augmentor, generator=generator)

    def batch(self, seed: int = 0) -> AudioBatch:
        batch = batches.waveform_batch(self.lengths, seed=seed)
        if self.num_classes is not None:
            generator = torch.Generator().manual_seed(seed + 17)
            labels = torch.randint(
                0, self.num_classes, (len(self.lengths),), generator=generator
            )
            return replace(batch, labels=labels)
        return batch


_EMA_HOOKS = frozenset({"metrics", "update_teacher", "on_after_optimizer_step"})
_DINO_FROZEN = frozenset({"head.prototypes.weight_g"})


def _conv_front_end_model() -> AudioModelConfig:
    return c.model(processor=c.conv_processor(), tokenizer=c.noop_tokenizer())


def _ufo_case(case_id: str, **objective_overrides: Any) -> ObjectiveCase:
    ema = objective_overrides.get("teacher_mode", "ema") == "ema"
    return ObjectiveCase(
        case_id,
        model=c.eat_model,
        objective=lambda: c.ufo_objective(**objective_overrides),
        teacher="ema" if ema else "stop_grad",
        uses_generator=True,
        teacher_pairs=(("teacher_model", "model"),) if ema else (),
        hooks=_EMA_HOOKS,
        lengths=configs.EAT_LENGTHS,
    )


OBJECTIVE_CASES: tuple[ObjectiveCase, ...] = (
    ObjectiveCase(
        "contrastive_augmentation",
        model=c.model,
        objective=c.contrastive_objective,
        teacher="none",
        uses_augmentor=True,
    ),
    ObjectiveCase(
        "contrastive_dropout",
        model=c.dropout_model,
        objective=lambda: c.contrastive_objective(view_strategy="dropout"),
        teacher="none",
        global_rng=True,
        eval_forward=False,
        known={
            "injected_rng_determinism": Known(
                "dropout views draw from the global torch RNG (nn.Dropout has no generator); "
                "runs are reproducible only with the global seed fixed",
                bug=False,
            )
        },
    ),
    ObjectiveCase(
        "byol_ema",
        model=_conv_front_end_model,
        objective=c.byol_objective,
        teacher="ema",
        uses_augmentor=True,
        teacher_pairs=(("teacher_model", "model"), ("teacher_projector", "projector")),
        hooks=_EMA_HOOKS,
    ),
    ObjectiveCase(
        "byol_stop_grad",
        model=c.model,
        objective=lambda: c.byol_objective(teacher_mode="stop_grad"),
        teacher="stop_grad",
        uses_augmentor=True,
        hooks=_EMA_HOOKS,
    ),
    ObjectiveCase(
        "dino_ema",
        model=_conv_front_end_model,
        objective=c.dino_objective,
        teacher="ema",
        uses_augmentor=True,
        teacher_pairs=(("teacher_model", "model"), ("teacher_head", "head")),
        frozen_student=_DINO_FROZEN,
        hooks=_EMA_HOOKS | {"on_after_forward"},
    ),
    ObjectiveCase(
        "dino_stop_grad",
        model=c.model,
        objective=lambda: c.dino_objective(teacher_mode="stop_grad"),
        teacher="stop_grad",
        uses_augmentor=True,
        frozen_student=_DINO_FROZEN,
        hooks=_EMA_HOOKS | {"on_after_forward"},
    ),
    ObjectiveCase(
        "jepa_ema",
        model=_conv_front_end_model,
        objective=c.jepa_objective,
        teacher="ema",
        uses_generator=True,
        teacher_pairs=(("teacher_model", "model"),),
        hooks=_EMA_HOOKS,
    ),
    ObjectiveCase(
        "jepa_stop_grad",
        model=c.model,
        objective=lambda: c.jepa_objective(teacher_mode="stop_grad"),
        teacher="stop_grad",
        uses_generator=True,
        hooks=_EMA_HOOKS,
    ),
    ObjectiveCase(
        "mae",
        model=c.mae_model,
        objective=c.mae_objective,
        teacher="none",
        uses_generator=True,
    ),
    _ufo_case("ufo_ema"),
    _ufo_case("ufo_stop_grad", teacher_mode="stop_grad"),
    ObjectiveCase(
        "wav2vec2",
        model=c.wav2vec2_model,
        objective=c.wav2vec2_objective,
        teacher="none",
        uses_generator=True,
        global_rng=True,
        hooks=frozenset({"metrics", "on_after_optimizer_step"}),
        known={
            "injected_rng_determinism": Known(
                "quantizer Gumbel noise comes from F.gumbel_softmax, which draws from the "
                "global torch RNG; only masking and negatives use the injected generator",
                bug=False,
            )
        },
    ),
    ObjectiveCase(
        "hubert",
        model=c.wav2vec2_model,
        objective=c.hubert_objective,
        teacher="frozen",
        uses_generator=True,
        construct=_construct_hubert,
    ),
    ObjectiveCase(
        "sft",
        model=c.model,
        objective=c.sft_objective,
        teacher="none",
        uses_augmentor=True,
        num_classes=3,
        known={
            "trainer_kwargs": Known(
                "SFT treats the augmentor as optional at build time",
                bug=False,
            ),
            "criterion_parity": Known(
                "SFT has no separate reference criterion harness yet",
                bug=False,
            ),
        },
    ),
)


def objective_params(
    prop: str,
    cases: Iterable[ObjectiveCase] = OBJECTIVE_CASES,
) -> list[Any]:
    return [pytest.param(case, id=case.id, marks=_marks(case.known, prop)) for case in cases]


def objective_case(case_id: str) -> ObjectiveCase:
    return _by_id(OBJECTIVE_CASES, case_id)


def _by_id(cases: Iterable[Any], case_id: str) -> Any:
    for case in cases:
        if case.id == case_id:
            return case
    raise KeyError(case_id)
