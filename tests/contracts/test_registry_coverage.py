"""Every registered component must be exercised by at least one contract case.

This is the extensibility gate: registering a new processor, tokenizer,
embedding, positional encoding, encoder, objective, or masking strategy fails
here until a case in ``testkit.cases`` exercises it, which in turn subjects it
to the full behavioural contract.
"""

from __future__ import annotations

import pytest

from audiossl.modelling import registry
from testkit.cases import MASKING_CASES, MODEL_CASES, OBJECTIVE_CASES


def _model_components() -> set[tuple[str, str]]:
    covered: set[tuple[str, str]] = set()
    for case in MODEL_CASES:
        covered |= case.components()
    for case in OBJECTIVE_CASES:
        config = case.model()
        covered |= {
            ("processor", config.processor.type),
            ("tokenizer", config.tokenizer.type),
            ("embedding", config.embedding.type),
            ("positional_encoding", config.embedding.positional_encoding.type),
            ("encoder", config.encoder.type),
        }
    return covered


@pytest.mark.parametrize(
    ("kind", "registered"),
    [
        ("processor", registry.get_processor_registry),
        ("tokenizer", registry.get_tokenizer_registry),
        ("embedding", registry.get_embedding_registry),
        ("positional_encoding", registry.get_embedding_positional_encoding_registry),
        ("encoder", registry.get_encoder_block_registry),
    ],
)
def test_every_model_component_is_exercised_by_a_model_case(kind, registered):
    covered = {name for k, name in _model_components() if k == kind}
    missing = sorted(set(registered()) - covered)
    assert missing == [], f"add a ModelCase exercising {kind}(s) {missing}"


def test_every_objective_is_exercised_by_an_objective_case():
    covered = {case.type for case in OBJECTIVE_CASES}
    missing = sorted(set(registry.get_objective_registry()) - covered)
    assert missing == [], f"add an ObjectiveCase for {missing}"


def test_every_masking_strategy_has_a_contract_case():
    missing = sorted(set(registry.get_masking_strategy_registry()) - set(MASKING_CASES))
    assert missing == [], f"add MASKING_CASES entries for {missing}"


def test_case_ids_are_unique():
    for cases in (MODEL_CASES, OBJECTIVE_CASES):
        ids = [case.id for case in cases]
        assert len(ids) == len(set(ids))
