"""Compare independent training trajectories in the same parameter coordinates."""

from collections.abc import Callable, Mapping

import torch
import torch.nn.functional as F

from .precision import Precision

TensorMap = Mapping[str, torch.Tensor]
Forward = Callable[[int, torch.Tensor], torch.Tensor]


def assert_training_parity(
    actual: torch.nn.Module,
    reference: torch.nn.Module,
    *,
    actual_forward: Forward,
    reference_forward: Forward,
    inputs: Callable[[int], torch.Tensor],
    precision: Precision,
    map_reference: Callable[[TensorMap], TensorMap] = dict,
    after_step: Callable[[], None] | None = None,
    lr: float = 0.02,
    steps: int = 3,
) -> None:
    actual_params = {n: p for n, p in actual.named_parameters() if p.requires_grad}
    reference_params = {n: p for n, p in reference.named_parameters() if p.requires_grad}
    assert actual_params and reference_params
    assert sum(p.numel() for p in actual_params.values()) == sum(
        p.numel() for p in reference_params.values()
    )
    actual_opt = torch.optim.SGD(actual_params.values(), lr=lr, momentum=0.9, foreach=False)
    reference_opt = torch.optim.SGD(
        reference_params.values(), lr=lr, momentum=0.9, foreach=False
    )

    def compare(left: TensorMap, right: TensorMap, label: str) -> None:
        expected = map_reference(right)
        assert left.keys() == expected.keys(), f"{label}: {left.keys() ^ expected.keys()}"
        for name, value in left.items():
            precision.close(value, expected[name], label=f"{label} {name}")

    compare(actual_params, reference_params, "initial parameter")
    for step in range(steps):
        source = inputs(step)
        actual_input = source.detach().clone().requires_grad_()
        reference_input = source.detach().clone().requires_grad_()
        actual_opt.zero_grad(set_to_none=True)
        reference_opt.zero_grad(set_to_none=True)
        actual_output = actual_forward(step, actual_input)
        reference_output = reference_forward(step, reference_input)
        precision.close(actual_output, reference_output, label=f"step {step} output")

        actual_loss, reference_loss = actual_output, reference_output
        if actual_output.ndim:
            # A dense, nonuniform probe avoids cancellations through LayerNorm.
            probe = torch.randn(
                actual_output.shape, dtype=actual_output.dtype,
                generator=torch.Generator().manual_seed(900 + step),
            )
            actual_loss = F.mse_loss(actual_output, probe)
            reference_loss = F.mse_loss(reference_output, probe)
            precision.close(actual_loss, reference_loss, label=f"step {step} loss")
        actual_loss.backward()
        reference_loss.backward()

        actual_grads = {n: p.grad for n, p in actual_params.items() if p.grad is not None}
        reference_grads = {n: p.grad for n, p in reference_params.items() if p.grad is not None}
        for side, params, grads in (
            ("actual", actual_params, actual_grads),
            ("reference", reference_params, reference_grads),
        ):
            missing = params.keys() - grads.keys()
            assert not missing, f"step {step}: {side} missing {sorted(missing)}"
        compare(actual_grads, reference_grads, f"step {step} gradient")
        assert actual_input.grad is not None and reference_input.grad is not None
        assert reference_input.grad.count_nonzero(), f"step {step}: input probe has no gradient"
        precision.close(
            actual_input.grad, reference_input.grad, label=f"step {step} input gradient"
        )
        actual_opt.step()
        reference_opt.step()
        if after_step is not None:
            after_step()
        compare(actual_params, reference_params, f"step {step} updated parameter")
