from typing import Any

import torch


class LoudnessAugmentation:
    required_parameters = ("p", "min_gain_db", "max_gain_db")

    def __init__(
        self,
        config: dict[str, Any],
        generator: torch.Generator | None = None,
    ):
        self.config = dict(config)
        self.generator = generator

        missing = [
            parameter
            for parameter in self.required_parameters
            if parameter not in config
        ]
        if missing:
            raise ValueError(
                "Missing required loudness augmentation parameter(s): "
                + ", ".join(missing)
            )

        self.min_gain_db = config["min_gain_db"]
        self.max_gain_db = config["max_gain_db"]
        self.p = config["p"]

        if self.min_gain_db > self.max_gain_db:
            raise ValueError("min_gain_db must be <= max_gain_db")

        if not (0.0 <= self.p <= 1.0):
            raise ValueError("p must be between 0 and 1")

    def _process_single(
        self,
        x: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        if x.ndim != 1:
            raise ValueError(f"Expected single waveform of shape (T,), got {x.shape}")

        if self.p <= 0.0:
            return x

        effective_generator = generator if generator is not None else self.generator

        if (
            self.p < 1.0
            and torch.rand((), device=x.device, generator=effective_generator) >= self.p
        ):
            return x

        if self.min_gain_db == self.max_gain_db:
            gain_db = torch.as_tensor(self.min_gain_db, device=x.device, dtype=x.dtype)
        else:
            gain_db = torch.empty((), device=x.device, dtype=x.dtype).uniform_(
                self.min_gain_db,
                self.max_gain_db,
                generator=effective_generator,
            )

        gain = 10.0 ** (gain_db / 20.0)
        return x * gain

    def __call__(
        self,
        x: list[torch.Tensor],
        generator: torch.Generator | None = None,
    ) -> list[torch.Tensor]:
        if not isinstance(x, list):
            raise TypeError(f"Expected list[torch.Tensor], got {type(x)}")

        return [self._process_single(wf, generator=generator) for wf in x]
