from audiossl.config import ConstantScalarSchedulerConfig, ExponentialDecayScalarSchedulerConfig
from audiossl.modelling.objectives.wav2vec2.scalar_scheduler import build_scalar_scheduler


def test_constant_scalar_scheduler():
    scheduler = build_scalar_scheduler(ConstantScalarSchedulerConfig(value=1.5))
    assert scheduler.current_value() == 1.5
    scheduler.step()
    assert scheduler.current_value() == 1.5
    scheduler.reset()
    assert scheduler.current_value() == 1.5


def test_exponential_decay_scalar_scheduler_clamps_to_end():
    scheduler = build_scalar_scheduler(
        ExponentialDecayScalarSchedulerConfig(start=2.0, end=0.5, decay=0.5)
    )
    assert scheduler.current_value() == 2.0
    scheduler.step()
    assert scheduler.current_value() == 1.0
    scheduler.step()
    assert scheduler.current_value() == 0.5
    scheduler.step()
    assert scheduler.current_value() == 0.5

