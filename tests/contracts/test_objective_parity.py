"""Reference criteria, objective wiring, and persistent updates over three steps."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from audiossl.config import AudioBatch
from audiossl.modelling.objectives.ufo.masking import inverse_block_masking
from audiotrainer.step import StepContext

from testkit.cases import objective_params
from testkit.parity import assert_training_parity


def _ctx(step: int = 0) -> StepContext:
    return StepContext(
        optimizer_step=step,
        total_optimizer_steps=3,
        epoch=0,
        global_step=step,
        step_was_skipped=False,
    )


def _contrastive(config, out, _objective, _batch):
    logits = F.cosine_similarity(out.zi[:, None, :], out.zj[None, :, :], dim=-1)
    logits = logits / config.temperature
    loss = (logits.logsumexp(dim=1) - logits.diagonal()).mean()
    if config.symmetric:
        loss = (loss + (logits.logsumexp(dim=0) - logits.diagonal()).mean()) / 2
    return loss


def _byol(_config, out, _objective, _batch):
    predictions = torch.cat((out.prediction_i, out.prediction_j))
    targets = torch.cat((out.target_j, out.target_i)).detach()
    return (2 - 2 * F.cosine_similarity(predictions, targets)).mean()


def _dino(config, out, objective, _batch):
    student = torch.cat((out.student_logits_i, out.student_logits_j))
    teacher = torch.cat((out.teacher_logits_j, out.teacher_logits_i)).detach()
    distribution = ((teacher - objective.center) / config.teacher_temperature).softmax(-1)
    return F.cross_entropy(student / config.student_temperature, distribution)


def _jepa(config, out, _objective, _batch):
    distance = F.mse_loss if config.loss_style == "mse" else F.l1_loss
    return distance(out.prediction[out.target_mask], out.target.detach()[out.target_mask])


def _mae(config, out, _objective, _batch):
    target = out.target.detach().float()
    if config.norm_pix_loss != "none":
        dims = -1 if config.norm_pix_loss == "per_patch" else (0, 1, 2)
        variance, mean = torch.var_mean(target, dim=dims, keepdim=True)
        target = (target - mean) * torch.rsqrt(variance + config.norm_pix_eps)
    return F.mse_loss(out.prediction.float()[out.target_mask], target[out.target_mask])


def _ufo(config, out, _objective, _batch):
    frame_loss = F.mse_loss(out.prediction[out.frame_mask], out.target.detach()[out.frame_mask])
    utterance_loss = F.mse_loss(out.student_cls, out.teacher_utterance_target.detach())
    return frame_loss + config.utterance_loss_weight * utterance_loss


def _wav2vec2(config, out, _objective, _batch):
    contrastive = F.cross_entropy(out.logits, torch.zeros_like(out.targets))
    diversity = 1 - out.prob_perplexity / (config.quantizer.num_groups * config.quantizer.num_vars)
    return (
        contrastive + config.diversity_loss_weight * diversity
        + config.features_pen_weight * out.raw_features.float().square().mean()
    )


def _hubert(config, out, _objective, _batch):
    weights = torch.where(
        out.time_mask_indices, config.masked_loss_weight, config.unmasked_loss_weight
    )
    selected = out.mask & (weights > 0)
    ce = F.cross_entropy(out.logits[selected], out.targets[selected], reduction="none")
    return torch.dot(ce, weights[selected].to(ce.dtype)) / selected.sum()


CRITERIA = {
    "contrastive": _contrastive,
    "byol": _byol,
    "dino": _dino,
    "jepa": _jepa,
    "mae": _mae,
    "ufo": _ufo,
    "wav2vec2": _wav2vec2,
    "hubert": _hubert,
}


def _project(head, x):
    projected = head.layers(x)
    if head.config.final_l2_normalize:
        projected = F.normalize(projected, dim=-1)
    return projected


def _prototype_logits(head, x):
    bottleneck = F.normalize(head.body(x), dim=-1)
    weight = F.normalize(head.prototypes.weight_v, dim=-1) * head.prototypes.weight_g
    return F.linear(bottleneck, weight)


def _view_outputs(kind, objective, batch):
    model = objective.model
    augment = objective.augmentor
    views = (batch, batch) if augment is None else (augment(batch), augment(batch))
    pooled = [model.get_cls(model(view)) for view in views]
    if kind == "contrastive":
        zi, zj = (_project(objective.projection, value) for value in pooled)
        return SimpleNamespace(zi=zi, zj=zj)

    head_name = "projector" if kind == "byol" else "head"
    apply_head = _project if kind == "byol" else _prototype_logits
    head = getattr(objective, head_name)
    students = [apply_head(head, value) for value in pooled]
    if kind == "byol":
        students = [_project(objective.predictor, value) for value in students]
    teacher = objective.teacher_model or model
    teacher_head = getattr(objective, f"teacher_{head_name}") or head
    teacher_mode, head_mode = teacher.training, teacher_head.training
    try:
        teacher.eval()
        teacher_head.eval()
        with torch.no_grad():
            targets = [apply_head(teacher_head, teacher.get_cls(teacher(view))) for view in views]
    finally:
        teacher.train(teacher_mode)
        teacher_head.train(head_mode)
    if kind == "byol":
        return SimpleNamespace(
            prediction_i=students[0], prediction_j=students[1],
            target_i=targets[0], target_j=targets[1],
        )
    return SimpleNamespace(
        student_logits_i=students[0], student_logits_j=students[1],
        teacher_logits_i=targets[0], teacher_logits_j=targets[1],
    )


def _jepa_outputs(objective, batch, target_mask):
    model = objective.model
    features = model.process(batch)
    layout = model.token_layout(features)
    tokens = model.tokenize(features)
    embedding = model.embedding
    selected = torch.zeros_like(tokens.mask)
    selected[:, ::embedding.subsampling_factor] = target_mask[:, embedding.num_prefix_tokens:]
    teacher = objective.teacher_model or model
    teacher_mode = teacher.training
    try:
        teacher.eval()
        with torch.no_grad():
            if teacher is model:
                target = teacher.encode_tokens(tokens, layout=layout).z
            else:
                target = teacher(batch).z
    finally:
        teacher.train(teacher_mode)
    masked = AudioBatch(
        x=torch.where(selected[..., None], objective.mask_token, tokens.x),
        mask=tokens.mask & ~selected,
    )
    predicted = objective.predictor(model.encode_tokens(masked, layout=layout).z)
    config = objective.config
    normalize = (
        (lambda x: F.normalize(x, dim=-1)) if config.output_norm_type == "l2"
        else (lambda x: F.layer_norm(x, (x.shape[-1],)))
    )
    return SimpleNamespace(
        prediction=normalize(predicted) if config.student_output_norm else predicted,
        target=normalize(target) if config.teacher_output_norm else target,
        target_mask=target_mask,
    )


def _mae_outputs(objective, batch, ids_restore):
    model = objective.model
    features = model.process(batch)
    layout = model.token_layout(features)
    tokens = model.tokenize(features)
    embedding = model.embedding
    positioned = embedding.position(embedding.project(embedding.prepare(tokens)), layout=layout)
    visible_fraction = 1 - objective.config.masking.mask_ratio
    keep = int(layout.num_tokens * visible_fraction)
    ids_keep = ids_restore.argsort(dim=-1)[:, :keep]
    rows = torch.arange(ids_keep.shape[0])[:, None]
    per_row_keep = (tokens.mask.sum(dim=-1) * visible_fraction).long().clamp_(1, keep)
    visible_mask = (ids_restore < per_row_keep[:, None]) & tokens.mask
    keep_valid = visible_mask[rows, ids_keep]
    visible = AudioBatch(x=positioned.x[rows, ids_keep], mask=keep_valid, positions=ids_keep)
    encoded = model.encode_embedded(embedding.finalize(visible))
    decoder = objective.decoder
    projected = decoder.input_proj(encoded.z[:, encoded.num_prefix_tokens:])
    full = decoder.mask_token.expand(*tokens.mask.shape, projected.shape[-1]).clone()
    full[rows.expand_as(ids_keep)[keep_valid], ids_keep[keep_valid]] = projected[keep_valid]
    full = decoder.positional_encoding(full, layout, mask=tokens.mask)
    prediction = decoder.outputlayer(decoder.blocks(AudioBatch(x=full, mask=tokens.mask)).z)
    target_tokens = model.tokenize(model.processor.reconstruction_target(batch))
    target = model.tokenizer.reorder_patch_targets(
        target_tokens.x, layout=layout, order=objective.config.target_patch_order,
    )
    return SimpleNamespace(
        prediction=prediction, target=target, target_mask=~visible_mask & tokens.mask,
    )


def _ufo_outputs(objective, batch):
    model = objective.model
    features = model.process(batch)
    layout = model.token_layout(features)
    tokens = model.tokenize(features)
    embedding = model.embedding
    positioned = embedding.position(embedding.project(embedding.prepare(tokens)), layout=layout)
    teacher = objective.teacher_model or model
    teacher_mode = teacher.training
    try:
        teacher.eval()
        with torch.no_grad():
            target = teacher.encode_tokens(
                teacher.tokenize(features), layout=layout, return_hidden_states=True
            )
            patch_target = torch.stack(target.hidden_states).mean(0)[:, target.num_prefix_tokens:]
            utterance_target = torch.stack([
                row[valid].mean(0) for row, valid in zip(patch_target, tokens.mask, strict=True)
            ])
    finally:
        teacher.train(teacher_mode)

    mask_config = objective.config.masking
    copies = mask_config.num_masks
    patches = positioned.x.repeat_interleave(copies, dim=0)
    valid = tokens.mask.repeat_interleave(copies, dim=0)
    _, removed, _, keep = inverse_block_masking(
        batch_size=patches.shape[0], time_patches=layout.time_patches,
        frequency_patches=layout.frequency_patches, mask_ratio=mask_config.mask_ratio,
        block_size=mask_config.block_size, device=patches.device, generator=objective.generator,
        valid_mask=valid,
    )
    rows = torch.arange(keep.shape[0])[:, None]
    visible = AudioBatch(x=patches[rows, keep], mask=valid[rows, keep], positions=keep)
    encoded = model.encode_embedded(embedding.finalize(visible))
    restored = objective.mask_token.expand_as(patches).clone()
    restored[rows, keep] = encoded.z[:, encoded.num_prefix_tokens:]
    x = restored.unflatten(1, (layout.time_patches, layout.frequency_patches))
    for conv, norm, activation in zip(
        objective.decoder.layers[::3], objective.decoder.layers[1::3],
        objective.decoder.layers[2::3], strict=True,
    ):
        x = activation(norm(conv(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)))
    return SimpleNamespace(
        prediction=x.flatten(1, 2), target=patch_target.repeat_interleave(copies, dim=0),
        frame_mask=removed & valid, student_cls=encoded.z[:, 0],
        teacher_utterance_target=utterance_target.repeat_interleave(copies, dim=0),
    )


def _masked_speech_outputs(objective, batch):
    config = objective.config
    model = objective.model
    raw = model.process(batch)
    x = raw.x.detach() + config.feature_grad_mult * (raw.x - raw.x.detach())
    tokens = model.tokenize(replace(raw, x=x))
    embedding = model.embedding
    prepared = embedding.prepare(tokens)
    projected = embedding.project(prepared)
    time_mask = objective.time_masking.compute_mask(
        shape=projected.mask.shape, valid_mask=projected.mask, generator=objective.generator,
    )
    assert time_mask.any(), "fixture must exercise masked prediction"
    x = projected.x
    if objective.feature_masking is not None:
        channels = objective.feature_masking.compute_mask(
            shape=(x.shape[0], x.shape[-1]), valid_mask=None, generator=objective.generator,
        )
        x = x.masked_fill(channels[:, None, :], 0)
    x = torch.where(time_mask[..., None], objective.mask_emb, x)
    positioned = embedding.position(replace(projected, x=x), layout=model.token_layout(raw))
    encoded = model.encode_embedded(embedding.finalize(positioned))
    hidden = objective.project_hid(encoded.z[:, encoded.num_prefix_tokens:])
    if config.type == "hubert":
        with torch.no_grad():
            targets = objective.teacher(batch)
        return SimpleNamespace(
            logits=F.linear(hidden, objective.label_emb) / config.logit_temp,
            targets=targets, time_mask_indices=time_mask, mask=projected.mask,
        )

    quantizer = objective.quantizer(prepared.x)
    candidates = objective.project_q(quantizer.quantized)
    negatives = []
    for row, mask in zip(candidates, time_mask, strict=True):
        group = row[mask]
        for index in range(len(group)):
            draws = torch.randint(
                max(len(group) - 1, 1), (config.num_negatives,), generator=objective.generator
            )
            negatives.append(group[(draws + (draws >= index)).clamp_max(len(group) - 1)])
    assert config.cross_sample_negatives == 0
    choices = torch.cat((candidates[time_mask].unsqueeze(1), torch.stack(negatives)), dim=1)
    logits = F.cosine_similarity(hidden[time_mask].unsqueeze(1).float(), choices.float(), dim=-1)
    return SimpleNamespace(
        logits=logits / config.logit_temp,
        targets=torch.zeros(logits.shape[0], dtype=torch.long),
        prob_perplexity=quantizer.prob_perplexity, raw_features=raw.x,
    )


@pytest.mark.parametrize("case", objective_params("criterion_parity"))
def test_objective_training_matches_reference_criterion(case, parity_precision):
    config = case.objective_config()
    if case.type == "mae":
        config = replace(config, norm_pix_eps=parity_precision.eps)
        case = replace(case, objective=lambda: config)
    actual = case.build().train()
    reference = case.build().train()
    criterion = CRITERIA[case.type]
    batch = case.batch()
    sampled = None

    def actual_forward(step, waveform):
        nonlocal sampled
        torch.manual_seed(100 + step)
        sampled = actual(replace(batch, x=waveform))
        actual.on_after_forward(sampled, _ctx(step))
        return sampled.loss

    def reference_forward(step, waveform):
        torch.manual_seed(100 + step)
        reference_batch = replace(batch, x=waveform)
        if case.type in {"contrastive", "byol", "dino"}:
            out = _view_outputs(case.type, reference, reference_batch)
        elif case.type == "jepa":
            out = _jepa_outputs(reference, reference_batch, sampled.target_mask)
        elif case.type == "mae":
            out = _mae_outputs(reference, reference_batch, sampled.ids_restore)
        elif case.type == "ufo":
            out = _ufo_outputs(reference, reference_batch)
        else:
            out = _masked_speech_outputs(reference, reference_batch)
        for name, expected in vars(out).items():
            if not hasattr(sampled, name):
                continue
            observed = getattr(sampled, name)
            if expected.is_floating_point():
                parity_precision.close(observed, expected, label=f"step {step} {name}")
            else:
                assert torch.equal(observed, expected), f"step {step} {name}"
        loss = criterion(config, out, reference, reference_batch)
        if case.type == "dino":
            with torch.no_grad():
                mean = torch.cat((out.teacher_logits_i, out.teacher_logits_j)).mean(0)
                reference.center.lerp_(mean, 1 - config.center_momentum)
        return loss

    @torch.no_grad()
    def after_step(step: int = 0):
        actual.on_after_optimizer_step(_ctx(step))
        for teacher_name, student_name in case.teacher_pairs:
            teacher = getattr(reference, teacher_name)
            student = getattr(reference, student_name)
            for target, source in zip(teacher.parameters(), student.parameters(), strict=True):
                target.lerp_(source, 1 - config.teacher_ema_decay)
            for target, source in zip(teacher.buffers(), student.buffers(), strict=True):
                target.copy_(source)
        if case.type == "wav2vec2":
            reference.quantizer.temperature_scheduler.step_count.add_(1)
        actual_state = actual.state_dict()
        for name, expected in reference.state_dict().items():
            if not actual_state[name].is_floating_point():
                assert torch.equal(actual_state[name], expected), name
            elif name == "center" or name.startswith("teacher"):
                parity_precision.close(actual_state[name], expected, label=name)
        assert all(p.grad is None for p in actual.parameters() if not p.requires_grad)
        assert all(p.grad is None for p in reference.parameters() if not p.requires_grad)

    assert_training_parity(
        actual, reference,
        actual_forward=actual_forward,
        reference_forward=reference_forward,
        inputs=lambda step: case.batch(seed=step).x,
        precision=parity_precision,
        after_step=after_step,
        lr=0.001,
    )
