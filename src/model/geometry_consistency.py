"""Loss utilities for paired counterfactual-geometry training."""

import torch
import torch.nn.functional as F


def linear_consistency_scale(
    global_step,
    steps_per_epoch,
    warmup_epochs,
    ramp_epochs,
):
    """Return a resume-stable 0..1 auxiliary-loss scale."""
    steps_per_epoch = int(steps_per_epoch)
    warmup_epochs = int(warmup_epochs)
    ramp_epochs = int(ramp_epochs)
    if steps_per_epoch <= 0:
        raise ValueError('steps_per_epoch must be positive')
    if warmup_epochs < 0:
        raise ValueError('warmup_epochs must be non-negative')
    if ramp_epochs < 0:
        raise ValueError('ramp_epochs must be non-negative')

    warmup_steps = warmup_epochs * steps_per_epoch
    if global_step < warmup_steps:
        return 0.0
    if ramp_epochs == 0:
        return 1.0

    ramp_steps = ramp_epochs * steps_per_epoch
    progress = (float(global_step) - warmup_steps) / ramp_steps
    return float(min(max(progress, 0.0), 1.0))


def symmetric_js_divergence(clean_logits, counterfactual_logits, temperature=1.0):
    """Mean Jensen-Shannon divergence between two logit batches."""
    if clean_logits.shape != counterfactual_logits.shape:
        raise ValueError('Paired logits must have identical shapes')
    if clean_logits.ndim != 2:
        raise ValueError('Paired logits must have shape [N, num_class]')
    temperature = float(temperature)
    if temperature <= 0.0:
        raise ValueError('temperature must be positive')

    clean_log_probability = F.log_softmax(
        clean_logits.float() / temperature, dim=1
    )
    counterfactual_log_probability = F.log_softmax(
        counterfactual_logits.float() / temperature, dim=1
    )
    clean_probability = clean_log_probability.exp()
    counterfactual_probability = counterfactual_log_probability.exp()
    mixture = 0.5 * (clean_probability + counterfactual_probability)
    log_mixture = mixture.clamp_min(1e-8).log()

    clean_kl = torch.sum(
        clean_probability * (clean_log_probability - log_mixture), dim=1
    )
    counterfactual_kl = torch.sum(
        counterfactual_probability
        * (counterfactual_log_probability - log_mixture),
        dim=1,
    )
    return 0.5 * (clean_kl + counterfactual_kl).mean() * temperature ** 2


def symmetric_stop_gradient_cosine(clean_feature, counterfactual_feature):
    """Symmetric cosine alignment with a detached target on each half."""
    if clean_feature.shape != counterfactual_feature.shape:
        raise ValueError('Paired features must have identical shapes')
    if clean_feature.ndim != 2:
        raise ValueError('Paired features must have shape [N, channel]')

    clean_feature = F.layer_norm(
        clean_feature.float(), (clean_feature.shape[1],)
    )
    counterfactual_feature = F.layer_norm(
        counterfactual_feature.float(),
        (counterfactual_feature.shape[1],),
    )
    clean_feature = F.normalize(clean_feature, dim=1)
    counterfactual_feature = F.normalize(counterfactual_feature, dim=1)

    counterfactual_to_clean = 1.0 - torch.sum(
        counterfactual_feature * clean_feature.detach(), dim=1
    )
    clean_to_counterfactual = 1.0 - torch.sum(
        clean_feature * counterfactual_feature.detach(), dim=1
    )
    return 0.5 * (
        counterfactual_to_clean.mean() + clean_to_counterfactual.mean()
    )
