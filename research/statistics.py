"""Participant and stimulus disjoint fitting, paired uncertainty and temporal nulls."""

import numpy as np

from .spec import RIDGE_ALPHAS


def matrix(value, name):
    value = np.asarray(value, dtype=float)
    if value.ndim == 1:
        value = value[:, None]
    if value.ndim != 2 or not value.size or not np.isfinite(value).all():
        raise ValueError(f"{name} must be a nonempty finite matrix")
    return value


def crossed_folds(participants, stimuli, participant_folds=5, stimulus_folds=3, seed=11):
    participants, stimuli = np.asarray(participants), np.asarray(stimuli)
    if participants.ndim != 1 or participants.shape != stimuli.shape:
        raise ValueError("participant and stimulus identities must be aligned vectors")
    people, materials = np.unique(participants), np.unique(stimuli)
    if not 2 <= participant_folds <= len(people) or not 2 <= stimulus_folds <= len(materials):
        raise ValueError(
            "each split axis requires at least two groups and enough groups for every fold"
        )
    rng = np.random.default_rng(seed)
    people, materials = rng.permutation(people), rng.permutation(materials)
    for held_people in np.array_split(people, participant_folds):
        for held_stimuli in np.array_split(materials, stimulus_folds):
            p = np.isin(participants, held_people)
            s = np.isin(stimuli, held_stimuli)
            train, test = np.flatnonzero(~p & ~s), np.flatnonzero(p & s)
            if not len(train) or not len(test):
                raise ValueError("crossed design has an empty train or test intersection")
            yield train, test


def ridge_predict(x_train, y_train, x_test, alpha):
    mean, scale = x_train.mean(axis=0), x_train.std(axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    train, test = (x_train - mean) / scale, (x_test - mean) / scale
    intercept = y_train.mean(axis=0)
    coefficients = np.linalg.solve(
        train.T @ train + alpha * np.eye(train.shape[1]), train.T @ (y_train - intercept)
    )
    return test @ coefficients + intercept


def scores(observed, predicted):
    observed, predicted = matrix(observed, "observed"), matrix(predicted, "predicted")
    if observed.shape != predicted.shape:
        raise ValueError("observations and predictions must have the same shape")
    variance = np.var(observed, axis=0)
    explained = np.full(variance.shape, np.nan)
    nonconstant = variance > 1e-15
    explained[nonconstant] = (
        1 - np.var(observed - predicted, axis=0)[nonconstant] / variance[nonconstant]
    )
    return {
        "mse": float(np.mean((observed - predicted) ** 2)),
        "mae": float(np.mean(np.abs(observed - predicted))),
        "explained_variance": float(np.mean(explained[nonconstant])) if nonconstant.any() else None,
        "constant_targets": int((~nonconstant).sum()),
    }


def nested_ridge(
    features,
    outcomes,
    participants,
    stimuli,
    participant_folds=5,
    stimulus_folds=3,
    seed=11,
    cancel=lambda: None,
):
    x, y = matrix(features, "features"), matrix(outcomes, "outcomes")
    p, s = np.asarray(participants), np.asarray(stimuli)
    if not len(x) == len(y) == len(p) == len(s):
        raise ValueError("features, outcomes and group identities must align")

    def features_for_fold(_train, indices, _method, _seed):
        return x[indices]

    return nested_feature_ridge(
        features_for_fold, "features", y, p, s, participant_folds, stimulus_folds, seed, cancel
    )


def crossfit_mean(outcomes, participants, stimuli, participant_folds=5, stimulus_folds=3, seed=11):
    y = matrix(outcomes, "outcomes")
    predictions = np.empty_like(y)
    for train, test in crossed_folds(
        participants, stimuli, participant_folds, stimulus_folds, seed
    ):
        predictions[test] = y[train].mean(axis=0)
    return predictions


def cluster_bootstrap(values, participants, stimuli, iterations=10000, seed=11):
    values = np.asarray(values, dtype=float)
    p, s = np.asarray(participants), np.asarray(stimuli)
    if (
        values.ndim != 1
        or len(values) != len(p)
        or len(values) != len(s)
        or not np.isfinite(values).all()
    ):
        raise ValueError(
            "bootstrap values and both cluster identities must be aligned finite vectors"
        )
    people, pidx = np.unique(p, return_inverse=True)
    materials, sidx = np.unique(s, return_inverse=True)
    if min(len(people), len(materials)) < 2 or iterations < 1:
        raise ValueError("bootstrap needs >=2 clusters on both axes and positive iterations")
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(iterations):
        pc = np.bincount(rng.integers(len(people), size=len(people)), minlength=len(people))
        sc = np.bincount(
            rng.integers(len(materials), size=len(materials)), minlength=len(materials)
        )
        weights = pc[pidx] * sc[sidx]
        if weights.sum():
            draws.append(float(np.average(values, weights=weights)))
    if len(draws) < iterations * 0.9:
        raise ValueError(
            "too many empty bootstrap intersections; sparse crossed design is unsupported"
        )
    lower, upper = np.quantile(draws, [0.025, 0.975])
    return {
        "estimate": float(values.mean()),
        "lower": float(lower),
        "upper": float(upper),
        "iterations": iterations,
        "valid_draws": len(draws),
        "seed": seed,
    }


def holm(p_values):
    p = np.asarray(p_values, dtype=float)
    if p.shape != (3,) or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError(
            "Holm correction requires exactly the three prespecified modality p-values"
        )
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    adjusted[order] = np.minimum(1.0, np.maximum.accumulate(p[order] * np.arange(3, 0, -1)))
    return adjusted


def temporal_null(observed, predicted, series, iterations=1000, seed=11, *, metric="mse"):
    y, pred = matrix(observed, "observed"), matrix(predicted, "predicted")
    series = np.asarray(series)
    if y.shape != pred.shape or len(series) != len(y) or iterations < 1:
        raise ValueError("temporal null inputs must align and iterations must be positive")
    groups = [np.flatnonzero(series == item) for item in np.unique(series)]
    if any(len(group) < 3 for group in groups):
        raise ValueError("each temporal series needs at least three ordered observations")
    if metric not in {"mse", "mae", "explained_variance"}:
        raise ValueError("unknown temporal-null metric")
    statistic = scores(y, pred)[metric]
    if statistic is None:
        raise ValueError("temporal-null statistic is undefined for constant targets")
    rng, extreme = np.random.default_rng(seed), 0
    for _ in range(iterations):
        shifted = np.empty_like(y)
        for group in groups:
            shifted[group] = np.roll(y[group], int(rng.integers(1, len(group))), axis=0)
        null_value = scores(shifted, pred)[metric]
        extreme += (
            null_value >= statistic if metric == "explained_variance" else null_value <= statistic
        )
    return {
        "p_value": (extreme + 1) / (iterations + 1),
        "iterations": iterations,
        "seed": seed,
        "metric": metric,
        "statistic": statistic,
    }


def calibrate_gains(settings, evaluate_training):
    from .spec import GAINS

    if not 1 <= len(settings) <= 32:
        raise ValueError("gain calibration is capped at 32 prespecified settings")
    losses = []
    for setting in settings:
        if set(setting) != set(GAINS) or any(not np.isfinite(v) or v < 0 for v in setting.values()):
            raise ValueError("each setting must contain the six finite nonnegative global gains")
        loss = float(evaluate_training(setting))
        if not np.isfinite(loss):
            raise ValueError("training calibration loss is not finite")
        losses.append(loss)
    best = int(np.argmin(losses))
    return {
        "gains": settings[best],
        "training_loss": losses[best],
        "settings_evaluated": len(settings),
    }


def nested_feature_ridge(
    builder,
    method,
    outcomes,
    participants,
    stimuli,
    participant_folds=5,
    stimulus_folds=3,
    seed=11,
    cancel=lambda: None,
    bold=False,
):
    from .observations import fit_bold_observation

    y = matrix(outcomes, "outcomes")
    p, s = np.asarray(participants), np.asarray(stimuli)
    predictions, alphas = np.empty_like(y), []
    for train, test in crossed_folds(p, s, participant_folds, stimulus_folds, seed):
        cancel()
        if bold and method != "conditions" and method != "stimulus":
            predictions[test], _ = fit_bold_observation(
                builder(train, train, method, seed), y[train], builder(train, test, method, seed)
            )
            continue
        inner = list(
            crossed_folds(
                p[train],
                s[train],
                min(3, len(np.unique(p[train]))),
                min(3, len(np.unique(s[train]))),
                seed,
            )
        )
        losses = np.zeros(len(RIDGE_ALPHAS))
        for fit, valid in inner:
            fitting, validation = train[fit], train[valid]
            xfit, xvalid = (
                builder(fitting, fitting, method, seed),
                builder(fitting, validation, method, seed),
            )
            for index, alpha in enumerate(RIDGE_ALPHAS):
                losses[index] += np.mean(
                    (y[validation] - ridge_predict(xfit, y[fitting], xvalid, alpha)) ** 2
                )
        alpha = RIDGE_ALPHAS[int(np.argmin(losses))]
        predictions[test] = ridge_predict(
            builder(train, train, method, seed), y[train], builder(train, test, method, seed), alpha
        )
        alphas.append(alpha)
    return {**scores(y, predictions), "predictions": predictions, "alphas": alphas}


def gain_prediction(builder, setting, train, test, outcomes, alpha, seed):
    from .observations import fit_bold_observation

    builder.gains = setting
    xtrain = builder(train, train, "calibrated_simulator", seed)
    xtest = builder(train, test, "calibrated_simulator", seed)
    if builder.spec.protocol == "fmri":
        return fit_bold_observation(xtrain, outcomes[train], xtest)[0]
    return ridge_predict(xtrain, outcomes[train], xtest, alpha)


def gain_inner_losses(builder, settings, outcomes, train, inner, seed):
    losses = np.zeros((len(settings), len(RIDGE_ALPHAS)))
    for setting_index, setting in enumerate(settings):
        for fit, valid in inner:
            for alpha_index, alpha in enumerate(RIDGE_ALPHAS):
                prediction = gain_prediction(
                    builder, setting, train[fit], train[valid], outcomes, alpha, seed
                )
                losses[setting_index, alpha_index] += np.mean(
                    (outcomes[train[valid]] - prediction) ** 2
                )
    return losses


def nested_gain_calibration(
    builder, settings, outcomes, participants, stimuli, stimulus_folds, seed=11, cancel=lambda: None
):
    from .spec import GAINS

    if not 1 <= len(settings) <= 32 or any(set(item) != set(GAINS) for item in settings):
        raise ValueError("calibration needs at most 32 prespecified settings of exactly six gains")
    y, p, s = matrix(outcomes, "outcomes"), np.asarray(participants), np.asarray(stimuli)
    predictions, selected = np.empty_like(y), []
    for train, test in crossed_folds(p, s, 5, stimulus_folds, seed):
        cancel()
        inner = list(
            crossed_folds(
                p[train],
                s[train],
                min(3, len(np.unique(p[train]))),
                min(3, len(np.unique(s[train]))),
                seed,
            )
        )
        losses = gain_inner_losses(builder, settings, y, train, inner, seed)
        setting_index, alpha_index = np.unravel_index(np.argmin(losses), losses.shape)
        setting, alpha = settings[setting_index], RIDGE_ALPHAS[alpha_index]
        predictions[test] = gain_prediction(builder, setting, train, test, y, alpha, seed)
        selected.append(
            {
                "gains": setting,
                "alpha": alpha,
                "training_loss": float(losses[setting_index, alpha_index] / len(inner)),
            }
        )
    builder.gains = None
    return {
        **scores(y, predictions),
        "predictions": predictions,
        "selected": selected,
        "interpretation": "Exploratory six-gain calibration after the frozen primary analysis; gains and ridge alpha selected inside training folds.",
    }


class UndefinedExplainedVariance(ValueError):
    """The selected observations have no target variance."""


def weighted_explained_variance(y, predicted, weights):
    def variance(values):
        mean = np.average(values, axis=0, weights=weights)
        return np.average((values - mean) ** 2, axis=0, weights=weights)

    target = variance(y)
    valid = target > 1e-15
    if not valid.any():
        raise UndefinedExplainedVariance(
            "BOLD explained variance is undefined for constant targets"
        )
    return float(np.mean(1 - variance(y - predicted)[valid] / target[valid]))


def bold_cluster_bootstrap(
    observed, predicted, baseline, participants, stimuli, iterations=10000, seed=11
):
    y = matrix(observed, "BOLD observations")
    people, p = np.unique(participants, return_inverse=True)
    materials, s = np.unique(stimuli, return_inverse=True)
    if min(len(people), len(materials)) < 2:
        raise ValueError("BOLD bootstrap requires >=2 participant and stimulus clusters")
    original_weights = np.ones(len(y))
    estimate = weighted_explained_variance(
        y, predicted, original_weights
    ) - weighted_explained_variance(y, baseline, original_weights)
    rng, draws = np.random.default_rng(seed), []
    empty_draws = undefined_draws = 0
    for _ in range(iterations):
        pc = np.bincount(rng.integers(len(people), size=len(people)), minlength=len(people))
        sc = np.bincount(
            rng.integers(len(materials), size=len(materials)), minlength=len(materials)
        )
        weights = pc[p] * sc[s]
        if not weights.sum():
            empty_draws += 1
            continue
        try:
            improvement = weighted_explained_variance(
                y, predicted, weights
            ) - weighted_explained_variance(y, baseline, weights)
        except UndefinedExplainedVariance:
            undefined_draws += 1
            continue
        draws.append(improvement)
    minimum_valid_fraction = 0.9
    if len(draws) < iterations * minimum_valid_fraction:
        raise ValueError(
            "BOLD bootstrap requires at least 90% valid draws; too many empty or constant-target resamples"
        )
    lower, upper = np.quantile(draws, [0.025, 0.975])
    return {
        "estimate": estimate,
        "empty_draws": empty_draws,
        "undefined_draws": undefined_draws,
        "minimum_valid_fraction": minimum_valid_fraction,
        "lower": float(lower),
        "upper": float(upper),
        "iterations": iterations,
        "valid_draws": len(draws),
        "seed": seed,
    }
