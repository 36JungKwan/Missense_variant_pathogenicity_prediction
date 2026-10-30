"""Shared Optuna helpers for resumable, sequential experiment stages."""
from __future__ import annotations

import re
from pathlib import Path


def _optuna():
    try:
        import optuna
    except ImportError as exc:
        raise ImportError(
            "Optuna is required for the optimization stages. Install it in the notebook kernel with `pip install optuna`."
        ) from exc
    return optuna


def suggest_parameters(trial, space):
    """Sample typed parameters, including conditional and dependent bounds.

    A spec may define ``when`` as a mapping of already-sampled parameter values
    and ``high_power_of_two`` as the name of an integer parameter whose value
    caps the upper bound (used to keep LightGBM leaves compatible with depth).
    """
    params = {}
    for name, spec in space.items():
        if isinstance(spec, dict):
            conditions = spec.get("when", {})
            if any(params.get(key) != value for key, value in conditions.items()):
                continue
            kind = spec["type"]
            if kind == "int":
                low = int(spec["low"])
                high = int(spec["high"])
                if "high_power_of_two" in spec:
                    high = min(high, 2 ** int(params[spec["high_power_of_two"]]))
                params[name] = trial.suggest_int(name, low, high, step=int(spec.get("step", 1)), log=bool(spec.get("log", False)))
            elif kind == "float":
                params[name] = trial.suggest_float(name, float(spec["low"]), float(spec["high"]), step=spec.get("step"), log=bool(spec.get("log", False)))
            elif kind == "categorical":
                params[name] = trial.suggest_categorical(name, list(spec["choices"]))
            else:
                raise ValueError(f"Unsupported Optuna parameter type {kind!r} for {name}")
        elif isinstance(spec, (list, tuple)):
            params[name] = trial.suggest_categorical(name, list(spec))
        else:
            params[name] = spec
    return params


def optuna_trial_plan(stage_dir, study_name, n_trials, search_space, seed=42, startup_trials=8):
    """Yield prior completed trials and new TPE trials, requiring tell() before advancing.

    The SQLite study is the source of truth for trial numbers and resume. Callers
    persist each model/CSV row before ``study.tell(trial, validation_mcc)``.
    """
    optuna = _optuna()
    directory = Path(stage_dir)
    directory.mkdir(parents=True, exist_ok=True)
    database = (directory / "optuna_studies.sqlite3").resolve().as_posix()
    storage = f"sqlite:///{database}"
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", study_name)
    study = optuna.create_study(
        study_name=safe_name,
        storage=storage,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=int(seed), n_startup_trials=int(startup_trials)),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=int(startup_trials), n_warmup_steps=5),
        load_if_exists=True,
    )

    # A notebook/kernel interruption can leave a RUNNING record. Retry its exact
    # parameters as a new trial so no pending trial silently consumes the budget.
    running = [t for t in study.trials if t.state == optuna.trial.TrialState.RUNNING]
    for old_trial in running:
        if old_trial.params:
            study.enqueue_trial(old_trial.params)
        study.tell(old_trial.number, state=optuna.trial.TrialState.FAIL)

    completed = sorted(
        (t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE),
        key=lambda t: t.number,
    )
    for trial in completed:
        yield study, trial, dict(trial.params), False

    terminal_count = sum(
        t.state in (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED)
        for t in study.get_trials(deepcopy=False)
    )
    while terminal_count < int(n_trials):
        trial = study.ask()
        params = suggest_parameters(trial, search_space)
        yield study, trial, params, True
        state = next(t.state for t in study.get_trials(deepcopy=False) if t.number == trial.number)
        if state in (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED):
            terminal_count += 1
        elif state == optuna.trial.TrialState.RUNNING:
            raise RuntimeError(f"Optuna trial {trial.number} was not told before requesting another trial")


def tell_trial(study, trial, value):
    """Finish a newly evaluated trial with its validation objective."""
    study.tell(trial, float(value))


def prune_trial(study, trial):
    """Mark an unpromising trial as pruned and continue the configured budget."""
    optuna = _optuna()
    study.tell(trial, state=optuna.trial.TrialState.PRUNED)
