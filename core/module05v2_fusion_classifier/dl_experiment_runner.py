"""Run the complete pure-DL experiment sequence and collect its results."""
from .dl_grid_search import run_pure_dl_optuna
from .dl_architecture_variants_v2 import run_architecture_stage as _run_variants_v2
from .dl_architecture_variants_v3 import run_architecture_stage as _run_variants_v3
from .dl_architecture_search_v1 import run_architecture_stage as _run_broad_search
from .dl_architecture_hparam_search import run_architecture_stage as _run_hparam_search
from .dl_seed_confirmation import run_seed_confirmation
from .dl_result_collector import collect_dl_results


def run_all_dl_experiments(context, search_config, display_fn=None):
    """Run/reuse each DL stage in order, then return detailed tables and top 3."""
    run_pure_dl_optuna(context, search_config, "v1", display_fn)
    run_pure_dl_optuna(context, search_config, "v2", display_fn)
    _run_variants_v2(context, search_config, display_fn)
    _run_variants_v3(context, search_config, display_fn)
    _run_broad_search(context, search_config, display_fn)
    _run_hparam_search(context, search_config, display_fn)
    run_seed_confirmation(context, search_config, display_fn)
    return collect_dl_results(context, search_config, display_fn)
