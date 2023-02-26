from .ensemble import EnsembleEstimator

ESTIMATOR_REGISTRY = {
    "ensemble": EnsembleEstimator,
}

def create_estimator(cfg, **kwargs):
    """Create estimator from config"""
    estimator_name = cfg["class"]
    if estimator_name not in ESTIMATOR_REGISTRY:
        raise ValueError(f"Unknown estimator : {estimator_name}")
    estimator = ESTIMATOR_REGISTRY[estimator_name](
        network_config=cfg.get("model"),
        optimizer_config = cfg.get("optimizer"),
        **kwargs)
    return estimator
