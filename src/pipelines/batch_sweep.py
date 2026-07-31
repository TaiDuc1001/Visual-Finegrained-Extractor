import copy
import logging
import torch
import torch.nn as nn
import time
from utils import (
    get_config_value,
    iter_dataset_configs,
    logger,
    set_global_seed,
    coerce_to_float,
)

DEFAULT_KSHOTS = [1, 2, 4, 8, 16]
DEFAULT_SEEDS = [1, 10, 100, 1000, 10000]


def _parse_int_list(value, default):
    if value is None:
        return list(default)
    if isinstance(value, str):
        return [int(part.strip()) for part in value.split(",") if part.strip()]
    if isinstance(value, (list, tuple)):
        return [int(item) for item in value]
    return [int(value)]


def _has_config_path(config, path):
    current = config
    for key in path.split("."):
        if not isinstance(current, dict) or key not in current:
            return False
        current = current[key]
    return True


def sweep_values(config, overrides):
    kshots = _parse_int_list(
        get_config_value(config, "data.kshots", None),
        [get_config_value(config, "data.kshot")]
        if _has_config_path(overrides, "data.kshot")
        else DEFAULT_KSHOTS,
    )
    seeds = _parse_int_list(
        get_config_value(config, "data.seeds", None),
        [get_config_value(config, "data.seed")]
        if _has_config_path(overrides, "data.seed")
        else DEFAULT_SEEDS,
    )
    return kshots, seeds


def _reset_trainer_for_run(pipeline, kshot, seed):
    trainer = pipeline.trainer
    if trainer is None:
        raise RuntimeError("Trainer must be initialized before a batch run.")

    # Reset model weights
    trainer.reset_model()
    # Reset optimizer and scheduler
    trainer.reset_optimizer_scheduler()

    # Update config values
    trainer.cfg.setdefault("data", {})["kshot"] = int(kshot)
    trainer.cfg.setdefault("data", {})["seed"] = int(seed)
    trainer.data_cfg = trainer.cfg.get("data", trainer.data_cfg)

    # Reset any specific attributes on pipeline for ViFE SSL
    if hasattr(pipeline, "ssl_student"):
        pipeline.ssl_student = None
    if hasattr(pipeline, "ssl_teacher"):
        pipeline.ssl_teacher = None
    if hasattr(pipeline, "ssl_classifier"):
        pipeline.ssl_classifier = None
    if hasattr(pipeline, "ssl_center"):
        pipeline.ssl_center = None
    if hasattr(pipeline, "fusion_weights"):
        pipeline.fusion_weights = None
    if hasattr(pipeline, "cached_apt_predictions"):
        pipeline.cached_apt_predictions = None
    if hasattr(pipeline, "dual_branch_eval_result"):
        pipeline.dual_branch_eval_result = None
    if hasattr(pipeline, "learned_acc"):
        pipeline.learned_acc = None


def _prepare_shared_pipeline(pipeline_cls, config, kshots, seeds):
    run_config = copy.deepcopy(config)
    data_cfg = run_config.setdefault("data", {})
    data_cfg["kshot"] = int(max(kshots))
    data_cfg["seed"] = int(seeds[0])
    data_cfg["run_eda"] = False
    run_config.setdefault("checkpoint", {})["enabled"] = False
    run_config.setdefault("logging", {})["summary_only"] = True

    set_global_seed(int(seeds[0]))
    pipeline = pipeline_cls(run_config)
    pipeline._prepare_directories()
    pipeline._load_dataset()
    pipeline._split_dataset()
    pipeline._initialize_trainer()
    return pipeline


def _run_once(pipeline, kshot, seed):
    set_global_seed(int(seed))
    pipeline.kshot = int(kshot)
    pipeline.seed = int(seed)
    pipeline.config.setdefault("data", {})["kshot"] = int(kshot)
    pipeline.config.setdefault("data", {})["seed"] = int(seed)
    pipeline.data_cfg["kshot"] = int(kshot)
    pipeline.data_cfg["seed"] = int(seed)
    
    pipeline._split_dataset()
    _reset_trainer_for_run(pipeline, kshot, seed)

    pipeline.metrics = []
    pipeline.best_val_acc = -float("inf")
    pipeline.global_epoch = 0
    
    # Train APT/base model
    pipeline._train_epochs()

    # Train SSL stages if ViFE pipeline
    if getattr(pipeline, "use_ssl", False):
        enable_stage1 = pipeline.ssl_cfg.get('enable_stage1', True)
        enable_stage2 = pipeline.ssl_cfg.get('enable_stage2', True)
        enable_stage3 = pipeline.ssl_cfg.get('enable_stage3', True)
        
        if enable_stage1:
            if pipeline.ssl_cfg.get('save_stage1_checkpoint', False) and pipeline._try_load_ssl_stage1_checkpoint():
                pass
            else:
                pipeline._train_ssl_stage1()
        
        if enable_stage2:
            pipeline._train_ssl_stage2()
            
        if enable_stage3 and pipeline.ssl_cfg.get('learn_fusion', False):
            pipeline._train_ssl_stage3()
            
        eval_result = pipeline._run_dual_branch_eval()
        pipeline.dual_branch_eval_result = eval_result
        if enable_stage3:
            pipeline.learned_acc = eval_result.get('learned_acc') if eval_result else None
        else:
            default_weight = coerce_to_float(pipeline.ssl_cfg.get('default_fusion_weight', 0.5), 0.5)
            if eval_result and eval_result.get('fusion_results'):
                pipeline.learned_acc = eval_result['fusion_results'].get(default_weight)
            else:
                pipeline.learned_acc = eval_result.get('apt_acc') if eval_result else None

    # Get final metrics
    if not pipeline.metrics:
        raise RuntimeError(
            f"{pipeline.METHOD_NAME} produced no metrics for "
            f"{kshot}-shot seed {seed}."
        )
    
    result = dict(pipeline.metrics[-1])
    if getattr(pipeline, "use_ssl", False) and getattr(pipeline, "learned_acc", None) is not None:
        result["accuracy"] = pipeline.learned_acc
        
    return result


def run_dataset_sweep(pipeline_cls, config, kshots, seeds):
    pipeline = _prepare_shared_pipeline(
        pipeline_cls,
        config,
        kshots,
        seeds,
    )
    results = {}
    for kshot in kshots:
        for seed in seeds:
            start_time = time.time()
            res = _run_once(
                pipeline,
                int(kshot),
                int(seed),
            )
            elapsed = time.time() - start_time
            acc = res.get('accuracy', 0.0)
            dataset_name = config.get("data", {}).get("dataset_name", "Unknown")
            method_name = pipeline.METHOD_NAME
            print(f"[{method_name}] {dataset_name} | {kshot}-shot | seed {seed} | Accuracy: {acc:.2f}% ({elapsed:.1f}s)", flush=True)
            results[(int(kshot), int(seed))] = res
    return results


def print_accuracy_variance_report(dataset_name, method_name, results, kshots, seeds, backbone):
    print("\n" + "=" * 80, flush=True)
    print(f"METHOD ACCURACY REPORT (Backbone: {backbone} | Dataset: {dataset_name})", flush=True)
    print("=" * 80, flush=True)
    print(f"{'Method':<16} {'Backbone':<12} {'Shot':<5} {'Avg Acc (%)':<12} {'Std (%)':<12}", flush=True)
    print("-" * 80, flush=True)
    
    for kshot in kshots:
        accs = [float(results[(int(kshot), int(seed))].get('accuracy', 0.0)) for seed in seeds]
        if not accs:
            continue
        avg_acc = float(torch.tensor(accs).mean().item())
        std_val = float(torch.tensor(accs).std(unbiased=False).item()) if len(accs) > 1 else 0.0
        print(f"{method_name:<16} {backbone:<12} {kshot:<5} {avg_acc:<12.2f} {std_val:<12.2f}", flush=True)
    print("=" * 80 + "\n", flush=True)


def run_batch_sweep(config, overrides, pipeline_cls):
    previous_level = logger.level
    logger.setLevel(logging.WARNING)
    try:
        dataset_configs = list(iter_dataset_configs(config))
        outputs = []
        for dataset_config, _ in dataset_configs:
            kshots, seeds = sweep_values(dataset_config, overrides)
            results = run_dataset_sweep(
                pipeline_cls,
                dataset_config,
                kshots,
                seeds,
            )
            dataset_name = str(dataset_config["data"]["dataset_name"])
            first_result = next(iter(results.values()), {})
            method_name = str(first_result.get("method", pipeline_cls.METHOD_NAME))
            backbone = dataset_config.get("model", {}).get("backbone", "ViT-B/16")
            
            is_singularity = (len(seeds) == 1 and len(kshots) == 1)
            if not is_singularity:
                print_accuracy_variance_report(
                    dataset_name,
                    method_name,
                    results,
                    kshots,
                    seeds,
                    backbone,
                )
            outputs.append(results)
        return outputs
    finally:
        logger.setLevel(previous_level)
