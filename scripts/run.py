import sys
import os
import copy

# Ensure project root is in sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils import (
    setup_logging,
    DEFAULT_ARG_SCHEMA,
    create_argument_parser,
    process_parsed_args,
    parse_override_arguments,
    merge_configs,
    load_config_file,
    BaseTrainingPipeline,
    logger,
)

# Import pipelines to register them with BaseTrainingPipeline
import src.pipelines.coop
import src.pipelines.cocoop
import src.pipelines.maple
import src.pipelines.apt
import src.pipelines.vife

ARG_SCHEMA = {
    **DEFAULT_ARG_SCHEMA,
    'method': {'type': str, 'help': 'Method/Pipeline name (e.g. CoOp, CoCoOp, MaPLe, APT, ViFE). If omitted, inferred from config filename.'},
}


def parse_args():
    parser = create_argument_parser("Run training/evaluation pipeline", ARG_SCHEMA)
    parsed, unknown = parser.parse_known_args()
    overrides = parse_override_arguments(unknown)
    overrides = process_parsed_args(parsed, ARG_SCHEMA, overrides)
    return parsed, overrides


def infer_method_name(config_path, config):
    if hasattr(config, 'get') and config.get('method'):
        return config.get('method')

    config_str = str(config_path).lower()
    if 'cocoop' in config_str:
        return 'CoCoOp'
    elif 'coop' in config_str:
        return 'CoOp'
    elif 'maple' in config_str:
        return 'MaPLe'
    elif 'vife' in config_str:
        return 'ViFE'
    elif 'apt' in config_str:
        return 'APT'
    raise ValueError(f"Could not infer method name from config path: '{config_path}'. Please specify --method.")


def main():
    args, overrides = parse_args()
    setup_logging(getattr(args, 'debug', True), getattr(args, 'disable_coloring', False))

    base_config = load_config_file(args.config)
    merged = merge_configs(base_config, overrides)

    method_name = getattr(args, 'method', None) or infer_method_name(args.config, merged)
    logger.debug(f"Executing pipeline method: {method_name}")

    pipeline_cls = BaseTrainingPipeline.get_pipeline_by_name(method_name)

    from src.pipelines.batch_sweep import run_batch_sweep
    run_batch_sweep(merged, overrides, pipeline_cls)


if __name__ == "__main__":
    main()
