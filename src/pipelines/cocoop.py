from utils import BaseTrainingPipeline
from src.models.cocoop import CoCoOP


class CoCoOPTrainingPipeline(BaseTrainingPipeline):
    METHOD_NAME = "CoCoOp"
    DEFAULT_OUTPUT_DIR = "outputs/cocoop"
    DEFAULT_CHECKPOINT_DIR = "checkpoints/cocoop"
    TRAINER_CLASS = CoCoOP
