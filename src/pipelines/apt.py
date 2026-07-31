import os
import time
import json
import torch
from torch.utils.data import Subset, DataLoader
from utils import BaseTrainingPipeline, get_config_value, visualize_attention_maps, visualize_gradcam_maps
from src.models.apt import APT


class APTTrainingPipeline(BaseTrainingPipeline):
    METHOD_NAME = "APT"
    DEFAULT_OUTPUT_DIR = "outputs/apt"
    DEFAULT_CHECKPOINT_DIR = "checkpoints/apt"
    TRAINER_CLASS = APT

    def __init__(self, config):
        super().__init__(config)
        self.sample_cache = {
            'images': None, 'labels': None, 'paths': [], 'decoded_prompts': None
        }

    def _run_epoch(self, epoch_idx, epochs_total, train_loader, run_dir):
        super()._run_epoch(epoch_idx, epochs_total, train_loader, run_dir)
        epoch_dir = os.path.join(run_dir, f'epoch_{epoch_idx:03d}')
        all_labels = self.metrics[-1].get('true_labels', [])

        self._refresh_sample_cache(all_labels)

        if bool(get_config_value(self.training_cfg, 'visualize_attention', False)):
            attention_dir = os.path.join(epoch_dir, 'attention')
            os.makedirs(attention_dir, exist_ok=True)
            self._export_attention_overlays(attention_dir)

        if bool(get_config_value(self.training_cfg, 'visualize_gradcam', False)):
            gradcam_dir = os.path.join(epoch_dir, 'gradcam')
            os.makedirs(gradcam_dir, exist_ok=True)
            self._export_gradcam_overlays(gradcam_dir)

    def _refresh_sample_cache(self, all_labels):
        if self.dataset is None:
            return
        if self.val_loader is None or len(self.val_indices) == 0:
            return

        num_display = min(10, len(self.classnames), len(self.val_indices))
        selected_indices = []
        seen_classes = set()
        for idx in self.val_indices:
            cls_idx = self.dataset.samples[idx][1]
            if cls_idx not in seen_classes:
                seen_classes.add(cls_idx)
                selected_indices.append(idx)
            if len(selected_indices) >= num_display:
                break

        if len(selected_indices) == 0:
            try:
                batch_data = next(iter(self.val_loader))
                if isinstance(batch_data, (list, tuple)) and len(batch_data) >= 2:
                    self.sample_cache['images'] = batch_data[0]
                    self.sample_cache['labels'] = batch_data[1]
                    batch_indices = self.val_indices[:len(batch_data[0])]
                    self.sample_cache['paths'] = [os.path.abspath(self.dataset.samples[idx][0]) for idx in batch_indices]
                else:
                    self.sample_cache['images'] = batch_data
                    self.sample_cache['labels'] = None
                    self.sample_cache['paths'] = []
            except StopIteration:
                self.sample_cache['images'] = None
                self.sample_cache['labels'] = None
                self.sample_cache['paths'] = []
        else:
            sample_images_list = []
            sample_labels_list = []
            sample_paths = []
            for idx in selected_indices:
                img, lbl = self.dataset[idx]
                sample_images_list.append(img)
                sample_labels_list.append(lbl)
                sample_paths.append(os.path.abspath(self.dataset.samples[idx][0]))

            self.sample_cache['images'] = torch.stack(sample_images_list)
            self.sample_cache['labels'] = torch.tensor(sample_labels_list)
            self.sample_cache['paths'] = sample_paths

    def _export_attention_overlays(self, maps_dir):
        if self.trainer is None:
            return
        visualize_attention_maps(
            self.trainer,
            self.dataset,
            self.sample_cache,
            self.classnames,
            self.global_epoch,
            maps_dir,
        )

    def _export_gradcam_overlays(self, maps_dir):
        if self.trainer is None:
            return
        visualize_gradcam_maps(
            self.trainer,
            self.dataset,
            self.sample_cache,
            self.classnames,
            self.global_epoch,
            maps_dir,
        )
