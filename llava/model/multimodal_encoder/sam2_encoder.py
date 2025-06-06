import os
import numpy as np
import torch
from torch import nn
from typing import Optional, Tuple, Union, Dict
from transformers import PretrainedConfig
from sam2.build_sam import build_sam2, build_sam2_video_predictor
from llava.utils import rank0_print

class SAM2VisionConfig(PretrainedConfig):
    """
    Configuration class for the SAM2VisionTower.

    Attributes:
        hidden_size (int): The hidden size of the vision features.
        image_size (int): The expected input image size.
        patch_size (int): The patch size for the vision model.
    """
    def __init__(
        self,
        hidden_size: int = 256,
        image_size: int = 1024,
        patch_size: int = 64,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.patch_size = patch_size
        self.image_size = image_size

class SAM2VisionTower(nn.Module):
    """
    Vision tower for SAM2 model, responsible for processing images/videos
    and generating segmentation masks and features.
    """
    def __init__(self, vision_tower: str, delay_load: bool = False):
        super().__init__()

        self.is_loaded = False
        self.config = SAM2VisionConfig()
        self.vision_tower_name = vision_tower # Storing the name for better logging

        current_dir = os.path.dirname(__file__)
        self.model_cfg_path = "configs/sam2.1/sam2.1_hiera_l.yaml"
        self.sam2_checkpoint_path = os.path.join(current_dir, "sam2.1_hiera_large.pt")
        self.sam2_model_predictor = None
        self.hidden_size = self.config.hidden_size # Use hidden_size from config

        if not delay_load:
            rank0_print(f"Loading vision tower: {self.vision_tower_name}")
            self.load_model()
        else:
            # TODO: better detector is needed.
            self.cfg_only = self.config

    def load_model(self, device_map: Optional[Union[str, torch.device]] = None):
        """
        Loads the SAM2 predictor model.
        """
        if self.is_loaded:
            rank0_print(f"{self.vision_tower_name} is already loaded, `load_model` called again, skipping.")
            return

        try:
            self.sam2_model_predictor = build_sam2_video_predictor(
                self.model_cfg_path, self.sam2_checkpoint_path, device=device_map
            )
            self.sam2_model_predictor.requires_grad_(False)
            self.sam2_model_predictor.eval()
            self.is_loaded = True
        except Exception as e:
            rank0_print(f"Error loading SAM2 model: {e}")
            raise

    def _prepare_visual_prompts(self, visual_prompt: np.ndarray) -> Dict:
        """
        Prepares the visual prompt arguments based on its dimension.
        """
        vp_array = np.asarray(visual_prompt)
        if vp_array.ndim == 1 and vp_array.size == 4:
            return {'box': vp_array}
        else:
            return {'points': vp_array, 'labels': np.asarray(visual_prompt)} # Using visual_prompt for labels here, adjust if labels come separately

    def _process_frame_output(self, inference_state: Dict, frame_type: str) -> Dict:
        """
        Extracts and processes output features from an inference state for a given frame type.
        """
        if frame_type == 'cond':
            output_dict = inference_state['temp_output_dict_per_obj'][0]['cond_frame_outputs'][0]
        elif frame_type == 'non_cond':
            output_dict = inference_state['output_dict_per_obj'][0]['non_cond_frame_outputs']
        else:
            raise ValueError(f"Unknown frame type: {frame_type}")

        model_dtype = next(self.sam2_model_predictor.parameters()).dtype
        return {
            'score': output_dict['object_score_logits'].detach().float().cpu().numpy(),
            'mid_image_embed': output_dict['mid_image_embed'].clone().detach().to(model_dtype),
            'mid_mask_tokens_out': output_dict['mid_mask_tokens_out'].clone().detach().to(model_dtype)
        }

    def forward(self, images: list, visual_prompts: list, vp_labels: list):
        """
        Performs inference on a list of images (single-frame processing).

        Args:
            images (list): A list of image paths.
            visual_prompts (list): A list of visual prompts (boxes or points).
            vp_labels (list): A list of labels corresponding to visual prompts (for points).

        Returns:
            tuple: A tuple containing concatenated mid_image_embeds, mid_mask_tokens,
                   a list of mask logits, and a list of scores.
        """
        if not self.is_loaded:
            raise RuntimeError("SAM2 model is not loaded. Call `load_model()` first.")
        if not isinstance(images, list):
            raise TypeError("Images input currently only supports list type.")

        model_dtype = next(self.sam2_model_predictor.parameters()).dtype
        
        all_masks_logits = []
        all_scores = []
        all_mid_image_embed = []
        all_mid_mask_tokens_out = []

        for img, vp, label in zip(images, visual_prompts, vp_labels):
            inference_state = self.sam2_model_predictor.init_state(video_path=[img], dtype=model_dtype)
            
            args = self._prepare_visual_prompts(vp)
            if 'points' in args: # If points, ensure labels are correctly assigned
                args['labels'] = np.asarray(label) # Use provided labels for points

            *_, mask_logits = self.sam2_model_predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=0,
                obj_id=1,
                **args,
                dtype=model_dtype
            )
            
            output_data = self._process_frame_output(inference_state, 'cond')
            all_masks_logits.append(mask_logits)
            all_scores.append(output_data['score'])
            all_mid_image_embed.append(output_data['mid_image_embed'])
            all_mid_mask_tokens_out.append(output_data['mid_mask_tokens_out'])
            
            self.sam2_model_predictor.reset_state(inference_state)

        mid_image_embeds = torch.cat(all_mid_image_embed, dim=0)
        mid_mask_tokens = torch.cat(all_mid_mask_tokens_out, dim=0)
        
        return mid_image_embeds, mid_mask_tokens, all_masks_logits, all_scores
        

    def forward_video(self, frames: list, visual_prompts: list, vp_labels: list):
        """
        Performs inference on a list of video frames, propagating segmentation through time.

        Args:
            frames (list): A list of video frame paths
            visual_prompts (list): A list of visual prompts (boxes or points) for the first frame of each video.
            vp_labels (list): A list of labels corresponding to visual prompts (for points).

        Returns:
            tuple: A tuple containing all_mid_image_embed (dict of tensors per frame),
                   all_mid_mask_tokens_out (dict of tensors per frame),
                   all_masks_logits (list of dicts of binary masks per frame),
                   all_scores (list of dicts of scores per frame).
        """
        if not self.is_loaded:
            raise RuntimeError("SAM2 model is not loaded. Call `load_model()` first.")
        if not isinstance(frames, list):
            raise TypeError("Frames input currently only supports list type.")
        
        model_dtype = next(self.sam2_model_predictor.parameters()).dtype

        results = {
            'all_masks_logits': [],
            'all_scores': [],
            'all_mid_image_embed': [],
            'all_mid_mask_tokens_out': []
        }

        for bs1_frames, vp, label in zip(frames, visual_prompts, vp_labels):
            video_segments = {}  # per-frame segmentation results (binary mask)
            dic_scores = {}
            dic_mid_image_embed = {}
            dic_mid_mask_tokens_out = {}
            
            inference_state = self.sam2_model_predictor.init_state(video_path=bs1_frames, dtype=model_dtype)
            
            args = self._prepare_visual_prompts(vp)
            if 'points' in args:
                args['labels'] = np.asarray(label) # Use provided labels for points

            *_, first_frame_out_mask_logits = self.sam2_model_predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=0,
                obj_id=1,
                **args,
                dtype=model_dtype
            )
            
            # Process first (conditional) frame
            first_frame_output = self._process_frame_output(inference_state, 'cond')
            dic_scores[0] = first_frame_output['score']
            dic_mid_image_embed[0] = first_frame_output['mid_image_embed']
            dic_mid_mask_tokens_out[0] = first_frame_output['mid_mask_tokens_out']

            # Propagate through video
            for out_frame_idx, out_obj_ids, out_mask_logits in self.sam2_model_predictor.propagate_in_video(inference_state):
                video_segments[out_frame_idx] = {
                    out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                    for i, out_obj_id in enumerate(out_obj_ids)
                }
                if out_frame_idx != 0:
                    non_cond_output = inference_state["output_dict_per_obj"][0]["non_cond_frame_outputs"][out_frame_idx]
                    dic_scores[out_frame_idx] = non_cond_output['object_score_logits'].detach().float().cpu().numpy()
                    dic_mid_image_embed[out_frame_idx] = non_cond_output['mid_image_embed'].clone().detach().to(model_dtype)
                    dic_mid_mask_tokens_out[out_frame_idx] = non_cond_output['mid_mask_tokens_out'].clone().detach().to(model_dtype)

            results['all_masks_logits'].append(video_segments)
            results['all_scores'].append(dic_scores)
            results['all_mid_image_embed'].append(dic_mid_image_embed)
            results['all_mid_mask_tokens_out'].append(dic_mid_mask_tokens_out)

            self.sam2_model_predictor.reset_state(inference_state)
            
        return results['all_mid_image_embed'], results['all_mid_mask_tokens_out'], results['all_masks_logits'], results['all_scores']

    @property
    def dummy_feature(self) -> torch.Tensor:
        """
        Returns a dummy feature tensor.
        """
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self) -> torch.dtype:
        """
        Returns the data type of the model parameters.
        """
        if self.sam2_model_predictor:
            return next(self.sam2_model_predictor.parameters()).dtype
        return torch.float32 # Default if model not loaded

    @property
    def device(self) -> torch.device:
        """
        Returns the device of the model parameters.
        """
        if self.sam2_model_predictor:
            return next(self.sam2_model_predictor.parameters()).device
        return torch.device("cpu") # Default if model not loaded
