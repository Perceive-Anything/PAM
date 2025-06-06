import os
from .sam2_encoder import SAM2VisionTower

def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, "mm_vision_tower", getattr(vision_tower_cfg, "vision_tower", None))
    is_absolute_path_exists = os.path.exists(vision_tower)
    # FIXME
    if "sam2" in vision_tower:
        return SAM2VisionTower(vision_tower, **kwargs)

    raise ValueError(f"Unknown vision tower: {vision_tower}")