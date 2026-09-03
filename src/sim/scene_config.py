import json
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple, Any


@dataclass
class ObjectConfig:
    """A single scene object (fixed or mobile).

    Attributes:
        name:           Identifier for this object.
        morph_type:     Genesis morph class name: "Box", "Sphere", "Mesh", etc.
        morph_params:   Kwargs passed to gs.morphs.<morph_type>(...).
        friction:       Surface friction coefficient.
        friction_torsional: Torsional friction coefficient in metres.
        friction_rolling: Rolling friction coefficient in metres.
        rho:            Rigid body density (kg/m^3).
        color:          RGBA surface colour.
        texture:        Optional image path for the diffuse surface texture.
    """
    name: str = ""
    morph_type: str = "Box"
    morph_params: Dict[str, Any] = field(default_factory=dict)
    friction: float = 1
    friction_torsional: float = 0.005
    friction_rolling: float = 0.0001
    rho: float = 1000.0
    color: Tuple[float, float, float, float] = (0.5, 0.5, 0.5, 1.0)
    texture: str | None = None


@dataclass
class CameraConfig:
    # World / viewer camera
    viewer_pos: Tuple[float, float, float] = (0.0, -4.0, 1.5)
    viewer_lookat: Tuple[float, float, float] = (0.0, 0.0, 0.5)
    viewer_fov: float = 45.0
    # Observation camera (rendered to tensor, lower resolution)
    obs_pos: Tuple[float, float, float] = (0.0, -4.0, 1.5)
    obs_lookat: Tuple[float, float, float] = (0.0, 0.0, 0.5)
    obs_res: Tuple[int, int] = (320, 240)
    obs_fov: float = 45.0


@dataclass
class SceneConfig:
    landscape: List[ObjectConfig] = field(default_factory=list)
    mobiles: List[ObjectConfig] = field(default_factory=list)
    camera: CameraConfig = field(default_factory=CameraConfig)
    gravity: Tuple[float, float, float] = (0.0, 0.0, -9.81)
    show_viewer: bool = True

    def to_json(self, path: str):
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=4)

    @classmethod
    def from_json(cls, path: str) -> 'SceneConfig':
        with open(path, 'r') as f:
            data = json.load(f)
        
        landscape = [ObjectConfig(**obj) for obj in data.get('landscape', [])]
        mobiles = [ObjectConfig(**obj) for obj in data.get('mobiles', [])]
        camera = CameraConfig(**data.get('camera', {}))
        
        return cls(
            landscape=landscape,
            mobiles=mobiles,
            camera=camera,
            gravity=tuple(data.get('gravity', (0.0, 0.0, -9.81))),
            show_viewer=data.get('show_viewer', True)
        )


# ---------------------------------------------------------------------------
# Pre-built configs
# ---------------------------------------------------------------------------

def _base_config() -> SceneConfig:
    """Ramp + sphere scene."""
    import math
    angle_deg = 25.0
    angle_rad = math.radians(angle_deg)
    ramp_length = 2.0
    ramp_thickness = 0.05

    # Ramp centre so its low end sits at ground level
    low_z = ramp_thickness / 2.0
    centre_z = low_z + (ramp_length / 2.0) * math.sin(angle_rad)

    # Ball at the high end of the ramp with clearance
    ball_radius = 0.2
    ball_x = -(ramp_length / 2.0) * math.cos(angle_rad)
    ball_z = (ramp_thickness / 2.0
              + ramp_length * math.sin(angle_rad)
              + ball_radius + 0.01)

    return SceneConfig(
        landscape=[
            ObjectConfig(
                name="ramp",
                morph_type="Box",
                morph_params=dict(
                    pos=(0.0, 0.0, centre_z),
                    euler=(0.0, -angle_deg, 0.0),
                    size=(ramp_length, 1.0, ramp_thickness),
                    fixed=True,
                ),
                friction=0.4,
                color=(0.6, 0.5, 0.4, 1.0),
            ),
            ObjectConfig(
                name="ground",
                morph_type="Box",
                morph_params=dict(
                    pos=(0.0, 0.0, -0.025),
                    size=(20.0, 20.0, 0.05),
                    fixed=True,
                ),
                friction=1.0,
                friction_rolling=0.005,
                color=(0.3, 0.3, 0.3, 1.0),
            ),
        ],
        mobiles=[
            ObjectConfig(
                name="ball",
                morph_type="Sphere",
                morph_params=dict(
                    pos=(ball_x, 0.0, ball_z),
                    radius=ball_radius,
                ),
                friction=0.3,
                rho=500.0,
                color=(0.2, 0.6, 1.0, 1.0),
            ),
        ],
    )


_scene_configs = {
    'base': _base_config,
}


def load_scene_config(key: str = 'base') -> SceneConfig:
    import os
    if key.endswith('.json') and os.path.exists(key):
        return SceneConfig.from_json(key)
        
    # Also check if it exists in the configs directory automatically
    current_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(current_dir, "configs", f"{key}.json")
    
    if os.path.exists(json_path):
        return SceneConfig.from_json(json_path)
        
    if key not in _scene_configs:
        raise ValueError(f"Scene config '{key}' not found. Available: {list(_scene_configs)}")
    
    scene_generator = _scene_configs[key]
    
    return scene_generator()
