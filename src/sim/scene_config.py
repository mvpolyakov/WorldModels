import json
import os
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
        )


def load_scene_config(config_path_or_file_name: str = "simple_rolling.json") -> SceneConfig:
    """Load a config from a path or from the built-in config directory."""

    if os.path.exists(config_path_or_file_name):
        return SceneConfig.from_json(config_path_or_file_name)

    config_file_name = (
        config_path_or_file_name
        if config_path_or_file_name.endswith(".json")
        else f"{config_path_or_file_name}.json"
    )
    config_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "configs",
        config_file_name,
    )
    return SceneConfig.from_json(config_path)
