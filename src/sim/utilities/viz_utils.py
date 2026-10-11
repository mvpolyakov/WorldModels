from pathlib import Path

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from ..scene_config import SceneConfig


def draw_ground_coordinate_lines(scene, scene_config: SceneConfig) -> None:
    """Draw fixed world X and Y coordinate lines across the configured ground.

    @param scene: Genesis scene on which to draw the debug lines.
    @param scene_config: Scene configuration containing the ground box geometry - 
        since the ground is actually a box for friction need its z coord 
    """
    ground_config = next(
        obj for obj in scene_config.landscape if obj.name == "ground"
    )
    ground_size = ground_config.morph_params["size"]
    ground_position = ground_config.morph_params["pos"]
    line_height = ground_position[2] + ground_size[2] / 2.0 + 0.01

    scene.draw_debug_line(
        (-ground_size[0] / 2.0, 0.0, line_height),
        (ground_size[0] / 2.0, 0.0, line_height),
        radius=0.006,
        color=(1.0, 0.1, 0.1, 1.0),
    )
    scene.draw_debug_line(
        (0.0, -ground_size[1] / 2.0, line_height),
        (0.0, ground_size[1] / 2.0, line_height),
        radius=0.006,
        color=(0.1, 1.0, 0.1, 1.0),
    )


def save_displacement_error_chart(error_values, path):
    """Save cumulative summed XYZ displacement error as a PNG line chart."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure = Figure(figsize=(8, 5))
    FigureCanvasAgg(figure)
    axes = figure.subplots()
    axes.plot(range(1, len(error_values) + 1), error_values)
    axes.set_xlabel("Simulation step")
    axes.set_ylabel("Cumulative XYZ displacement error (distance × steps)")
    axes.set_title("Cumulative XYZ Displacement Error")
    axes.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)

    return output_path


def save_position_mse_chart(mse_values, path):
    """Save per-step XYZ position MSE and annotate prediction step ten."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure = Figure(figsize=(8, 5))
    FigureCanvasAgg(figure)
    axes = figure.subplots()
    axes.plot(range(1, len(mse_values) + 1), mse_values)
    if len(mse_values) >= 10:
        axes.annotate(
            f"Step 10 MSE: {mse_values[9]:.6f}",
            xy=(10, mse_values[9]),
            xytext=(8, 8),
            textcoords="offset points",
        )
    axes.set_xlabel("Prediction step")
    axes.set_ylabel("XYZ position MSE")
    axes.set_title("XYZ Position MSE by Prediction Step")
    axes.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)

    return output_path
