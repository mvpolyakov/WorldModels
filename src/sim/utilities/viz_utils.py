from pathlib import Path

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure


def save_displacement_error_chart(error_values, path):
    """Save cumulative summed displacement error as a PNG line chart."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure = Figure(figsize=(8, 5))
    FigureCanvasAgg(figure)
    axes = figure.subplots()
    axes.plot(range(1, len(error_values) + 1), error_values)
    axes.set_xlabel("Simulation step")
    axes.set_ylabel("Cumulative displacement error (distance × steps)")
    axes.set_title("Cumulative Displacement Error")
    axes.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)

    return output_path
