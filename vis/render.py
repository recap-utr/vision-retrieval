from srip import SRIP1, SRIP2, SRIP_Config, default_weight
from enum import Enum
import arguebuf as ab
from PIL import Image
import typer
from pathlib import Path
from logical import render as logical_render
from treemaps import visualize_treemap_inmem

app = typer.Typer()


class RenderMethod(str, Enum):
    SRIP1 = "srip1"
    SRIP2 = "srip2"
    LOGICAL = "logical"
    TREEMAP = "treemap"


@app.command()
def render_command(
    graph_path: Path,
    output_path: Path,
    method: RenderMethod = RenderMethod.SRIP2,
    dpi: int = 100,
    normalize_height: bool = False,
) -> None:
    graph = ab.load.file(graph_path)
    render(graph, output_path, normalize_height, method, dpi)


def render(
    graph: ab.Graph,
    output_path: Path,
    normalize_height: bool,
    method: RenderMethod = RenderMethod.SRIP2,
    dpi: int = 100,
) -> None:
    roots = graph.root_nodes
    if method == RenderMethod.SRIP2:
        config = SRIP_Config()
        config.W = 10 / len(roots)
        config.epsilon = config.W
        config.dpi = dpi
        config.normalize_height = normalize_height
        images = [SRIP2(root, graph, default_weight, config=config) for root in roots]
        image = images[0]
        if len(images) > 1:
            image = _concat_images(images)
        image.save(output_path)
    elif method == RenderMethod.SRIP1:
        config = SRIP_Config()
        config.W = 10 / len(roots)
        config.epsilon = config.W
        config.dpi = dpi
        config.normalize_height = normalize_height
        images = [SRIP1(root, graph, config=config) for root in roots]
        image = images[0]
        if len(images) > 1:
            image = _concat_images(images)
        image.save(output_path)
    elif method == RenderMethod.LOGICAL:
        width = 10 / len(roots)
        images = [
            logical_render(
                graph,
                root,
                outer_width=width,
                dpi=dpi,
                normalize_height=normalize_height,
            )
            for root in roots
        ]
        image = images[0]
        if len(images) > 1:
            image = _concat_images(images)
        image.save(output_path)
    elif method == RenderMethod.TREEMAP:
        width = 10 / len(roots)
        images = [
            visualize_treemap_inmem(graph, root, width=width, dpi=dpi) for root in roots
        ]
        image = images[0]
        if len(images) > 1:
            image = _concat_images(images)
        image.save(output_path)
    else:
        raise ValueError("Invalid render method")


def _concat_images(images: list[Image.Image], margin: int = 10) -> Image.Image:
    margin = 10  # Adjust this value for wider or narrower margins

    # Calculate total width including margins
    total_width = sum(img.width for img in images) + margin * (len(images) - 1)
    outer_height = images[0].height  # Assuming all images have the same height

    # Create a new image with white background
    result_image = Image.new("RGB", (total_width, outer_height), color=(255, 255, 255))

    # Paste images horizontally with margins
    current_width = 0
    for img in images:
        result_image.paste(img, (current_width, 0))
        current_width += (
            img.width + margin
        )  # Add margin after each image except the last one

    # resize to square
    result_image = result_image.resize((outer_height, outer_height))

    # Save the result
    return result_image


if __name__ == "__main__":
    app()
