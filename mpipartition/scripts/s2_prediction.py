import sys
from typing import Optional

import click

from mpipartition.spherical_partition.s2_partition import (
    _s2_partition,
    visualize_s2_partition,
    _build_s2_segment_list,
    _print_area_imabalance,
    _print_edge_to_area_ratio,
    _print_segmentation_info,
)


@click.command()
@click.argument(
    "nranks",
    type=int,
)
@click.option(
    "--equal-area",
    is_flag=True,
    help=(
        "If set, partition S2 into equal area segments by varying delta_theta of "
        "rings."
    ),
)
@click.option(
    "--precision",
    type=int,
    default=6,
    help="Number of decimal places to use for printing.",
)
@click.option(
    "--figure",
    type=click.Path(dir_okay=False, writable=True),
    help="If set, save a visual representation of the S2 partitioning to this file.",
)
@click.option(
    "--use-mollweide",
    is_flag=True,
    help=(
        "If set, use the Mollweide projection. Otherwise, use a rectangular theta-phi "
        "plot for visualization."
    ),
)
@click.option(
    "--figure-pad",
    type=float,
    help=(
        "pad_inches argument used for saving the figure. Set to 0 to remove whitespace."
    ),
)
def cli(
    nranks: int,
    equal_area: bool,
    precision: int,
    figure: Optional[str],
    use_mollweide: bool,
    figure_pad: float,
):
    theta_cap, ring_thetas, ring_segments = _s2_partition(nranks, equal_area)
    all_s2_segments = _build_s2_segment_list(theta_cap, ring_thetas, ring_segments)
    _print_segmentation_info(
        nranks,
        theta_cap,
        ring_thetas,
        ring_segments,
        precision=precision,
    )
    _print_area_imabalance(all_s2_segments, precision=precision)
    _print_edge_to_area_ratio(all_s2_segments, precision=precision)

    if figure is not None:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print(
                "matplotlib is required for visualization, install mpipartition with "
                "`pip install mpipartition[viz]",
                file=sys.stderr,
            )
            return
        fig, ax = visualize_s2_partition(nranks, equal_area, use_mollweide)
        if use_mollweide:
            ax.set(xticklabels=[], yticklabels=[])
        fig.savefig(figure, bbox_inches="tight", pad_inches=figure_pad)
