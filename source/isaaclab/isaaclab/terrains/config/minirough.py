
"""Configuration for custom terrains."""

import isaaclab.terrains as terrain_gen

from ..terrain_generator_cfg import TerrainGeneratorCfg

MINI_ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(4.0, 4.0),
    border_width=4.0,
    num_rows=5,
    num_cols=3,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.25, grid_width=0.45, grid_height_range=(0.0, 0.05), platform_width=1.0
        ),
        # "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
        #     proportion=0.25, noise_range=(0.0, 0.02), noise_step=0.01, border_width=0.25, horizontal_scale=1.0
        # ),
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.25, slope_range=(0.0, 0.4), platform_width=1.0, border_width=0.25
        ),
        "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.25, slope_range=(0.0, 0.4), platform_width=1.0, border_width=0.25
        ),
    },
)
"""Rough terrains configuration."""
