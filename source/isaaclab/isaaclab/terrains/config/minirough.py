
"""Configuration for custom terrains."""

import isaaclab.terrains as terrain_gen

from ..terrain_generator_cfg import TerrainGeneratorCfg

MINI_ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(4.0, 4.0),
    border_width=4.0,
    num_rows=10,
    num_cols=10,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        # "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
        #     proportion=0.2,
        #     step_height_range=(0.01, 0.05),
        #     step_width=0.2,
        #     platform_width=1.0,
        #     border_width=1.0,
        #     holes=False,
        # ),
        # "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
        #     proportion=0.2,
        #     step_height_range=(0.01, 0.05),
        #     step_width=0.2,
        #     platform_width=1.0,
        #     border_width=1.0,
        #     holes=False,
        # ),
        "flat": terrain_gen.MeshPlaneTerrainCfg(),
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            grid_width=0.45, grid_height_range=(0.0, 0.025), platform_width=1.0
        ),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            noise_range=(0.0, 0.02), noise_step=0.01, border_width=0.25, horizontal_scale=1.0
        ),
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            slope_range=(0.0, 0.2), platform_width=1.0, border_width=0.25
        ),
        "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            slope_range=(0.0, 0.2), platform_width=1.0, border_width=0.25
        ),
    },
)
"""Rough terrains configuration."""
