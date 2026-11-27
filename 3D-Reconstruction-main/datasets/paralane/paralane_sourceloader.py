"""ParaLane scenes use the KITTI-compatible processed data format."""

from datasets.kitti.kitti_sourceloader import KITTILiDARSource, KITTIPixelSource


class ParaLanePixelSource(KITTIPixelSource):
    """Load ParaLane camera data using the KITTI-compatible file layout."""


class ParaLaneLiDARSource(KITTILiDARSource):
    """Load ParaLane LiDAR data using the KITTI-compatible file layout."""
