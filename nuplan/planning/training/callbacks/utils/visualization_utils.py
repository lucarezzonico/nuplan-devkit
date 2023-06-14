from enum import Enum
from typing import Optional, Tuple, Union, List, cast

import cv2
import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation

from nuplan.common.actor_state.vehicle_parameters import VehicleParameters, get_pacifica_parameters
from nuplan.planning.training.preprocessing.features.agents import Agents
from nuplan.planning.training.preprocessing.features.generic_agents import GenericAgents
from nuplan.planning.training.preprocessing.features.raster import Raster
from nuplan.planning.training.preprocessing.features.trajectory import Trajectory
from nuplan.planning.training.preprocessing.features.trajectories import Trajectories
from nuplan.planning.training.preprocessing.features.vector_map import VectorMap
from nuplan.planning.training.preprocessing.features.vector_set_map import VectorSetMap

import torch
import random

class Color(Enum):
    """
    Collection of colors for visualizing map elements, trajectories and agents.
    """

    BACKGROUND: Tuple[float, float, float] = (0, 0, 0)
    ROADMAP: Tuple[float, float, float] = (54, 67, 94)
    AGENTS: Tuple[float, float, float] = (113, 100, 222)
    EGO: Tuple[float, float, float] = (82, 86, 92)
    TARGET_TRAJECTORY: Tuple[float, float, float] = (61, 160, 179)
    PREDICTED_TRAJECTORY: Tuple[float, float, float] = (158, 63, 120)
    BASELINE_PATHS: Tuple[float, float, float] = (210, 220, 220)
    RED: Tuple[float, float, float] = (255, 0, 0)
    BLUE: Tuple[float, float, float] = (0, 0, 255)
    YELLOW: Tuple[float, float, float] = (255, 255, 0)
    GREEN: Tuple[float, float, float] = (0, 255, 0)
    ORANGE: Tuple[float, float, float] = (255, 127, 0)
    PURPLE: Tuple[float, float, float] = (255, 0, 255)
    RANDOM: Tuple[float, float, float] = (random.randint(0,255), random.randint(0,255), random.randint(0,255))

class MultiTrajColor(Enum):
    COLOR1: Tuple[float, float, float] = (255, 0, 0)
    COLOR2: Tuple[float, float, float] = (0, 255, 0)
    COLOR3: Tuple[float, float, float] = (0, 0, 255)
    COLOR4: Tuple[float, float, float] = (255, 255, 0)
    COLOR5: Tuple[float, float, float] = (255, 0, 255)
    COLOR6: Tuple[float, float, float] = (0, 255, 255)
    COLOR7: Tuple[float, float, float] = (255, 127, 0)
    COLOR8: Tuple[float, float, float] = (75, 0, 130)
    COLOR9: Tuple[float, float, float] = (148, 0, 211)
    COLOR10: Tuple[float, float, float] = (255, 255, 255)

        # , # Red
        # (255, 127, 0), # Orange
        # (255, 255, 0), # Yellow
        # (0, 255, 0), # Green
        # (0, 0, 255), # Blue
        # (75, 0, 130), # Indigo
        # (148, 0, 211) # Violet
        # )
    @classmethod
    def get_color_by_number(cls, number):
        colors = list(cls.__members__.values())
        return colors[number%len(colors)]
    
# color gradient that fades from yellow to red
class ColorFading(Enum):
    # for 6 modes
    COLOR1: Tuple[float, float, float] = (255, 255, 0)
    COLOR2: Tuple[float, float, float] = (255, 204, 0)
    COLOR3: Tuple[float, float, float] = (255, 153, 0)
    COLOR4: Tuple[float, float, float] = (255, 102, 0)
    COLOR5: Tuple[float, float, float] = (255, 51, 0)
    COLOR6: Tuple[float, float, float] = (255, 0, 0)
    # for 10 modes
    # COLOR1: Tuple[float, float, float] = (255, 255, 0)
    # COLOR2: Tuple[float, float, float] = (255, 230, 0)
    # COLOR3: Tuple[float, float, float] = (255, 204, 0)
    # COLOR4: Tuple[float, float, float] = (255, 179, 0)
    # COLOR5: Tuple[float, float, float] = (255, 153, 0)
    # COLOR6: Tuple[float, float, float] = (255, 128, 0)
    # COLOR7: Tuple[float, float, float] = (255, 102, 0)
    # COLOR8: Tuple[float, float, float] = (255, 77, 0)
    # COLOR9: Tuple[float, float, float] = (255, 51, 0)
    # COLOR10: Tuple[float, float, float] = (255, 0, 0)

    @classmethod
    def get_color_by_number(cls, number):
        colors = list(cls.__members__.values())
        return colors[number%len(colors)]

def _draw_trajectory(
    image: npt.NDArray[np.uint8],
    trajectory: Trajectory,
    color: Color,
    pixel_size: float,
    radius: int = 7,
    thickness: int = 3,
) -> None:
    """
    Draws a trajectory overlayed on an RGB image.

    :param image: image canvas
    :param trajectory: input trajectory
    :param color: desired trajectory color
    :param pixel_size: [m] size of pixel in meters
    :param radius: radius of each trajectory pose to be visualized
    :param thickness: thickness of lines connecting trajectory poses to be visualized
    """
    grid_shape = image.shape[:2]
    grid_height = grid_shape[0]
    grid_width = grid_shape[1]
    center_x = grid_width // 2
    center_y = grid_height // 2

    coords_x = (center_x - trajectory.numpy_position_x / pixel_size).astype(np.int32)
    coords_y = (center_y - trajectory.numpy_position_y / pixel_size).astype(np.int32)
    idxs = np.logical_and.reduce([0 <= coords_x, coords_x < grid_width, 0 <= coords_y, coords_y < grid_height])
    coords_x = coords_x[idxs]
    coords_y = coords_y[idxs]

    for point in zip(coords_y, coords_x):
        cv2.circle(image, point, radius=radius, color=color.value, thickness=-1)

    for point_1, point_2 in zip(zip(coords_y[:-1], coords_x[:-1]), zip(coords_y[1:], coords_x[1:])):
        cv2.line(image, point_1, point_2, color=color.value, thickness=thickness)


def _create_map_raster(
    vector_map: Union[VectorMap, VectorSetMap],
    radius: float,
    size: int,
    bit_shift: int,
    pixel_size: float,
    color: int = 1,
    thickness: int = 2,
) -> npt.NDArray[np.uint8]:
    """
    Create vector map raster layer to be visualized.

    :param vector_map: Vector map feature object.
    :param radius: [m] Radius of grid.
    :param bit_shift: Bit shift when drawing or filling precise polylines/rectangles.
    :param pixel_size: [m] Size of each pixel.
    :param size: [pixels] Size of grid.
    :param color: Grid color.
    :param thickness: Map lane/baseline thickness.
    :return: Instantiated grid.
    """
    # Extract coordinates from vector map feature
    vector_coords = vector_map.get_lane_coords(0)  # Get first sample in batch

    # Align coordinates to map and clip them based on radius
    num_elements, num_points, _ = vector_coords.shape
    map_ortho_align = Rotation.from_euler('z', 90, degrees=True).as_matrix().astype(np.float32)
    coords = vector_coords.reshape(num_elements * num_points, 2)
    coords = np.concatenate((coords, np.zeros_like(coords[:, 0:1])), axis=-1)
    coords = (map_ortho_align @ coords.T).T
    coords = coords[:, :2].reshape(num_elements, num_points, 2)
    coords[..., 0] = np.clip(coords[..., 0], -radius, radius)
    coords[..., 1] = np.clip(coords[..., 1], -radius, radius)

    # Instantiate grid
    map_raster: npt.NDArray[np.uint8] = np.zeros((size, size), dtype=np.uint8)

    # Convert coordinates to grid indices
    index_coords = (radius + coords) / pixel_size
    shifted_index_coords = (index_coords * 2**bit_shift).astype(np.int64)

    # Paint the grid
    cv2.polylines(
        map_raster,
        shifted_index_coords,
        isClosed=False,
        color=color,
        thickness=thickness,
        shift=bit_shift,
        lineType=cv2.LINE_AA,
    )
    
    # shifted_index_coords_tensor = torch.Tensor(shifted_index_coords).view(-1, shifted_index_coords.shape[-1])
    # for i in range(shifted_index_coords_tensor.size(0)):
    #     cv2.drawMarker(
    #         map_raster,
    #         tuple(shifted_index_coords_tensor[i].numpy()),
    #         color=color,
    #         markerType=cv2.MARKER_CROSS,
    #         markerSize=10,
    #         thickness=5,
    #         line_type=cv2.LINE_AA
    #     )
    #     cv2.circle(map_raster, tuple(shifted_index_coords_tensor[i].numpy()), radius=10, color=color, thickness=5)
    
    # Flip grid upside down
    map_raster = np.flipud(map_raster)

    return map_raster

def _create_green_traffic_light_colored_lane_raster(
    vector_map: Union[VectorMap, VectorSetMap],
    radius: float,
    size: int,
    bit_shift: int,
    pixel_size: float,
    color: int = 1,
    thickness: int = 2,
) -> npt.NDArray[np.uint8]:
    """
    Create vector map raster layer to be visualized.

    :param vector_map: Vector map feature object.
    :param radius: [m] Radius of grid.
    :param bit_shift: Bit shift when drawing or filling precise polylines/rectangles.
    :param pixel_size: [m] Size of each pixel.
    :param size: [pixels] Size of grid.
    :param color: Grid color.
    :param thickness: Map lane/baseline thickness.
    :return: Instantiated grid.
    """
    # Extract coordinates from vector map feature
    vector_coords = vector_map.get_lane_coords(0)  # Get first sample in batch
    traffic_lights = vector_map.traffic_light_data["LANE"][0][:,:,0:1] # green traffic lights
    vector_coords = torch.mul(vector_coords, traffic_lights)

    # Align coordinates to map and clip them based on radius
    num_elements, num_points, _ = vector_coords.shape
    map_ortho_align = Rotation.from_euler('z', 90, degrees=True).as_matrix().astype(np.float32)
    coords = vector_coords.reshape(num_elements * num_points, 2)
    coords = np.concatenate((coords, np.zeros_like(coords[:, 0:1])), axis=-1)
    coords = (map_ortho_align @ coords.T).T
    coords = coords[:, :2].reshape(num_elements, num_points, 2)
    coords[..., 0] = np.clip(coords[..., 0], -radius, radius)
    coords[..., 1] = np.clip(coords[..., 1], -radius, radius)

    # Instantiate grid
    map_raster: npt.NDArray[np.uint8] = np.zeros((size, size), dtype=np.uint8)

    # Convert coordinates to grid indices
    index_coords = (radius + coords) / pixel_size
    shifted_index_coords = (index_coords * 2**bit_shift).astype(np.int64)

    # Paint the grid
    cv2.polylines(
        map_raster,
        shifted_index_coords,
        isClosed=False,
        color=color,
        thickness=thickness,
        shift=bit_shift,
        lineType=cv2.LINE_AA,
    )
    
    # Flip grid upside down
    map_raster = np.flipud(map_raster)

    return map_raster

def _create_yellow_traffic_light_colored_lane_raster(
    vector_map: Union[VectorMap, VectorSetMap],
    radius: float,
    size: int,
    bit_shift: int,
    pixel_size: float,
    color: int = 1,
    thickness: int = 2,
) -> npt.NDArray[np.uint8]:
    """
    Create vector map raster layer to be visualized.

    :param vector_map: Vector map feature object.
    :param radius: [m] Radius of grid.
    :param bit_shift: Bit shift when drawing or filling precise polylines/rectangles.
    :param pixel_size: [m] Size of each pixel.
    :param size: [pixels] Size of grid.
    :param color: Grid color.
    :param thickness: Map lane/baseline thickness.
    :return: Instantiated grid.
    """
    # Extract coordinates from vector map feature
    vector_coords = vector_map.get_lane_coords(0)  # Get first sample in batch
    traffic_lights = vector_map.traffic_light_data["LANE"][0][:,:,1:2] # green traffic lights
    vector_coords = torch.mul(vector_coords, traffic_lights)
    
    # Align coordinates to map and clip them based on radius
    num_elements, num_points, _ = vector_coords.shape
    map_ortho_align = Rotation.from_euler('z', 90, degrees=True).as_matrix().astype(np.float32)
    coords = vector_coords.reshape(num_elements * num_points, 2)
    coords = np.concatenate((coords, np.zeros_like(coords[:, 0:1])), axis=-1)
    coords = (map_ortho_align @ coords.T).T
    coords = coords[:, :2].reshape(num_elements, num_points, 2)
    coords[..., 0] = np.clip(coords[..., 0], -radius, radius)
    coords[..., 1] = np.clip(coords[..., 1], -radius, radius)

    # Instantiate grid
    map_raster: npt.NDArray[np.uint8] = np.zeros((size, size), dtype=np.uint8)

    # Convert coordinates to grid indices
    index_coords = (radius + coords) / pixel_size
    shifted_index_coords = (index_coords * 2**bit_shift).astype(np.int64)

    # Paint the grid
    cv2.polylines(
        map_raster,
        shifted_index_coords,
        isClosed=False,
        color=color,
        thickness=thickness,
        shift=bit_shift,
        lineType=cv2.LINE_AA,
    )
    
    # Flip grid upside down
    map_raster = np.flipud(map_raster)

    return map_raster

def _create_red_traffic_light_colored_lane_raster(
    vector_map: Union[VectorMap, VectorSetMap],
    radius: float,
    size: int,
    bit_shift: int,
    pixel_size: float,
    color: int = 1,
    thickness: int = 2,
) -> npt.NDArray[np.uint8]:
    """
    Create vector map raster layer to be visualized.

    :param vector_map: Vector map feature object.
    :param radius: [m] Radius of grid.
    :param bit_shift: Bit shift when drawing or filling precise polylines/rectangles.
    :param pixel_size: [m] Size of each pixel.
    :param size: [pixels] Size of grid.
    :param color: Grid color.
    :param thickness: Map lane/baseline thickness.
    :return: Instantiated grid.
    """
    # Extract coordinates from vector map feature
    vector_coords = vector_map.get_lane_coords(0)  # Get first sample in batch
    traffic_lights = vector_map.traffic_light_data["LANE"][0][:,:,2:3] # green traffic lights
    vector_coords = torch.mul(vector_coords, traffic_lights)
    
    # Align coordinates to map and clip them based on radius
    num_elements, num_points, _ = vector_coords.shape
    map_ortho_align = Rotation.from_euler('z', 90, degrees=True).as_matrix().astype(np.float32)
    coords = vector_coords.reshape(num_elements * num_points, 2)
    coords = np.concatenate((coords, np.zeros_like(coords[:, 0:1])), axis=-1)
    coords = (map_ortho_align @ coords.T).T
    coords = coords[:, :2].reshape(num_elements, num_points, 2)
    coords[..., 0] = np.clip(coords[..., 0], -radius, radius)
    coords[..., 1] = np.clip(coords[..., 1], -radius, radius)

    # Instantiate grid
    map_raster: npt.NDArray[np.uint8] = np.zeros((size, size), dtype=np.uint8)

    # Convert coordinates to grid indices
    index_coords = (radius + coords) / pixel_size
    shifted_index_coords = (index_coords * 2**bit_shift).astype(np.int64)

    # Paint the grid
    cv2.polylines(
        map_raster,
        shifted_index_coords,
        isClosed=False,
        color=color,
        thickness=thickness,
        shift=bit_shift,
        lineType=cv2.LINE_AA,
    )
    
    # Flip grid upside down
    map_raster = np.flipud(map_raster)

    return map_raster

def _create_left_boundary_raster(
    vector_map: Union[VectorMap, VectorSetMap],
    radius: float,
    size: int,
    bit_shift: int,
    pixel_size: float,
    color: int = 1,
    thickness: int = 2,
) -> npt.NDArray[np.uint8]:
    """
    Create vector map raster layer to be visualized.

    :param vector_map: Vector map feature object.
    :param radius: [m] Radius of grid.
    :param bit_shift: Bit shift when drawing or filling precise polylines/rectangles.
    :param pixel_size: [m] Size of each pixel.
    :param size: [pixels] Size of grid.
    :param color: Grid color.
    :param thickness: Map lane/baseline thickness.
    :return: Instantiated grid.
    """
    # # Extract coordinates from vector map feature
    vector_coords = vector_map.get_left_boundary_coords(0)
    # Align coordinates to map and clip them based on radius
    num_elements, num_points, _ = vector_coords.shape
    map_ortho_align = Rotation.from_euler('z', 90, degrees=True).as_matrix().astype(np.float32)
    coords = vector_coords.reshape(num_elements * num_points, 2)
    coords = np.concatenate((coords, np.zeros_like(coords[:, 0:1])), axis=-1)
    coords = (map_ortho_align @ coords.T).T
    coords = coords[:, :2].reshape(num_elements, num_points, 2)
    coords[..., 0] = np.clip(coords[..., 0], -radius, radius)
    coords[..., 1] = np.clip(coords[..., 1], -radius, radius)

    # Instantiate grid
    map_raster: npt.NDArray[np.uint8] = np.zeros((size, size), dtype=np.uint8)

    # Convert coordinates to grid indices
    index_coords = (radius + coords) / pixel_size
    shifted_index_coords = (index_coords * 2**bit_shift).astype(np.int64)
    
    # Paint the grid
    cv2.polylines(
        map_raster,
        shifted_index_coords,
        isClosed=False,
        color=color,
        thickness=thickness,
        shift=bit_shift,
        lineType=cv2.LINE_AA,
    )
    
    # Flip grid upside down
    map_raster = np.flipud(map_raster)

    return map_raster

def _create_right_boundary_raster(
    vector_map: Union[VectorMap, VectorSetMap],
    radius: float,
    size: int,
    bit_shift: int,
    pixel_size: float,
    color: int = 1,
    thickness: int = 2,
) -> npt.NDArray[np.uint8]:
    """
    Create vector map raster layer to be visualized.

    :param vector_map: Vector map feature object.
    :param radius: [m] Radius of grid.
    :param bit_shift: Bit shift when drawing or filling precise polylines/rectangles.
    :param pixel_size: [m] Size of each pixel.
    :param size: [pixels] Size of grid.
    :param color: Grid color.
    :param thickness: Map lane/baseline thickness.
    :return: Instantiated grid.
    """
    # # Extract coordinates from vector map feature
    vector_coords = vector_map.get_right_boundary_coords(0)
    # Align coordinates to map and clip them based on radius
    num_elements, num_points, _ = vector_coords.shape
    map_ortho_align = Rotation.from_euler('z', 90, degrees=True).as_matrix().astype(np.float32)
    coords = vector_coords.reshape(num_elements * num_points, 2)
    coords = np.concatenate((coords, np.zeros_like(coords[:, 0:1])), axis=-1)
    coords = (map_ortho_align @ coords.T).T
    coords = coords[:, :2].reshape(num_elements, num_points, 2)
    coords[..., 0] = np.clip(coords[..., 0], -radius, radius)
    coords[..., 1] = np.clip(coords[..., 1], -radius, radius)

    # Instantiate grid
    map_raster: npt.NDArray[np.uint8] = np.zeros((size, size), dtype=np.uint8)

    # Convert coordinates to grid indices
    index_coords = (radius + coords) / pixel_size
    shifted_index_coords = (index_coords * 2**bit_shift).astype(np.int64)
    
    # Paint the grid
    cv2.polylines(
        map_raster,
        shifted_index_coords,
        isClosed=False,
        color=color,
        thickness=thickness,
        shift=bit_shift,
        lineType=cv2.LINE_AA,
    )
    
    # Flip grid upside down
    map_raster = np.flipud(map_raster)

    return map_raster

def _create_stopline_raster(
    vector_map: Union[VectorMap, VectorSetMap],
    radius: float,
    size: int,
    bit_shift: int,
    pixel_size: float,
    color: int = 1,
    thickness: int = 2,
) -> npt.NDArray[np.uint8]:
    """
    Create vector map raster layer to be visualized.

    :param vector_map: Vector map feature object.
    :param radius: [m] Radius of grid.
    :param bit_shift: Bit shift when drawing or filling precise polylines/rectangles.
    :param pixel_size: [m] Size of each pixel.
    :param size: [pixels] Size of grid.
    :param color: Grid color.
    :param thickness: Map lane/baseline thickness.
    :return: Instantiated grid.
    """
    # # Extract coordinates from vector map feature
    vector_coords = vector_map.get_stopline_coords(0)
    # mask = vector_coords.any(axis=-1).unsqueeze(dim=-1).repeat(1,1,2)
    
    # Instantiate grid
    map_raster: npt.NDArray[np.uint8] = np.zeros((size, size), dtype=np.uint8)
    
    # if mask.any():
    #     vector_coords = vector_coords[mask].view(-1, int(mask.sum(dim=1).max()), vector_coords.shape[-1])
            
    # Align coordinates to map and clip them based on radius
    num_elements, num_points, _ = vector_coords.shape
    map_ortho_align = Rotation.from_euler('z', 90, degrees=True).as_matrix().astype(np.float32)
    coords = vector_coords.reshape(num_elements * num_points, 2)
    coords = np.concatenate((coords, np.zeros_like(coords[:, 0:1])), axis=-1)
    coords = (map_ortho_align @ coords.T).T
    coords = coords[:, :2].reshape(num_elements, num_points, 2)
    coords[..., 0] = np.clip(coords[..., 0], -radius, radius)
    coords[..., 1] = np.clip(coords[..., 1], -radius, radius)

    # Convert coordinates to grid indices
    index_coords = (radius + coords) / pixel_size
    shifted_index_coords = (index_coords * 2**bit_shift).astype(np.int64)
    
    for i in range(num_elements):
        pts = np.expand_dims(shifted_index_coords[i][coords[i].nonzero()].reshape(-1, coords.shape[-1]), axis=0)
        # Paint the grid
        cv2.polylines(
            map_raster,
            pts, # [N,1,2]
            isClosed=False,
            color=color,
            thickness=thickness,
            shift=bit_shift,
            lineType=cv2.LINE_AA,
        )
        
    # Flip grid upside down
    map_raster = np.flipud(map_raster)

    return map_raster

def _create_crosswalk_raster(
    vector_map: Union[VectorMap, VectorSetMap],
    radius: float,
    size: int,
    bit_shift: int,
    pixel_size: float,
    color: int = 1,
    thickness: int = 2,
) -> npt.NDArray[np.uint8]:
    """
    Create vector map raster layer to be visualized.

    :param vector_map: Vector map feature object.
    :param radius: [m] Radius of grid.
    :param bit_shift: Bit shift when drawing or filling precise polylines/rectangles.
    :param pixel_size: [m] Size of each pixel.
    :param size: [pixels] Size of grid.
    :param color: Grid color.
    :param thickness: Map lane/baseline thickness.
    :return: Instantiated grid.
    """
    # # Extract coordinates from vector map feature
    vector_coords = vector_map.get_crosswalk_coords(0)
    # mask = vector_coords.any(axis=-1).unsqueeze(dim=-1).repeat(1,1,2)
    
    # Instantiate grid
    map_raster: npt.NDArray[np.uint8] = np.zeros((size, size), dtype=np.uint8)
        
    # if mask.any():
        # vector_coords = vector_coords[mask].view(-1, int(mask.sum(dim=1).max()), vector_coords.shape[-1])
    
    # Align coordinates to map and clip them based on radius
    num_elements, num_points, _ = vector_coords.shape
    map_ortho_align = Rotation.from_euler('z', 90, degrees=True).as_matrix().astype(np.float32)
    coords = vector_coords.reshape(num_elements * num_points, 2)
    coords = np.concatenate((coords, np.zeros_like(coords[:, 0:1])), axis=-1)
    coords = (map_ortho_align @ coords.T).T
    coords = coords[:, :2].reshape(num_elements, num_points, 2)
    coords[..., 0] = np.clip(coords[..., 0], -radius, radius)
    coords[..., 1] = np.clip(coords[..., 1], -radius, radius)

    # Convert coordinates to grid indices
    index_coords = (radius + coords) / pixel_size
    shifted_index_coords = (index_coords * 2**bit_shift).astype(np.int64)
    
    for i in range(num_elements):
        pts = np.expand_dims(shifted_index_coords[i][coords[i].nonzero()].reshape(-1, coords.shape[-1]), axis=0)
        # Paint the grid
        cv2.polylines(
            map_raster,
            pts, # [N,1,2]
            isClosed=False,
            color=color,
            thickness=thickness,
            shift=bit_shift,
            lineType=cv2.LINE_AA,
        )
    
    # Flip grid upside down
    map_raster = np.flipud(map_raster)

    return map_raster

def _create_route_lanes_raster(
    vector_map: Union[VectorMap, VectorSetMap],
    radius: float,
    size: int,
    bit_shift: int,
    pixel_size: float,
    color: int = 1,
    thickness: int = 2,
) -> npt.NDArray[np.uint8]:
    """
    Create vector map raster layer to be visualized.

    :param vector_map: Vector map feature object.
    :param radius: [m] Radius of grid.
    :param bit_shift: Bit shift when drawing or filling precise polylines/rectangles.
    :param pixel_size: [m] Size of each pixel.
    :param size: [pixels] Size of grid.
    :param color: Grid color.
    :param thickness: Map lane/baseline thickness.
    :return: Instantiated grid.
    """
    # # Extract coordinates from vector map feature
    vector_coords = vector_map.get_route_lanes_coords(0)
    # vector_coords = vector_coords[:1]
    # Align coordinates to map and clip them based on radius
    num_elements, num_points, _ = vector_coords.shape
    map_ortho_align = Rotation.from_euler('z', 90, degrees=True).as_matrix().astype(np.float32)
    coords = vector_coords.reshape(num_elements * num_points, 2)
    coords = np.concatenate((coords, np.zeros_like(coords[:, 0:1])), axis=-1)
    coords = (map_ortho_align @ coords.T).T
    coords = coords[:, :2].reshape(num_elements, num_points, 2)
    coords[..., 0] = np.clip(coords[..., 0], -radius, radius)
    coords[..., 1] = np.clip(coords[..., 1], -radius, radius)

    # Instantiate grid
    map_raster: npt.NDArray[np.uint8] = np.zeros((size, size), dtype=np.uint8)

    # Convert coordinates to grid indices
    index_coords = (radius + coords) / pixel_size
    shifted_index_coords = (index_coords * 2**bit_shift).astype(np.int64)
    
    # Paint the grid
    cv2.polylines(
        map_raster,
        shifted_index_coords,
        isClosed=False,
        color=color,
        thickness=thickness,
        shift=bit_shift,
        lineType=cv2.LINE_AA,
    )
    
    # _create_point_markers(map_raster, index_coords, color=color)
    
    # Flip grid upside down
    map_raster = np.flipud(map_raster)

    return map_raster

def _create_point_markers(
    map_raster: npt.NDArray[np.uint8],
    polylines,
    color: int = 1,
) -> None:
    
    polylines = torch.Tensor(polylines).view(-1, polylines.shape[-1])
    for i in range(0, polylines.size(0)):
        # cv2.drawMarker(map_raster, tuple(polylines[i].numpy()), color=color, markerType=cv2.MARKER_CROSS, markerSize=10, thickness=3, line_type=cv2.LINE_AA)
        cv2.circle(map_raster, tuple(polylines[i].numpy()), radius=6, color=color, thickness=2)


def _create_agents_raster(
    agents: Union[Agents, GenericAgents], radius: float, size: int, bit_shift: int, pixel_size: float, color: int = 1
) -> npt.NDArray[np.uint8]:
    """
    Create agents raster layer to be visualized.

    :param agents: agents feature object (either Agents or GenericAgents).
    :param radius: [m] Radius of grid.
    :param bit_shift: Bit shift when drawing or filling precise polylines/rectangles.
    :param pixel_size: [m] Size of each pixel.
    :param size: [pixels] Size of grid.
    :param color: Grid color.
    :return: Instantiated grid.
    """
    # Instantiate grid
    agents_raster: npt.NDArray[np.uint8] = np.zeros((size, size), dtype=np.uint8)

    # Extract array data from features
    agents_array: npt.NDArray[np.float32] = np.asarray(
        agents.get_present_agents_in_sample(0)
    )  # Get first sample in batch
    agents_corners: npt.NDArray[np.float32] = np.asarray(
        agents.get_agent_corners_in_sample(0)
    )  # Get first sample in batch

    if len(agents_array) == 0:
        return agents_raster

    # Align coordinates to map, transform them to ego's reference and clip them based on radius
    map_ortho_align = Rotation.from_euler('z', 90, degrees=True).as_matrix().astype(np.float32)
    transform = Rotation.from_euler('z', agents_array[:, 2], degrees=False).as_matrix().astype(np.float32)
    transform[:, :2, 2] = agents_array[:, :2]
    points = (map_ortho_align @ transform @ agents_corners.transpose([0, 2, 1])).transpose([0, 2, 1])[..., :2]
    points[..., 0] = np.clip(points[..., 0], -radius, radius)
    points[..., 1] = np.clip(points[..., 1], -radius, radius)

    # Convert coordinates to grid indices
    index_points = (radius + points) / pixel_size
    shifted_index_points = (index_points * 2**bit_shift).astype(np.int64)

    # Paint the grid
    for box in shifted_index_points:
        cv2.fillPoly(agents_raster, box[None], color=color, shift=bit_shift, lineType=cv2.LINE_AA)

    # Flip grid upside down
    agents_raster = np.flipud(agents_raster)

    return agents_raster


def _create_ego_raster(
    vehicle_parameters: VehicleParameters, pixel_size: float, size: int, color: int = 1, thickness: int = -1
) -> npt.NDArray[np.uint8]:
    """
    Create ego raster layer to be visualized.

    :param vehicle_parameters: Ego vehicle parameters dataclass object.
    :param pixel_size: [m] Size of each pixel.
    :param size: [pixels] Size of grid.
    :param color: Grid color.
    :param thickness: Box line thickness (-1 means fill).
    :return: Instantiated grid.
    """
    # Instantiate grid
    ego_raster: npt.NDArray[np.uint8] = np.zeros((size, size), dtype=np.uint8)

    # Extract ego vehicle dimensions
    ego_width = vehicle_parameters.width
    ego_front_length = vehicle_parameters.front_length
    ego_rear_length = vehicle_parameters.rear_length

    # Convert coordinates to grid indices
    ego_width_pixels = int(ego_width / pixel_size)
    ego_front_length_pixels = int(ego_front_length / pixel_size)
    ego_rear_length_pixels = int(ego_rear_length / pixel_size)
    map_x_center = int(ego_raster.shape[1] * 0.5)
    map_y_center = int(ego_raster.shape[0] * 0.5)
    ego_top_left = (map_x_center - ego_width_pixels // 2, map_y_center - ego_front_length_pixels)
    ego_bottom_right = (map_x_center + ego_width_pixels // 2, map_y_center + ego_rear_length_pixels)

    # Paint the grid
    cv2.rectangle(ego_raster, ego_top_left, ego_bottom_right, color=color, thickness=thickness, lineType=cv2.LINE_AA)

    return ego_raster


def get_raster_from_vector_map_with_agents(
    vector_map: Union[VectorMap, VectorSetMap],
    agents: Union[Agents, GenericAgents],
    target_trajectory: Optional[Trajectory] = None,
    predicted_trajectory: Optional[Trajectory] = None,
    pixel_size: float = 0.5,
    bit_shift: int = 12,
    radius: float = 50.0,
    vehicle_parameters: VehicleParameters = get_pacifica_parameters(),
    vector_map_feature: str = None,
) -> npt.NDArray[np.uint8]:
    """
    Create rasterized image from vector map and list of agents.

    :param vector_map: Vector map/vector set map feature to visualize.
    :param agents: Agents/GenericAgents feature to visualize.
    :param target_trajectory: Target trajectory to visualize.
    :param predicted_trajectory: Predicted trajectory to visualize.
    :param pixel_size: [m] Size of a pixel.
    :param bit_shift: Bit shift when drawing or filling precise polylines/rectangles.
    :param radius: [m] Radius of raster.
    :param vehicle_parameters: Parameters of the ego vehicle.
    :return: Composed rasterized image.
    """
    # Raster size
    size = int(2 * radius / pixel_size)

    # Create map layers
    map_raster = _create_map_raster(vector_map, radius, size, bit_shift, pixel_size, thickness=4)
    # if vector_map_feature == 'vector_set_map':
    #     left_boundary_raster = _create_left_boundary_raster(vector_map, radius, size, bit_shift, pixel_size, thickness=2)
    #     right_boundary_raster = _create_right_boundary_raster(vector_map, radius, size, bit_shift, pixel_size, thickness=2)
    #     stopline_raster = _create_stopline_raster(vector_map, radius, size, bit_shift, pixel_size, thickness=2)
    #     crosswalk_raster = _create_crosswalk_raster(vector_map, radius, size, bit_shift, pixel_size, thickness=2)
    #     route_lanes_raster = _create_route_lanes_raster(vector_map, radius, size, bit_shift, pixel_size, thickness=4)
    # green_traffic_light_raster = _create_green_traffic_light_colored_lane_raster(vector_map, radius, size, bit_shift, pixel_size, thickness=4)
    # yellow_traffic_light_raster = _create_yellow_traffic_light_colored_lane_raster(vector_map, radius, size, bit_shift, pixel_size, thickness=4)
    # red_traffic_light_raster = _create_red_traffic_light_colored_lane_raster(vector_map, radius, size, bit_shift, pixel_size, thickness=4)
    agents_raster = _create_agents_raster(agents, radius, size, bit_shift, pixel_size)
    ego_raster = _create_ego_raster(vehicle_parameters, pixel_size, size)

    # Compose and paint image
    image: npt.NDArray[np.uint8] = np.full((size, size, 3), Color.BACKGROUND.value, dtype=np.uint8)
    image[map_raster.nonzero()] = Color.BASELINE_PATHS.value
    # if vector_map_feature == 'vector_set_map':
    #     image[left_boundary_raster.nonzero()] = Color.BLUE.value
    #     image[right_boundary_raster.nonzero()] = Color.BLUE.value
    #     image[stopline_raster.nonzero()] = Color.RED.value
    #     image[crosswalk_raster.nonzero()] = Color.YELLOW.value
    #     image[route_lanes_raster.nonzero()] = Color.EGO.value
    # image[green_traffic_light_raster.nonzero()] = Color.GREEN.value
    # image[yellow_traffic_light_raster.nonzero()] = Color.ORANGE.value
    # image[red_traffic_light_raster.nonzero()] = Color.RED.value
    image[agents_raster.nonzero()] = Color.AGENTS.value
    image[ego_raster.nonzero()] = Color.EGO.value

    # Draw predicted and target trajectories
    if target_trajectory is not None:
        _draw_trajectory(image, target_trajectory, Color.TARGET_TRAJECTORY, pixel_size)
    if predicted_trajectory is not None:
        _draw_trajectory(image, predicted_trajectory, Color.PREDICTED_TRAJECTORY, pixel_size)

    return image


def get_raster_from_vector_map_with_agents_multimodal(
    vector_map: Union[VectorMap, VectorSetMap],
    agents: Union[Agents, GenericAgents],
    target_trajectory: Optional[Trajectory] = None,
    predicted_trajectories: Optional[Trajectories] = None,
    pixel_size: float = 0.5,
    bit_shift: int = 12,
    radius: float = 50.0,
    vehicle_parameters: VehicleParameters = get_pacifica_parameters(),
    vector_map_feature: str = None,
) -> npt.NDArray[np.uint8]:
    """
    Create rasterized image from vector map and list of agents.

    :param vector_map: Vector map/vector set map feature to visualize.
    :param agents: Agents/GenericAgents feature to visualize.
    :param target_trajectory: Target trajectory to visualize.
    :param predicted_trajectories: List of predicted trajectories to visualize.
    :param pixel_size: [m] Size of a pixel.
    :param bit_shift: Bit shift when drawing or filling precise polylines/rectangles.
    :param radius: [m] Radius of raster.
    :param vehicle_parameters: Parameters of the ego vehicle.
    :return: Composed rasterized image.
    """
    # Raster size
    size = int(2 * radius / pixel_size)

    # Create map layers
    map_raster = _create_map_raster(vector_map, radius, size, bit_shift, pixel_size, thickness=4)
    # if vector_map_feature == 'vector_set_map':
    #     left_boundary_raster = _create_left_boundary_raster(vector_map, radius, size, bit_shift, pixel_size, thickness=2)
    #     right_boundary_raster = _create_right_boundary_raster(vector_map, radius, size, bit_shift, pixel_size, thickness=2)
    #     stopline_raster = _create_stopline_raster(vector_map, radius, size, bit_shift, pixel_size, thickness=2)
    #     crosswalk_raster = _create_crosswalk_raster(vector_map, radius, size, bit_shift, pixel_size, thickness=2)
    #     route_lanes_raster = _create_route_lanes_raster(vector_map, radius, size, bit_shift, pixel_size, thickness=4)
    # green_traffic_light_raster = _create_green_traffic_light_colored_lane_raster(vector_map, radius, size, bit_shift, pixel_size, thickness=4)
    # yellow_traffic_light_raster = _create_yellow_traffic_light_colored_lane_raster(vector_map, radius, size, bit_shift, pixel_size, thickness=4)
    # red_traffic_light_raster = _create_red_traffic_light_colored_lane_raster(vector_map, radius, size, bit_shift, pixel_size, thickness=4)
    agents_raster = _create_agents_raster(agents, radius, size, bit_shift, pixel_size)
    ego_raster = _create_ego_raster(vehicle_parameters, pixel_size, size)

    # Compose and paint image
    image: npt.NDArray[np.uint8] = np.full((size, size, 3), Color.BACKGROUND.value, dtype=np.uint8)
    image[map_raster.nonzero()] = Color.BASELINE_PATHS.value
    # if vector_map_feature == 'vector_set_map':
    #     image[left_boundary_raster.nonzero()] = Color.BLUE.value
    #     image[right_boundary_raster.nonzero()] = Color.BLUE.value
    #     image[stopline_raster.nonzero()] = Color.RED.value
    #     image[crosswalk_raster.nonzero()] = Color.YELLOW.value
    #     image[route_lanes_raster.nonzero()] = Color.EGO.value
    # image[green_traffic_light_raster.nonzero()] = Color.GREEN.value
    # image[yellow_traffic_light_raster.nonzero()] = Color.ORANGE.value
    # image[red_traffic_light_raster.nonzero()] = Color.RED.value
    image[agents_raster.nonzero()] = Color.AGENTS.value
    image[ego_raster.nonzero()] = Color.EGO.value

    # Draw target trajectory
    if target_trajectory is not None:
        _draw_trajectory(image, target_trajectory, Color.TARGET_TRAJECTORY, pixel_size)

    # Draw predicted trajectories
    if predicted_trajectories is not None:
        for pred_traj in predicted_trajectories.trajectories: # [50]Trajectory(6,16,3) or [1]Trajectory(6,16,3) 
            for i, predicted_trajectory in enumerate(pred_traj.unpack()): # unpack all Trajectory in the first and only Trajectories
                # Color.RANDOM.value = (random.randint(0,255), random.randint(0,255), random.randint(0,255))
                if predicted_trajectory.data.any():
                    _draw_trajectory(image, predicted_trajectory, ColorFading.get_color_by_number(i), pixel_size)

    return image

def get_raster_from_vector_map_with_agents_multiple_trajectories(
    vector_map: Union[VectorMap, VectorSetMap],
    agents: Union[Agents, GenericAgents],
    target_trajectory: Optional[Trajectory] = None,
    predicted_trajectory: Optional[Trajectory] = None,
    pixel_size: float = 0.5,
    bit_shift: int = 12,
    radius: float = 50.0,
    vehicle_parameters: VehicleParameters = get_pacifica_parameters(),
) -> npt.NDArray[np.uint8]:
    """
    Create rasterized image from vector map and list of agents.
    :param vector_map: Vector map/vector set map feature to visualize.
    :param agents: Agents/GenericAgents feature to visualize.
    :param target_trajectory: Target trajectory to visualize.
    :param predicted_trajectory: Predicted trajectory to visualize.
    :param pixel_size: [m] Size of a pixel.
    :param bit_shift: Bit shift when drawing or filling precise polylines/rectangles.
    :param radius: [m] Radius of raster.
    :param vehicle_parameters: Parameters of the ego vehicle.
    :return: Composed rasterized image.
    """
    # Raster size
    size = int(2 * radius / pixel_size)

    # Create map layers
    map_raster = _create_map_raster(vector_map, radius, size, bit_shift, pixel_size)
    agents_raster = _create_agents_raster(agents, radius, size, bit_shift, pixel_size)
    ego_raster = _create_ego_raster(vehicle_parameters, pixel_size, size)

    # Compose and paint image
    image: npt.NDArray[np.uint8] = np.full((size, size, 3), Color.BACKGROUND.value, dtype=np.uint8)
    image[map_raster.nonzero()] = Color.BASELINE_PATHS.value
    image[agents_raster.nonzero()] = Color.AGENTS.value
    image[ego_raster.nonzero()] = Color.EGO.value

    # Draw predicted and target trajectories
    if target_trajectory is not None:
        _draw_trajectory(image, target_trajectory, Color.TARGET_TRAJECTORY, pixel_size)
    if predicted_trajectory is not None:
        trajectory_list = predicted_trajectory.unpack()
        for i, trajectory in enumerate(trajectory_list):
            _draw_trajectory(image, trajectory, MultiTrajColor.get_color_by_number(i), pixel_size)

    return image

def get_raster_with_trajectories_as_rgb(
    raster: Raster,
    target_trajectory: Optional[Trajectory] = None,
    predicted_trajectory: Optional[Trajectory] = None,
    pixel_size: float = 0.5,
) -> npt.NDArray[np.uint8]:
    """
    Create an RGB images of the raster layers overlayed with predicted / ground truth trajectories

    :param raster: input raster to visualize
    :param target_trajectory: target (ground truth) trajectory to visualize
    :param predicted_trajectory: predicted trajectory to visualize
    :param background_color: desired color of the image's background
    :param roadmap_color: desired color of the map raster layer
    :param agents_color: desired color of the agents raster layer
    :param ego_color: desired color of the ego raster layer
    :param target_trajectory_color: desired color of the target trajectory
    :param predicted_trajectory_color: desired color of the predicted trajectory
    :param pixel_size: [m] size of pixel in meters
    :return: constructed RGB image
    """
    grid_shape = (raster.height, raster.width)

    # Compose and paint image
    image: npt.NDArray[np.uint8] = np.full((*grid_shape, 3), Color.BACKGROUND.value, dtype=np.uint8)
    image[raster.roadmap_layer[0] > 0] = Color.ROADMAP.value
    image[raster.baseline_paths_layer[0] > 0] = Color.BASELINE_PATHS.value
    image[raster.agents_layer.squeeze() > 0] = Color.AGENTS.value  # squeeze to shape of W*H only
    image[raster.ego_layer.squeeze() > 0] = Color.EGO.value

    # Draw predicted and target trajectories
    if target_trajectory is not None:
        _draw_trajectory(image, target_trajectory, Color.TARGET_TRAJECTORY, pixel_size, 2, 1)
    if predicted_trajectory is not None:
        _draw_trajectory(image, predicted_trajectory, Color.PREDICTED_TRAJECTORY, pixel_size, 2, 1)

    return image
