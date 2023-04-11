from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from nuplan.planning.script.builders.utils.utils_type import validate_type
from nuplan.planning.training.preprocessing.feature_builders.abstract_feature_builder import AbstractModelFeature
from nuplan.planning.training.preprocessing.features.abstract_model_feature import FeatureDataType, to_tensor


@dataclass
class TensorTarget(AbstractModelFeature):
    """
    Dataclass that holds trajectory signals produced from the model or from the dataset for supervision.
    :param data: either a [num_batches, num_states, 3] or [num_states, 3] representing the trajectory
                 where se2_state is [x, y, heading] with units [meters, meters, radians].
    """

    data: FeatureDataType

    # def __post_init__(self) -> None:
    #     """Sanitize attributes of the dataclass."""
    #     state_size = self.data.shape[-1]

    #     if (state_size != 2) and (state_size != 4):
    #         raise RuntimeError(f'Invalid tensor target data. Expected 2 (mode_probs) or 4 (pred) on last dimension, got {state_size}.')

    @cached_property
    def is_valid(self) -> bool:
        """Inherited, see superclass."""
        return True

    def to_device(self, device: torch.device) -> TensorTarget:
        """Implemented. See interface."""
        validate_type(self.data, torch.Tensor)
        return TensorTarget(data=self.data.to(device=device))

    def to_feature_tensor(self) -> TensorTarget:
        """Inherited, see superclass."""
        return TensorTarget(data=to_tensor(self.data))

    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> TensorTarget:
        """Implemented. See interface."""
        return TensorTarget(data=data["data"])

    def unpack(self) -> List[TensorTarget]:
        """Implemented. See interface."""
        return [TensorTarget(data[None]) for data in self.data]

    @property
    def num_dimensions(self) -> int:
        """
        :return: dimensions of underlying data
        """
        return len(self.data.shape)

    @property
    def num_batches(self) -> Optional[int]:
        """
        :return: number of batches in the trajectory, None if trajectory does not have batch dimension
        """
        return None if self.num_dimensions <= 2 else self.data.shape[0]

