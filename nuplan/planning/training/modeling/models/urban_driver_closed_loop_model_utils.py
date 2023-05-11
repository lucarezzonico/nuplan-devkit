from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.nn import functional as F

import numpy as np


def transform_points(
    element: torch.Tensor, matrix: torch.Tensor, avail: torch.Tensor, yaw: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Transform points element using the translation tr. Reapply avail afterwards to
    ensure we don't generate any "ghosts" in the past

    Args:
        element (torch.Tensor): tensor with points to transform (B,N,P,3)
        matrix (torch.Tensor): Bx3x3 RT matrices
        avail (torch.Tensor): the availability of element
        yaw (Optional[torch.Tensor]): optional yaws of the rotation matrices to apply to yaws in element

    Returns:
        torch.Tensor: the transformed tensor
    """
    tr = matrix[:, :-1, -1:].view(-1, 1, 1, 2)
    rot = matrix[:, None, :2, :2].transpose(2, 3)  # NOTE: required because we post-multiply

    # NOTE: before we did this differently - why?
    transformed_xy = element[..., :2] @ rot + tr
    transformed_yaw = element[..., 2:3]
    if yaw is not None:
        transformed_yaw = element[..., 2:3] + yaw.view(-1, 1, 1, 1)

    element = torch.cat([transformed_xy, transformed_yaw], dim=3)
    element = element * avail[..., None].clone()  # NOTE: no idea why clone is required actually
    return element

def update_transformation_matrices(pred_xy_step_unnorm: torch.Tensor, pred_yaw_step: torch.Tensor,
                                    t0_from_ts: torch.Tensor, ts_from_t0: torch.Tensor, yaw_t0_from_ts: torch.Tensor,
                                    yaw_ts_from_t0: torch.Tensor, zero: torch.Tensor, one: torch.Tensor
                                    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """ Updates the used transformation matrices to reflect AoI's new position.
    """
    tr_tsplus_from_ts = -pred_xy_step_unnorm
    yaw_tsplus_from_ts = -pred_yaw_step
    yaw_ts_from_tsplus = pred_yaw_step

    # NOTE: these are full roto-translation matrices. We use the closed form and not invert for performance reasons.
    # tsplus_from_ts will bring the current predictions at ts into 0.
    tsplus_from_ts = torch.cat(
        [
            yaw_tsplus_from_ts.cos(),
            -yaw_tsplus_from_ts.sin(),
            tr_tsplus_from_ts[:, :1] * yaw_tsplus_from_ts.cos() - tr_tsplus_from_ts[:, 1:] * yaw_tsplus_from_ts.sin(),
            yaw_tsplus_from_ts.sin(),
            yaw_tsplus_from_ts.cos(),
            tr_tsplus_from_ts[:, :1] * yaw_tsplus_from_ts.sin() + tr_tsplus_from_ts[:, 1:] * yaw_tsplus_from_ts.cos(),
            zero,
            zero,
            one,
        ],
        dim=1,
    ).view(-1, 3, 3)
    # this is only required to keep t0_from_ts updated
    ts_from_tsplus = torch.cat(
        [
            yaw_ts_from_tsplus.cos(),
            -yaw_ts_from_tsplus.sin(),
            -tr_tsplus_from_ts[:, :1],
            yaw_ts_from_tsplus.sin(),
            yaw_ts_from_tsplus.cos(),
            -tr_tsplus_from_ts[:, 1:],
            zero,
            zero,
            one,
        ],
        dim=1,
    ).view(-1, 3, 3)

    # update RTs and yaws by including tsplus (next step ts)
    t0_from_ts = t0_from_ts @ ts_from_tsplus
    ts_from_t0 = tsplus_from_ts @ ts_from_t0
    yaw_t0_from_ts = yaw_t0_from_ts + yaw_ts_from_tsplus
    yaw_ts_from_t0 = yaw_ts_from_t0 + yaw_tsplus_from_ts

    return t0_from_ts, ts_from_t0, yaw_t0_from_ts, yaw_ts_from_t0

def build_target_normalization(nsteps: int) -> torch.Tensor:
    """Normalization coefficients approximated with 3-rd degree polynomials
    to avoid storing them explicitly, and allow changing the length

    :param nsteps: number of steps to generate normalisation for
    :type nsteps: int
    :return: XY scaling for the steps
    :rtype: torch.Tensor
    """

    normalization_polynomials = np.asarray(
        [
            # x scaling
            [3.28e-05, -0.0017684, 1.8088969, 2.211737],
            # y scaling
            [-5.67e-05, 0.0052056, 0.0138343, 0.0588579],  # manually decreased by 5
        ]
    )
    # assuming we predict x, y and yaw
    coefs = np.stack([np.poly1d(p)(np.arange(nsteps)) for p in normalization_polynomials])
    coefs = coefs.astype(np.float32)
    return torch.from_numpy(coefs).T