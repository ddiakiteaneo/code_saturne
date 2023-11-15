/*============================================================================
 * Gradient reconstruction, CUDA implementations.
 *============================================================================*/

/*
  This file is part of code_saturne, a general-purpose CFD tool.

  Copyright (C) 1998-2023 EDF S.A.

  This program is free software; you can redistribute it and/or modify it under
  the terms of the GNU General Public License as published by the Free Software
  Foundation; either version 2 of the License, or (at your option) any later
  version.

  This program is distributed in the hope that it will be useful, but WITHOUT
  ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
  FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
  details.

  You should have received a copy of the GNU General Public License along with
  this program; if not, write to the Free Software Foundation, Inc., 51 Franklin
  Street, Fifth Floor, Boston, MA 02110-1301, USA.
*/

/*----------------------------------------------------------------------------*/


__global__ static void
_compute_rhs_lsq_v_i_face_step1(cs_lnum_t            n_i_faces,
                          const cs_lnum_2_t      *restrict i_face_cells,
                          const cs_real_3_t    *restrict cell_f_cen,
                          cs_real_33_t         *restrict fctb,
                          const cs_real_3_t    *restrict pvar,
                          const cs_real_t         *restrict weight,
                          const cs_real_t      *restrict c_weight)
{
  for (
    cs_lnum_t f_id = blockIdx.x * blockDim.x + threadIdx.x;
    f_id < n_i_faces;
    f_id += blockDim.x * gridDim.x
  ) {
    cs_lnum_t c_id1 = i_face_cells[f_id][0];
    cs_lnum_t c_id2 = i_face_cells[f_id][1];

    cs_real_3_t dc;

    dc[0] = cell_f_cen[c_id2][0] - cell_f_cen[c_id1][0];
    dc[1] = cell_f_cen[c_id2][1] - cell_f_cen[c_id1][1];
    dc[2] = cell_f_cen[c_id2][2] - cell_f_cen[c_id1][2];
    cs_real_t ddc = 1. / (dc[0] * dc[0] + dc[1] * dc[1] + dc[2] * dc[2]);

    cs_real_t weight1 = 1., weight2 = 1.;

    if (c_weight) {
      weight1 = c_weight[c_id1];
      weight2 = c_weight[c_id2];

      cs_real_t pond  = weight[f_id];
      ddc /= (pond * weight1 + (1. - pond) * weight2);
    }

    for (cs_lnum_t i = 0; i < 3; i++) {
      cs_real_t pfac = (pvar[c_id2][i] - pvar[c_id1][i]) * ddc;
      for (cs_lnum_t j = 0; j < 3; j++) {
        fctb[f_id][i][j] = dc[j] * pfac;
      }
    }
  }
}

__global__ static void
_compute_rhs_lsq_v_i_face_step2(cs_lnum_t n_cells,
                                const cs_lnum_t *restrict cell_cells_idx,
                                const cs_lnum_t *restrict cell_cells,
                                const cs_lnum_t *restrict cell_i_faces,
                                const cs_real_33_t *restrict fctb,
                                cs_real_33_t *restrict rhs,
                                const cs_real_t *restrict c_weight)
{
  for (
    cs_lnum_t c_id1 = blockIdx.x * blockDim.x + threadIdx.x;
    c_id1 < n_cells;
    c_id1 += blockDim.x * gridDim.x
  ) {
    cs_lnum_t s_id = cell_cells_idx[c_id1];
    cs_lnum_t e_id = cell_cells_idx[c_id1 + 1];

    cs_real_33_t rhs1 = { { 0., 0., 0. }, { 0., 0., 0. }, { 0., 0., 0. } };

    for (cs_lnum_t index = s_id; index < e_id; index++)
    {
      cs_lnum_t f_id  = cell_i_faces[index];
      cs_real_t w     = 1.;
      if (c_weight) {
        w = c_weight[cell_cells[index]];
      }

      for (cs_lnum_t i = 0; i < 3; i++) {
        for (cs_lnum_t j = 0; j < 3; j++) {
          rhs1[i][j] += w * fctb[f_id][i][j];
        }
      }
    }
    for (cs_lnum_t i = 0; i < 3; i++) {
      for (cs_lnum_t j = 0; j < 3; j++) {
        rhs[c_id1][i][j] = rhs1[i][j];
      }
    }
  }
}

__global__ static void
_compute_rhs_lsq_v_i_face_step2_v2(cs_lnum_t n_cells,
                                const cs_lnum_t *restrict cell_cells_idx,
                                const cs_lnum_t *restrict cell_cells,
                                const cs_lnum_t *restrict cell_i_faces,
                                const cs_real_33_t *restrict fctb,
                                cs_real_33_t *restrict rhs,
                                const cs_real_t *restrict c_weight)
{
  for (cs_lnum_t x = blockIdx.x * blockDim.x + threadIdx.x; x < n_cells * 9;
       x += blockDim.x * gridDim.x) {
    cs_lnum_t c_id1 = x / 9;
    cs_lnum_t i     = (x / 3) % 3;
    cs_lnum_t j     = x % 3;

    cs_lnum_t s_id  = cell_cells_idx[c_id1];
    cs_lnum_t e_id = cell_cells_idx[c_id1 + 1];

    cs_real_t rhs1 = 0.;

    for (cs_lnum_t index = s_id; index < e_id; index++) {
      cs_lnum_t f_id = cell_i_faces[index];
      cs_real_t w    = 1.;
      if (c_weight) {
        w = c_weight[cell_cells[index]];
      }

      rhs1 += w * fctb[f_id][i][j];
    }
    rhs[c_id1][i][j] = rhs1;
  }
}
