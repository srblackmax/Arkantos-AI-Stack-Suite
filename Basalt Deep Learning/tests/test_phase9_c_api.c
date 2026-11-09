/*
 * Copyright (C) 2026 Kival
 * This file is part of the Arkantos AI Stack Suite.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <https://www.gnu.org/licenses/>.
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <basalt/basalt_c.h>

static void
check_status (basalt_status_t st, const char *ctx)
{
  if (st != BASALT_STATUS_OK)
    {
      fprintf (stderr, "C API failure in %s: %s\n", ctx, basalt_last_error ());
      exit (1);
    }
}

static void
test_c_tensor_autograd (void)
{
  size_t shape[2] = {2, 2};
  float a_vals[4] = {1.0f, 2.0f, 3.0f, 4.0f};
  float b_vals[4] = {2.0f, 0.5f, -1.0f, 3.0f};

  basalt_tensor_t a = NULL;
  basalt_tensor_t b = NULL;
  basalt_tensor_t prod = NULL;
  basalt_tensor_t loss = NULL;
  basalt_tensor_t grad_a = NULL;

  check_status (basalt_tensor_from_data (
                  a_vals, shape, 2, BASALT_DTYPE_FLOAT32, 1, &a),
                "from_data a");
  check_status (basalt_tensor_from_data (
                  b_vals, shape, 2, BASALT_DTYPE_FLOAT32, 0, &b),
                "from_data b");

  if (basalt_tensor_ndim (a) != 2 || basalt_tensor_numel (a) != 4)
    exit (1);

  check_status (basalt_tensor_mul (a, b, &prod), "mul");
  check_status (basalt_tensor_sum (prod, &loss), "sum");
  check_status (basalt_tensor_backward (loss), "backward");
  check_status (basalt_tensor_grad (a, &grad_a), "grad_a");

  float g_host[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  check_status (basalt_tensor_copy_to_host (grad_a, g_host, sizeof (g_host)),
                "copy_to_host");

  for (size_t i = 0; i < 4; ++i)
    {
      if (fabs ((double) g_host[i] - (double) b_vals[i]) > 1e-6)
        exit (1);
    }

  basalt_tensor_destroy (grad_a);
  basalt_tensor_destroy (loss);
  basalt_tensor_destroy (prod);
  basalt_tensor_destroy (b);
  basalt_tensor_destroy (a);
}

static void
test_c_mlp_training (void)
{
  check_status (basalt_seed_everything (42), "seed");

  size_t x_shape[2] = {4, 2};
  size_t y_shape[2] = {4, 1};
  float x_vals[8] = {0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f, 1.0f};
  float y_vals[4] = {0.0f, 1.0f, 1.0f, 2.0f};

  basalt_tensor_t x = NULL;
  basalt_tensor_t y = NULL;
  check_status (basalt_tensor_from_data (
                  x_vals, x_shape, 2, BASALT_DTYPE_FLOAT32, 0, &x),
                "x");
  check_status (basalt_tensor_from_data (
                  y_vals, y_shape, 2, BASALT_DTYPE_FLOAT32, 0, &y),
                "y");

  basalt_mlp_t mlp = NULL;
  check_status (basalt_mlp_create (2, 8, 1, 0.05, &mlp), "mlp_create");

  double first_loss = 0.0;
  double last_loss = 0.0;
  for (int step = 0; step < 80; ++step)
    {
      double l = 0.0;
      check_status (basalt_mlp_train_step (mlp, x, y, &l), "train_step");
      if (step == 0)
        first_loss = l;
      last_loss = l;
    }

  if (!(last_loss < first_loss * 0.2))
    {
      fprintf (stderr, "C MLP did not converge: %f -> %f\n",
               first_loss, last_loss);
      exit (1);
    }

  const char *ckpt = "c_api_mlp.bckpt";
  check_status (basalt_mlp_save (mlp, ckpt), "mlp_save");
  check_status (basalt_mlp_load (mlp, ckpt), "mlp_load");
  remove (ckpt);

  basalt_mlp_destroy (mlp);
  basalt_tensor_destroy (y);
  basalt_tensor_destroy (x);
}

int
main (void)
{
  if (strcmp (basalt_version_string (), BASALT_VERSION_STRING) != 0)
    return 1;
  test_c_tensor_autograd ();
  test_c_mlp_training ();
  return 0;
}
