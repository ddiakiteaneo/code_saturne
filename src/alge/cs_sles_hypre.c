/*============================================================================
 * Sparse Linear Equation Solvers using HYPRE
 *============================================================================*/

/*
  This file is part of Code_Saturne, a general-purpose CFD tool.

  Copyright (C) 1998-2021 EDF S.A.

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

#include "cs_defs.h"

/*----------------------------------------------------------------------------
 * Standard C library headers
 *----------------------------------------------------------------------------*/

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>

#if defined(HAVE_MPI)
#include <mpi.h>
#endif

/*----------------------------------------------------------------------------
 * HYPRE headers
 *----------------------------------------------------------------------------*/

#include <HYPRE_krylov.h>
#include <HYPRE_parcsr_ls.h>
#include <HYPRE_utilities.h>

/*----------------------------------------------------------------------------
 * Local headers
 *----------------------------------------------------------------------------*/

#include "bft_mem.h"
#include "bft_error.h"
#include "bft_printf.h"

#include "cs_base.h"
#include "cs_log.h"
#include "cs_fp_exception.h"
#include "cs_halo.h"
#include "cs_matrix.h"
#include "cs_matrix_default.h"
#include "cs_matrix_hypre.h"
#include "cs_matrix_hypre_priv.h"
#include "cs_timer.h"

/*----------------------------------------------------------------------------
 *  Header for the current file
 *----------------------------------------------------------------------------*/

#include "cs_sles.h"
#include "cs_sles_hypre.h"

/*----------------------------------------------------------------------------*/

BEGIN_C_DECLS

/*=============================================================================
 * Additional doxygen documentation
 *============================================================================*/

/*!
  \file cs_sles_hypre.c

  \brief handling of HYPRE-based linear solvers

  \page sles_hypre HYPRE-based linear solvers.

  \typedef cs_sles_hypre_setup_hook_t

  \brief Function pointer for settings of a HYPRE solver setup.

  This function is called during the setup stage for a HYPRE solver.

  When first called, the solver argument is NULL, and must be created
  using HYPRE functions.

  Note: if the context pointer is non-NULL, it must point to valid data
  when the selection function is called so that value or structure should
  not be temporary (i.e. local);

  \param[in]       verbosity  verbosity level
  \param[in, out]  context    pointer to optional (untyped) value or structure
  \param[in, out]  solver     handle to HYPRE solver (to be cast as HYPRE_Solver)
*/

/*! \cond DOXYGEN_SHOULD_SKIP_THIS */

/*=============================================================================
 * Local Macro Definitions
 *============================================================================*/

/*=============================================================================
 * Local Structure Definitions
 *============================================================================*/

/* Basic per linear system options and logging */
/*---------------------------------------------*/

typedef struct _cs_sles_hypre_setup_t {

  cs_matrix_coeffs_hypre_t *coeffs;  /* HYPRE matrix and vectors */

  HYPRE_Solver          solver;            /* Solver data */
  HYPRE_Solver          precond;           /* Preconditioner data */

} cs_sles_hypre_setup_t;

typedef struct _cs_sles_hypre_t {

  cs_sles_hypre_type_t  solver_type;       /* Solver type */
  cs_sles_hypre_type_t  precond_type;      /* Preconditioner type */

  /* Performance data */

  int                  n_setups;           /* Number of times system setup */
  int                  n_solves;           /* Number of times system solved */

  int                  n_iterations_last;  /* Number of iterations for last
                                              system resolution */
  int                  n_iterations_min;   /* Minimum number of iterations
                                              in system resolution history */
  int                  n_iterations_max;   /* Maximum number of iterations
                                              in system resolution history */
  int long long        n_iterations_tot;   /* Total accumulated number of
                                              iterations */

  cs_timer_counter_t   t_setup;            /* Total setup */
  cs_timer_counter_t   t_solve;            /* Total time used */

  /* Additional setup options */

  void                        *hook_context;   /* Optional user context */
  cs_sles_hypre_setup_hook_t  *setup_hook;     /* Post setup function */

  cs_sles_hypre_setup_t     *setup_data;

} cs_sles_hypre_t;

/*============================================================================
 * Private function definitions
 *============================================================================*/

/*----------------------------------------------------------------------------*/
/*!
 * \brief Return name of hypre solver type.
 *
 * \param[in]      solver_type   solver type id
 *
 * \return  name od associated solver type.
 */
/*----------------------------------------------------------------------------*/

static const char *
_cs_hypre_type_name(cs_sles_hypre_type_t  solver_type)
{
  switch(solver_type) {

  case CS_SLES_HYPRE_BOOMERAMG:
    return N_("BoomerAMG");
    break;
  case CS_SLES_HYPRE_HYBRID:
    return N_("Hybrid");
    break;
  case CS_SLES_HYPRE_ILU:
    return N_("ILU");
    break;
  case CS_SLES_HYPRE_BICGSTAB:
    return N_("BiCGSTAB");
    break;
  case CS_SLES_HYPRE_GMRES:
    return N_("GMRES");
    break;
  case CS_SLES_HYPRE_FLEXGMRES:
    return N_("Flexible GMRES");
    break;
  case CS_SLES_HYPRE_LGMRES:
    return N_("LGMRES");
    break;
  case CS_SLES_HYPRE_PCG:
    return N_("PCG");
    break;
  case CS_SLES_HYPRE_EUCLID:
    return N_("EUCLID");
    break;
  case CS_SLES_HYPRE_PARASAILS:
    return N_("ParaSails");
    break;
  case CS_SLES_HYPRE_NONE:
    return N_("None");
    break;

  default:
    return NULL;
  }
}

/*============================================================================
 * Public function definitions
 *============================================================================*/

/*----------------------------------------------------------------------------*/
/*!
 * \brief Define and associate a HYPRE linear system solver
 *        for a given field or equation name.
 *
 * If this system did not previously exist, it is added to the list of
 * "known" systems. Otherwise, its definition is replaced by the one
 * defined here.
 *
 * This is a utility function: if finer control is needed, see
 * \ref cs_sles_define and \ref cs_sles_petsc_create.
 *
 * The associated solver required that the matrix passed to it is a HYPRE
 * matrix (see cs_matrix_set_type_hypre).
 *
 * Note that this function returns a pointer directly to the iterative solver
 * management structure. This may be used to set further options.
 * If needed, \ref cs_sles_find may be used to obtain a pointer to the matching
 * \ref cs_sles_t container.
 *
 * \param[in]      f_id          associated field id, or < 0
 * \param[in]      name          associated name if f_id < 0, or NULL
 * \param[in]      solver_type   HYPRE solver type
 * \param[in]      precond_type  HYPRE preconditioner type
 * \param[in]      setup_hook    pointer to optional setup epilogue function
 * \param[in,out]  context       pointer to optional (untyped) value or
 *                               structure for setup_hook, or NULL
 *
 * \return  pointer to newly created iterative solver info object.
 */
/*----------------------------------------------------------------------------*/

cs_sles_hypre_t *
cs_sles_hypre_define(int                          f_id,
                     const char                  *name,
                     cs_sles_hypre_type_t         solver_type,
                     cs_sles_hypre_type_t         precond_type,
                     cs_sles_hypre_setup_hook_t  *setup_hook,
                     void                        *context)
{
  if (solver_type < 0 || solver_type >= CS_SLES_HYPRE_NONE)
    bft_error(__FILE__, __LINE__, 0,
              _("Incorrect solver type argument %d for HYPRE."),
              (int)solver_type);

  cs_sles_hypre_t *c = cs_sles_hypre_create(solver_type,
                                            precond_type,
                                            setup_hook,
                                            context);

  cs_sles_t *sc = cs_sles_define(f_id,
                                 name,
                                 c,
                                 "cs_sles_hypre_t",
                                 cs_sles_hypre_setup,
                                 cs_sles_hypre_solve,
                                 cs_sles_hypre_free,
                                 cs_sles_hypre_log,
                                 cs_sles_hypre_copy,
                                 cs_sles_hypre_destroy);

  cs_sles_set_error_handler(sc,
                            cs_sles_hypre_error_post_and_abort);

  return c;
}

/*----------------------------------------------------------------------------*/
/*!
 * \brief Create HYPRE linear system solver info and context.
 *
 * In case of rotational periodicity for a block (non-scalar) matrix,
 * the matrix type will be forced to MATSHELL ("shell") regardless
 * of the option used.
 *
 * \param[in]  solver_type   HYPRE solver type
 * \param[in]  precond_type  HYPRE preconditioner type
 * \param[in]  setup_hook    pointer to optional setup epilogue function
 * \param[in]  context       pointer to optional (untyped) value or structure
 *                           for setup_hook, or NULL
 *
 * \return  pointer to newly created linear system object.
 */
/*----------------------------------------------------------------------------*/

cs_sles_hypre_t *
cs_sles_hypre_create(cs_sles_hypre_type_t         solver_type,
                     cs_sles_hypre_type_t         precond_type,
                     cs_sles_hypre_setup_hook_t  *setup_hook,
                     void                        *context)
{
  cs_sles_hypre_t *c;

  BFT_MALLOC(c, 1, cs_sles_hypre_t);

  c->solver_type = solver_type;
  c->precond_type = precond_type;

  c->n_setups = 0;
  c->n_solves = 0;
  c->n_iterations_last = 0;
  c->n_iterations_min = 0;
  c->n_iterations_max = 0;
  c->n_iterations_tot = 0;

  CS_TIMER_COUNTER_INIT(c->t_setup);
  CS_TIMER_COUNTER_INIT(c->t_solve);

  /* Options */

  c->hook_context = context;
  c->setup_hook = setup_hook;

  /* Setup data */
  c->setup_data = NULL;

  return c;
}

/*----------------------------------------------------------------------------*/
/*!
 * \brief Destroy iterative sparse linear system solver info and context.
 *
 * \param[in, out]  context  pointer to iterative solver info and context
 *                           (actual type: cs_sles_hypre_t  **)
 */
/*----------------------------------------------------------------------------*/

void
cs_sles_hypre_destroy(void **context)
{
  cs_sles_hypre_t *c = (cs_sles_hypre_t *)(*context);
  if (c != NULL) {
    cs_sles_hypre_free(c);
    BFT_FREE(c);
    *context = c;
  }
}

/*----------------------------------------------------------------------------*/
/*!
 * \brief Create HYPRE sparse linear system solver info and context
 *        based on existing info and context.
 *
 * \param[in]  context  pointer to reference info and context
 *                     (actual type: cs_sles_hypre_t  *)
 *
 * \return  pointer to newly created solver info object.
 *          (actual type: cs_sles_hypre_t  *)
 */
/*----------------------------------------------------------------------------*/

void *
cs_sles_hypre_copy(const void  *context)
{
  cs_sles_hypre_t *d = NULL;

  if (context != NULL) {
    const cs_sles_hypre_t *c = context;
    d = cs_sles_hypre_create(c->solver_type,
                             c->precond_type,
                             c->setup_hook,
                             c->hook_context);
  }

  return d;
}

/*----------------------------------------------------------------------------*/
/*!
 * \brief Error handler for HYPRE solver.
 *
 * In case of divergence or breakdown, this error handler outputs an error
 * message
 * It does nothing in case the maximum iteration count is reached.

 * \param[in, out]  sles           pointer to solver object
 * \param[in]       state          convergence state
 * \param[in]       a              matrix
 * \param[in]       rhs            right hand side
 * \param[in, out]  vx             system solution
 *
 * \return  false (do not attempt new solve)
 */
/*----------------------------------------------------------------------------*/

bool
cs_sles_hypre_error_post_and_abort(cs_sles_t                    *sles,
                                   cs_sles_convergence_state_t   state,
                                   const cs_matrix_t            *a,
                                   const cs_real_t              *rhs,
                                   cs_real_t                    *vx)
{
  CS_UNUSED(a);
  CS_UNUSED(rhs);
  CS_UNUSED(vx);

  if (state >= CS_SLES_BREAKDOWN)
    return false;

  const char *name = cs_sles_get_name(sles);

  const cs_sles_hypre_t  *c = cs_sles_get_context(sles);
  CS_UNUSED(c);

  const char *error_type[] = {N_("divergence"), N_("breakdown")};
  int err_id = (state == CS_SLES_BREAKDOWN) ? 1 : 0;

  bft_error(__FILE__, __LINE__, 0,
            _("HYPRE: error (%s) solving for %s"),
            _(error_type[err_id]),
            name);

  return false;
}

/*----------------------------------------------------------------------------*/
/*!
 * \brief Log sparse linear equation solver info.
 *
 * \param[in]  context   pointer to iterative solver info and context
 *                       (actual type: cs_sles_hypre_t  *)
 * \param[in]  log_type  log type
 */
/*----------------------------------------------------------------------------*/

void
cs_sles_hypre_log(const void  *context,
                  cs_log_t     log_type)
{
  const cs_sles_hypre_t  *c = context;

  if (log_type == CS_LOG_SETUP) {

    cs_log_printf(log_type,
                  _("  Solver type:                       HYPRE (%s)\n"),
                  _cs_hypre_type_name(c->solver_type));
    if (c->precond_type < CS_SLES_HYPRE_NONE)
      cs_log_printf(log_type,
                    _("    Preconditioning:                 %s\n"),
                    _cs_hypre_type_name(c->precond_type));

  }

  else if (log_type == CS_LOG_PERFORMANCE) {

    int n_calls = c->n_solves;
    int n_it_min = c->n_iterations_min;
    int n_it_max = c->n_iterations_max;
    int n_it_mean = 0;

    if (n_calls > 0)
      n_it_mean = (int)(  c->n_iterations_tot
          / ((int long long)n_calls));

    cs_log_printf(log_type,
                  _("\n"
                    "  Solver type:                   HYPRE (%s)\n"),
                  _cs_hypre_type_name(c->solver_type));
    if (c->precond_type < CS_SLES_HYPRE_NONE)
      cs_log_printf(log_type,
                    _("    Preconditioning:             %s\n"),
                    _cs_hypre_type_name(c->precond_type));
    cs_log_printf(log_type,
                  _("  Number of setups:              %12d\n"
                    "  Number of calls:               %12d\n"
                    "  Minimum number of iterations:  %12d\n"
                    "  Maximum number of iterations:  %12d\n"
                    "  Mean number of iterations:     %12d\n"
                    "  Construction:                  %12.3f\n"
                    "  Resolution:                    %12.3f\n"),
                  c->n_setups, n_calls, n_it_min, n_it_max, n_it_mean,
                  c->t_setup.nsec*1e-9,
                  c->t_solve.nsec*1e-9);

  }
}

/*----------------------------------------------------------------------------*/
/*!
 * \brief Setup iterative sparse linear equation solver.
 *
 * \param[in, out]  context    pointer to iterative solver info and context
 *                             (actual type: cs_sles_hypre_t  *)
 * \param[in]       name       pointer to system name
 * \param[in]       a          associated matrix
 * \param[in]       verbosity  associated verbosity
 */
/*----------------------------------------------------------------------------*/

void
cs_sles_hypre_setup(void               *context,
                    const char         *name,
                    const cs_matrix_t  *a,
                    int                 verbosity)
{
  CS_UNUSED(name);

  cs_timer_t t0;
  t0 = cs_timer_time();

  cs_sles_hypre_t  *c  = context;
  cs_sles_hypre_setup_t *sd = c->setup_data;
  if (sd == NULL) {
    BFT_MALLOC(c->setup_data, 1, cs_sles_hypre_setup_t);
    sd = c->setup_data;
    sd->solver = NULL;
    sd->precond = NULL;
  }

  const char expected_matrix_type[] = "HYPRE_PARCSR";
  if (strcmp(cs_matrix_get_type_name(a), expected_matrix_type))
    bft_error(__FILE__, __LINE__, 0,
              _("HYPRE [%s]:\n"
                "  expected matrix type: %s\n"
                "  provided matrix type: %s"),
              name, expected_matrix_type, cs_matrix_get_type_name(a));

  sd->coeffs = cs_matrix_hypre_get_coeffs(a);

  MPI_Comm comm = cs_glob_mpi_comm;
  if (comm == MPI_COMM_NULL)
    comm = MPI_COMM_WORLD;

  for (int i = 0; i < 2; i++) {

    HYPRE_Solver hs = (i == 0) ? sd->precond : sd->solver;
    cs_sles_hypre_type_t hs_type = (i == 0) ? c->precond_type : c->solver_type;

    /* hs should be NULL at this point, unless we do not relly free
     it (when cs_sles_hypre_free is called (for example to amortize setup) */

    HYPRE_PtrToParSolverFcn precond_solve = NULL;
    HYPRE_PtrToParSolverFcn precond_setup = NULL;

    if (hs != NULL || hs_type >= CS_SLES_HYPRE_NONE)
      continue;

    switch(hs_type) {

    case CS_SLES_HYPRE_BOOMERAMG:
      {
        HYPRE_BoomerAMGCreate(&hs);

        HYPRE_BoomerAMGSetCoarsenType(hs, 10);        /* HMIS */
        HYPRE_BoomerAMGSetAggNumLevels(hs, 2);
        HYPRE_BoomerAMGSetPMaxElmts(hs, 4);
        HYPRE_BoomerAMGSetInterpType(hs, 7);          /* extended+i */
        HYPRE_BoomerAMGSetStrongThreshold(hs, 0.5);   /* 2d=>0.25 3d=>0.5 */
        HYPRE_BoomerAMGSetRelaxType(hs, 6);   /* Sym G.S./Jacobi hybrid */
        HYPRE_BoomerAMGSetRelaxOrder(hs, 0);

        if (i == 0) { /* preconditioner */
          HYPRE_BoomerAMGSetTol(hs, 0.0);
          HYPRE_BoomerAMGSetMaxIter(hs, 1);
          precond_solve = HYPRE_BoomerAMGSolve;
          precond_setup = HYPRE_BoomerAMGSetup;
        }
        else { /* solver */
          HYPRE_BoomerAMGSetMaxIter(hs, 1000);
        }

        if (verbosity > 2 ) {
          HYPRE_BoomerAMGSetPrintLevel(hs, 1);
          HYPRE_BoomerAMGSetPrintFileName(hs, "hypre.log");
        }
      }
      break;

    case CS_SLES_HYPRE_PCG:
      {
        HYPRE_ParCSRPCGCreate(comm, &hs);
        HYPRE_PCGSetMaxIter(hs, 1000);  /* Max iterations */

        if (verbosity > 2 ) {
          HYPRE_PCGSetPrintLevel(hs, 2);  /* Print solve info */
          HYPRE_PCGSetLogging(hs, 1);     /* Needed to get run info later */
        }

        if (precond_solve != NULL) {
          HYPRE_ParCSRPCGSetPrecond(hs,
                                    precond_solve,
                                    precond_setup,
                                    sd->precond);
        }
      }
      break;

    default:
      assert(0);

    }

    if (i == 0)
      sd->precond = hs;
    else
      sd->solver = hs;
  }

  /* Call optional setup hook for user setting changes */

  if (c->setup_hook != NULL)
    c->setup_hook(verbosity, c->hook_context, &(sd->solver));

  /* Now setup systems (where rhs and vx values may be different
     when solving, but their shapes and addresses are the same) */

  HYPRE_ParCSRMatrix par_a;              /* Associted matrix */
  HYPRE_ParVector p_x, p_rhs;

  HYPRE_IJMatrixGetObject(sd->coeffs->hm, (void **)&par_a);
  HYPRE_IJVectorGetObject(sd->coeffs->hx, (void **)&p_x);
  HYPRE_IJVectorGetObject(sd->coeffs->hy, (void **)&p_rhs);

  switch(c->solver_type) {

  case CS_SLES_HYPRE_BOOMERAMG:
    HYPRE_BoomerAMGSetup(sd->solver, par_a, p_rhs, p_x);
    break;

  case CS_SLES_HYPRE_PCG:
    HYPRE_ParCSRPCGSetup(sd->solver, par_a, p_rhs, p_x);
    break;

  default:
    bft_error(__FILE__, __LINE__, 0,
              _("HYPRE: solver type (%s) not handled yet."),
              _cs_hypre_type_name(c->solver_type));
  }

  /* Update return values */
  c->n_setups += 1;

  cs_timer_t t1 = cs_timer_time();
  cs_timer_counter_add_diff(&(c->t_setup), &t0, &t1);
}

/*----------------------------------------------------------------------------*/
/*!
 * \brief Call HYPRE linear equation solver.
 *
 * \param[in, out]  context        pointer to iterative solver info and context
 *                                 (actual type: cs_sles_hypre_t  *)
 * \param[in]       name           pointer to system name
 * \param[in]       a              matrix
 * \param[in]       verbosity      associated verbosity
 * \param[in]       precision      solver precision
 * \param[in]       r_norm         residue normalization
 * \param[out]      n_iter         number of "equivalent" iterations
 * \param[out]      residue        residue
 * \param[in]       rhs            right hand side
 * \param[in, out]  vx             system solution
 * \param[in]       aux_size       number of elements in aux_vectors (in bytes)
 * \param           aux_vectors    optional working area
 *                                 (internal allocation if NULL)
 *
 * \return  convergence state
 */
/*----------------------------------------------------------------------------*/

cs_sles_convergence_state_t
cs_sles_hypre_solve(void                *context,
                    const char          *name,
                    const cs_matrix_t   *a,
                    int                  verbosity,
                    double               precision,
                    double               r_norm,
                    int                 *n_iter,
                    double              *residue,
                    const cs_real_t     *rhs,
                    cs_real_t           *vx,
                    size_t               aux_size,
                    void                *aux_vectors)
{
  CS_UNUSED(aux_size);
  CS_UNUSED(aux_vectors);

  cs_timer_t t0;
  t0 = cs_timer_time();
  cs_sles_convergence_state_t cvg = CS_SLES_ITERATING;
  cs_sles_hypre_t  *c = context;
  cs_sles_hypre_setup_t  *sd = c->setup_data;
  cs_lnum_t n_rows = cs_matrix_get_n_rows(a);
  int its;
  double res;

  precision = precision;
  if (sd == NULL) {
    cs_sles_hypre_setup(c, name, a, verbosity);
    sd = c->setup_data;
  }

  HYPRE_ParCSRMatrix par_a;              /* Associted matrix */
  HYPRE_IJMatrixGetObject(sd->coeffs->hm, (void **)&par_a);

  HYPRE_Real *_t = NULL;

  /* Set RHS and starting solution */

  if (sizeof(cs_real_t) == sizeof(HYPRE_Real)) {
    HYPRE_IJVectorSetValues(sd->coeffs->hx, n_rows, NULL, vx);
    HYPRE_IJVectorSetValues(sd->coeffs->hy, n_rows, NULL, rhs);
  }
  else {
    BFT_MALLOC(_t, n_rows, HYPRE_Real);
    for (HYPRE_BigInt ii = 0; ii < n_rows; ii++) {
      _t[ii] = vx[ii];;
    }
    HYPRE_IJVectorSetValues(sd->coeffs->hx, n_rows, NULL, _t);
    for (HYPRE_BigInt ii = 0; ii < n_rows; ii++) {
      _t[ii] = rhs[ii];;
    }
    HYPRE_IJVectorSetValues(sd->coeffs->hy, n_rows, NULL, _t);
  }

  HYPRE_IJVectorAssemble(sd->coeffs->hx);
  HYPRE_IJVectorAssemble(sd->coeffs->hy);

  HYPRE_ParVector p_x, p_rhs;
  HYPRE_IJVectorGetObject(sd->coeffs->hx, (void **)&p_x);
  HYPRE_IJVectorGetObject(sd->coeffs->hy, (void **)&p_rhs);

  switch(c->solver_type) {

  case CS_SLES_HYPRE_BOOMERAMG:
    {
      /* Finalize setup and solve; no absolute tolerance is available,
         so we use the available function. */
      HYPRE_BoomerAMGSetTol(sd->solver, precision*r_norm);
      HYPRE_BoomerAMGSolve(sd->solver, par_a, p_rhs, p_x);

      /* Get solution and information */
      HYPRE_BoomerAMGGetFinalRelativeResidualNorm(sd->solver, &res);
      HYPRE_BoomerAMGGetNumIterations(sd->solver, &its);
    }
    break;

  case CS_SLES_HYPRE_PCG:
    {
      /* Finalize setup and solve */
      HYPRE_PCGSetAbsoluteTol(sd->solver, precision*r_norm);
      HYPRE_ParCSRPCGSolve(sd->solver, par_a, p_rhs, p_x);

      /* Get solution and information */
      HYPRE_ParCSRPCGGetFinalRelativeResidualNorm(sd->solver, &res);
      HYPRE_ParCSRPCGGetNumIterations(sd->solver, &its);
    }
    break;

  default:
    bft_error(__FILE__, __LINE__, 0,
              _("HYPRE: solver type (%s) not handled yet."),
              _cs_hypre_type_name(c->solver_type));
  }

  if (sizeof(cs_real_t) == sizeof(HYPRE_Real)) {
    HYPRE_IJVectorGetValues(sd->coeffs->hx, n_rows, NULL, vx);
  }
  else {
    HYPRE_IJVectorGetValues(sd->coeffs->hx, n_rows, NULL, _t);
    for (HYPRE_BigInt ii = 0; ii < n_rows; ii++) {
      vx[ii] = _t[ii];
    }
    BFT_FREE(_t);
  }

  *residue = res;
  *n_iter = its;

  /* Update return values */
  if (c->n_solves == 0)
    c->n_iterations_min = its;

  c->n_iterations_last = its;
  c->n_iterations_tot += its;
  if (c->n_iterations_min > its)
    c->n_iterations_min = its;
  if (c->n_iterations_max < its)
    c->n_iterations_max = its;
  c->n_solves += 1;
  cs_timer_t t1 = cs_timer_time();
  cs_timer_counter_add_diff(&(c->t_solve), &t0, &t1);

  return cvg;
}

/*----------------------------------------------------------------------------*/
/*!
 * \brief Free HYPRE linear equation solver setup context.
 *
 * This function frees resolution-related data, such as
 * buffers and preconditioning but does not free the whole context,
 * as info used for logging (especially performance data) is maintained.
 *
 * \param[in, out]  context  pointer to iterative solver info and context
 *                           (actual type: cs_sles_hypre_t  *)
 */
/*----------------------------------------------------------------------------*/

void
cs_sles_hypre_free(void  *context)
{
  cs_timer_t t0;
  t0 = cs_timer_time();

  cs_sles_hypre_t  *c  = context;

  if (c->setup_data != NULL) {

    cs_sles_hypre_setup_t *sd = c->setup_data;

    for (int i = 0; i < 2; i++) {
      HYPRE_Solver hs = (i == 0) ? sd->solver : sd->precond;
      cs_sles_hypre_type_t hs_type = (i == 0) ? c->solver_type : c->precond_type;

      if (hs == NULL)
        continue;

      switch(hs_type) {

      case CS_SLES_HYPRE_BOOMERAMG:
        HYPRE_BoomerAMGDestroy(hs);
        break;

      case CS_SLES_HYPRE_PCG:
        HYPRE_ParCSRPCGDestroy(hs);
        break;

      default:
        assert(0);

      }

      if (i == 0)
        sd->solver = NULL;
      else
        sd->precond = NULL;
    }

    BFT_FREE(c->setup_data);
  }

  cs_timer_t t1 = cs_timer_time();
  cs_timer_counter_add_diff(&(c->t_setup), &t0, &t1);
}

/*----------------------------------------------------------------------------*/

END_C_DECLS
