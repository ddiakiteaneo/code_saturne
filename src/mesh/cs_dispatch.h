#ifndef __CS_DISPATCH_H__
#define __CS_DISPATCH_H__

/*============================================================================
 * Definitions, global variables, and base functions
 *============================================================================*/

/*
  This file is part of code_saturne, a general-purpose CFD tool.

  Copyright (C) 1998-2024 EDF S.A.

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

// Valid only for C++
#ifdef __cplusplus

/*----------------------------------------------------------------------------*/

#include "cs_defs.h"
#include "cs_mesh.h"

/*----------------------------------------------------------------------------
 * Standard C library headers
 *----------------------------------------------------------------------------*/

//#include <stdio.h>

/*----------------------------------------------------------------------------
 *  Local headers
 *----------------------------------------------------------------------------*/

/*=============================================================================
 * Macro definitions
 *============================================================================*/

#ifdef __NVCC__
#define CS_CUDA_HOST __host__
#define CS_CUDA_DEVICE __device__
#define CS_CUDA_HOST_DEVICE __host__ __device__
#else
#define CS_CUDA_HOST
#define CS_CUDA_DEVICE
#define CS_CUDA_HOST_DEVICE
#endif

#define CS_HOST_FUNCTOR(capture, args, ...) cs_host_functor([CS_REMOVE_PARENTHESES(capture)] CS_CUDA_HOST args __VA_ARGS__)
#define CS_DEVICE_FUNCTOR(capture, args, ...) cs_device_functor([CS_REMOVE_PARENTHESES(capture)] CS_CUDA_DEVICE args __VA_ARGS__)
#define CS_HOST_DEVICE_FUNCTOR(capture, args, ...) cs_host_device_functor([CS_REMOVE_PARENTHESES(capture)] CS_CUDA_HOST_DEVICE args __VA_ARGS__)

/*----------------------------------------------------------------------------*/


/*============================================================================
 * Type definitions
 *============================================================================*/

template <class F>
class HostFunctor {
  private:
    F f;

  public:
    static constexpr bool is_host = true;
    static constexpr bool is_device = false;

  public:
    CS_CUDA_HOST HostFunctor(F f) noexcept(noexcept(F(std::move(f)))) : f(std::move(f)) {}

  public:
    template <class... Args>
    CS_CUDA_HOST auto operator()(Args&&... args) noexcept(noexcept(this->f(static_cast<Args&&>(args)...))) -> decltype(this->f(static_cast<Args&&>(args)...)) {
      return this->f(static_cast<Args&&>(args)...);
    }
};

template <class F>
class DeviceFunctor {
  private:
    F f;

    static_assert(sizeof(F) == 0, "Cannot create device functions without nvcc");

  public:
    static constexpr bool is_host = false;
    static constexpr bool is_device = true;

  public:
    CS_CUDA_HOST_DEVICE DeviceFunctor(F f) noexcept(noexcept(F(std::move(f)))) : f(std::move(f)) {}

  public:
    template <class... Args>
    CS_CUDA_DEVICE auto operator()(Args&&... args) noexcept(noexcept(this->f(static_cast<Args&&>(args)...))) -> decltype(this->f(static_cast<Args&&>(args)...)) {
      return this->f(static_cast<Args&&>(args)...);
    }
};

template <class F>
class HostDeviceFunctor {
  private:
    F f;

  public:
    static constexpr bool is_host = true;
    static constexpr bool is_device = true;

  public:
    CS_CUDA_HOST_DEVICE HostDeviceFunctor(F f) noexcept(noexcept(F(std::move(f)))) : f(std::move(f)) {}

  public:
    template <class... Args>
    CS_CUDA_HOST_DEVICE auto operator()(Args&&... args) noexcept(noexcept(this->f(static_cast<Args&&>(args)...))) -> decltype(this->f(static_cast<Args&&>(args)...)) {
      return this->f(static_cast<Args&&>(args)...);
    }
};

template <class Derived>
class CsContextMixin {
  public:
    template <class F, class... Args>
    void iter_(cs_lnum_t n, F* f, Args&&... args) = delete;
    template <class F, class... Args>
    decltype(auto) iter_i_faces_(const cs_mesh_t* m, F* f, Args&&... args);
    template <class F, class... Args>
    decltype(auto) iter_b_faces_(const cs_mesh_t* m, F* f, Args&&... args);
    template <class F, class... Args>
    decltype(auto) iter_cells_(const cs_mesh_t* m, F* f, Args&&... args);
  public:
    template <class F, class... Args>
    decltype(auto) iter_i_faces(const cs_mesh_t* m, F&& f, Args&&... args);
    template <class F, class... Args>
    decltype(auto) iter_b_faces(const cs_mesh_t* m, F&& f, Args&&... args);
    template <class F, class... Args>
    decltype(auto) iter_cells(const cs_mesh_t* m, F&& f, Args&&... args);
    template <class F, class... Args>
    decltype(auto) iter(cs_lnum_t n, F&& f, Args&&... args);
};

template <class Derived>
template <class F, class... Args>
decltype(auto) CsContextMixin<Derived>::iter_i_faces_(const cs_mesh_t* m, F* f, Args&&... args) {
  return static_cast<Derived*>(this)->iter_(m->n_i_faces, f, static_cast<Args&&>(args)...);
}
template <class Derived>
template <class F, class... Args>
decltype(auto) CsContextMixin<Derived>::iter_b_faces_(const cs_mesh_t* m, F* f, Args&&... args) {
  return static_cast<Derived*>(this)->iter_(m->n_b_faces, f, static_cast<Args&&>(args)...);
}
template <class Derived>
template <class F, class... Args>
decltype(auto) CsContextMixin<Derived>::iter_cells_(const cs_mesh_t* m, F* f, Args&&... args) {
  return static_cast<Derived*>(this)->iter_(m->n_cells, f, static_cast<Args&&>(args)...);
}
template <class Derived>
template <class F, class... Args>
decltype(auto) CsContextMixin<Derived>::iter_i_faces(const cs_mesh_t* m, F&& f, Args&&... args) {
  return static_cast<Derived*>(this)->iter_i_faces_(m, &f, static_cast<Args&&>(args)...);
}
template <class Derived>
template <class F, class... Args>
decltype(auto) CsContextMixin<Derived>::iter_b_faces(const cs_mesh_t* m, F&& f, Args&&... args) {
  return static_cast<Derived*>(this)->iter_b_faces_(m, &f, static_cast<Args&&>(args)...);
}
template <class Derived>
template <class F, class... Args>
decltype(auto) CsContextMixin<Derived>::iter_cells(const cs_mesh_t* m, F&& f, Args&&... args) {
  return static_cast<Derived*>(this)->iter_cells_(m, &f, static_cast<Args&&>(args)...);
}
template <class Derived>
template <class F, class... Args>
decltype(auto) CsContextMixin<Derived>::iter(cs_lnum_t n, F&& f, Args&&... args) {
  return static_cast<Derived*>(this)->iter_(n, &f, static_cast<Args&&>(args)...);
}

class CsOpenMpContext : public CsContextMixin<CsOpenMpContext> {
  public:
    template <class F, class... Args>
    auto iter_(cs_lnum_t n, F* f, Args&&... args) -> typename std::enable_if<F::is_host, bool>::type {
#pragma omp parallel for
      for (cs_lnum_t i = 0; i < n; ++i) {
        (*f)(i, static_cast<Args&&>(args)...);
      }
      return true;
    }
    bool iter_(cs_lnum_t n, void*, ...) {
      return false;
    }

    template <class F, class... Args>
    auto iter_i_faces_(const cs_mesh_t* m, F* f, Args&&... args) -> typename std::enable_if<F::is_host, bool>::type {
      const int n_i_groups                    = m->i_face_numbering->n_groups;
      const int n_i_threads                   = m->i_face_numbering->n_threads;
      const cs_lnum_t *restrict i_group_index = m->i_face_numbering->group_index;

      const cs_lnum_2_t *restrict i_face_cells
        = (const cs_lnum_2_t *restrict)m->i_face_cells;
      const cs_lnum_t *restrict b_face_cells
        = (const cs_lnum_t *restrict)m->b_face_cells;

      for (int g_id = 0; g_id < n_i_groups; g_id++) {

      #pragma omp parallel for
        for (int t_id = 0; t_id < n_i_threads; t_id++) {
          for (cs_lnum_t f_id = i_group_index[(t_id * n_i_groups + g_id) * 2];
                f_id < i_group_index[(t_id * n_i_groups + g_id) * 2 + 1];
                f_id++) {
            (*f)(f_id, static_cast<Args&&>(args)...);
          }
        }
      }

      return true;
    }
    bool iter_i_faces_(const cs_mesh_t*, void*, ...) {
      return false;
    }
};

#ifdef __NVCC__
template <class F, class... Args>
__global__ void cuda_kernel_iter_n(cs_lnum_t n, F f, Args... args) {
  // grid-stride loop
  for (cs_lnum_t id = blockIdx.x * blockDim.x + threadIdx.x; id < n;
       id += blockDim.x * gridDim.x) {
    f(id, args...);
  }
}
class CsCudaContext : public CsContextMixin<CsCudaContext> {
  private:
    long grid;
    long block;
    cudaStream_t stream;
    int device;
  public:
    CsCudaContext() : grid(0), block(0), stream(nullptr), device(-1) {}
    CsCudaContext(long grid, long block, cudaStream_t stream, int device) : grid(grid), block(block), stream(stream), device(device) {}
    CsCudaContext(long grid, long block, cudaStream_t stream) : grid(grid), block(block), stream(stream), device(0) {
      cudaGetDevice(&device);
    }
    CsCudaContext(long grid, long block) : grid(grid), block(block), stream(cudaStreamLegacy), device(0) {
      cudaGetDevice(&device);
    }

    void set_cuda_config(long grid, long block) {
      this->grid = grid;
      this->block = block;
    }
    void set_cuda_config(long grid, long block, cudaStream_t stream) {
      this->grid = grid;
      this->block = block;
      this->stream = stream;
    }
    void set_cuda_config(long grid, long block, cudaStream_t stream, int device) {
      this->grid = grid;
      this->block = block;
      this->stream = stream;
      this->device = device;
    }
  public:
    template <class F, class... Args>
    auto iter_(cs_lnum_t n, F* f, Args&&... args) -> typename std::enable_if<F::is_device, bool>::type {
      if (device < 0) {
        return false;
      }
      cuda_kernel_iter_n<<<grid, block, 0, stream>>>(n, std::move(*f), static_cast<Args&&>(args)...);
    }
    bool iter_(cs_lnum_t n, void*, ...) {
      return false;
    }
};
#endif

template <class... Contexts>
class CsCombinedContext : public CsContextMixin<CsCombinedContext<Contexts...>>, public Contexts... {
  private:
    using mixin_t = CsContextMixin<CsCombinedContext<Contexts...>>;
  public:
    CsCombinedContext() = default;
    CsCombinedContext(Contexts... contexts) : Contexts(std::move(contexts))... {}
  public:
    using mixin_t::iter_i_faces;
    using mixin_t::iter_b_faces;
    using mixin_t::iter_cells;
    using mixin_t::iter;

    template <class F, class... Args>
    void iter_i_faces_(const cs_mesh_t* m, F* f, Args&&... args) {
      bool executed = false;
      decltype(nullptr) try_execute[] = {
        (executed = executed || Contexts::iter_i_faces(m, f, args...), nullptr)...
      };
    }
    template <class F, class... Args>
    auto iter_b_faces_(const cs_mesh_t* m, F* f, Args&&... args) {
      bool executed = false;
      decltype(nullptr) try_execute[] = {
        (executed = executed || Contexts::iter_b_faces(m, f, args...), nullptr)...
      };
    }
    template <class F, class... Args>
    auto iter_cells_(const cs_mesh_t* m, F* f, Args&&... args) {
      bool executed = false;
      decltype(nullptr) try_execute[] = {
        (executed = executed || Contexts::iter_cells_(m, f, args...), nullptr)...
      };
    }
    template <class F, class... Args>
    auto iter_(cs_lnum_t n, F* f, Args&&... args) {
      bool executed = false;
      decltype(nullptr) try_execute[] = {
        (executed = executed || Contexts::iter_(n, f, args...), nullptr)...
      };
    }
};

class CsContext : public CsCombinedContext<
#ifdef __NVCC__
  CsCudaContext,
#endif
  CsOpenMpContext
> {
  private:
    using base_t = CsCombinedContext<
#ifdef __NVCC__
  CsCudaContext,
#endif
  CsOpenMpContext
>;
  public:
    using base_t::base_t;
    using base_t::operator=;
};

/*=============================================================================
 * Global variable definitions
 *============================================================================*/

/*=============================================================================
 * Public function prototypes
 *============================================================================*/



template <class F>
CS_CUDA_HOST HostFunctor<F> cs_host_functor(F f) noexcept(noexcept(F(std::move(f)))) {
  return HostFunctor<F>(std::move(f));
}

template <class F>
CS_CUDA_HOST_DEVICE DeviceFunctor<F> cs_device_functor(F f) noexcept(noexcept(F(std::move(f)))) {
  return DeviceFunctor<F>(std::move(f));
}

template <class F>
CS_CUDA_HOST_DEVICE HostDeviceFunctor<F> cs_host_device_functor(F f) noexcept(noexcept(F(std::move(f)))) {
  return HostDeviceFunctor<F>(std::move(f));
}



#if 0
/**
 * Other examples of dispatch
 */
template <class F, class... Args>
void
cpu_iter_face(const cs_mesh_t *m, F f, Args... args)
{
  const int n_i_groups                    = m->i_face_numbering->n_groups;
  const int n_i_threads                   = m->i_face_numbering->n_threads;
  const cs_lnum_t *restrict i_group_index = m->i_face_numbering->group_index;

  const cs_lnum_2_t *restrict i_face_cells
    = (const cs_lnum_2_t *restrict)m->i_face_cells;
  const cs_lnum_t *restrict b_face_cells
    = (const cs_lnum_t *restrict)m->b_face_cells;

  for (int g_id = 0; g_id < n_i_groups; g_id++) {

#pragma omp parallel for
    for (int t_id = 0; t_id < n_i_threads; t_id++) {
      for (cs_lnum_t f_id = i_group_index[(t_id * n_i_groups + g_id) * 2];
           f_id < i_group_index[(t_id * n_i_groups + g_id) * 2 + 1];
           f_id++) {
        f(f_id, args...);
      }
    }
  }
}

template <class F, class... Args>
void
cpu_iter_cell(const cs_mesh_t *m, F f, Args... args)
{
  const cs_lnum_t n_cells = m->n_cells;
#pragma omp parallel for
  for (cs_lnum_t c_id = 0; c_id < n_cells; c_id++) {
    f(c_id, args...);
  }
}

template <class F, class... Args>
__global__ void
gpu_iter_kernel(cs_lnum_t n, F f, Args... args)
{
  for (cs_lnum_t id = blockIdx.x * blockDim.x + threadIdx.x; id < n;
       id += blockDim.x * gridDim.x) {
    f(id, args...);
  }
}
template <class F, class... Args>
void
gpu_iter_face(const cs_mesh_t *m,
              int              block,
              int              grid,
              int              stream,
              F                f,
              Args... args)
{
  gpu_iter_kernel<<<block, grid> > >(m->n_i_faces, f, args...);
}
template <class F, class... Args>
void
gpu_iter_cell(const cs_mesh_t *m,
              int              block,
              int              grid,
              int              stream,
              F                f,
              Args... args)
{
  gpu_iter_kernel<<<block, grid, 0, stream> > >(m->n_cells, f, args...);
}

// // Example:
//
// cpu_iter_cell(m, [=] (cs_lnum_t c_id) {
//   // Do some stuff with c_id
// });
//
// gpu_iter_cell(m, bloc, grid, 0, [=] __device__ (cs_lnum_t c_id) {
//   // Do some stuff with c_id
// });

template <class... Args> class Kernel {
public:
  virtual void operator()(const cs_mesh_t *m, Args... args) = 0;
  virtual ~Kernel() {}
};

template <class Derived, class... Args> class CpuFaceKernel : Kernel<Args...> {
public:
  void
  operator()(const cs_mesh_t *m, Args... args) override
  {
    const int n_i_groups                    = m->i_face_numbering->n_groups;
    const int n_i_threads                   = m->i_face_numbering->n_threads;
    const cs_lnum_t *restrict i_group_index = m->i_face_numbering->group_index;

    const cs_lnum_2_t *restrict i_face_cells
      = (const cs_lnum_2_t *restrict)m->i_face_cells;
    const cs_lnum_t *restrict b_face_cells
      = (const cs_lnum_t *restrict)m->b_face_cells;

    for (int g_id = 0; g_id < n_i_groups; g_id++) {

#pragma omp parallel for
      for (int t_id = 0; t_id < n_i_threads; t_id++) {
        for (cs_lnum_t f_id = i_group_index[(t_id * n_i_groups + g_id) * 2];
             f_id < i_group_index[(t_id * n_i_groups + g_id) * 2 + 1];
             f_id++) {
          static_cast<Derived *>(this)->call(f_id, static_cast<Args>(args)...);
        }
      }
    }
  }
};

template <class Derived, class... Args> class CpuCellKernel : Kernel<Args...> {
public:
  void
  operator()(const cs_mesh_t *m, Args... args) override
  {
    const cs_lnum_t n_cells = m->n_cells;
#pragma omp parallel for
    for (cs_lnum_t c_id = 0; c_id < n_cells; c_id++) {
      static_cast<Derived *>(this)->call(c_id, static_cast<Args>(args)...);
    }
  }
};

template <class... Args> class GpuKernel : Kernel<Args...> {
protected:
  int block;
  int grid;
  int stream;

protected:
  template <class Derived>
  __global__ static void
  kernel(cs_lnum_t n, Args... args)
  {
    for (cs_lnum_t id = blockIdx.x * blockDim.x + threadIdx.x; id < n;
         id += blockDim.x * gridDim.x) {
      Derived::call(id, static_cast<Args>(args)...);
    }
  }

public:
  GpuKernel(int block, int grid, int stream)
    : block(block), grid(grid), stream(stream)
  {
  }
};

template <class Derived, class... Args>
class GpuFaceKernel : GpuKernel<Args...> {
public:
  using GpuKernel<Args...>::GpuKernel;

  void
  operator()(const cs_mesh_t *m, Args... args) override
  {
    kernel<Derived>
      <<<block, grid, 0, stream> > >(m.n_i_faces, static_cast<Args>(args)...);
  }
};

template <class Derived, class... Args>
class GpuCellKernel : GpuKernel<Args...> {
public:
  using GpuKernel<Args...>::GpuKernel;

  void
  operator()(const cs_mesh_t *m, Args... args) override
  {
    kernel<Derived>
      <<<block, grid, 0, stream> > >(m.n_cells, static_cast<Args>(args)...);
  }
};

// // Example:
//
// class MyCpuKernel : CpuFaceKernel<MyCpuKernel, ...> {
// public:
//   static void call(cs_lnum_t f_id, ...) {
//     // Do something with f_id
//   }
// }
//
// class MyGpuKernel : GpuFaceKernel<MyGpuKernel, ...> {
// public:
//   __device__ static void call(cs_lnum_t f_id, ...) {
//     // Do something with f_id
//   }
// }
//
// class MyKernel : CpuFaceKernel<MyKernel, ...>, GpuFaceKernel<MyKernel,
// ...> {
// public:
//   __host__ __device__ static void call(cs_lnum_t f_id, ...) {
//     // Do something with f_id
//   }
// }

#endif // commented out

#endif /* __cplusplus */

#endif /* __CS_DISPATCH_H__ */
