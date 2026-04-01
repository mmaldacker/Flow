#ifdef WIN32
#define NOMINMAX
#include <windows.h>
#endif

#include "gl_interop.h"

#include <cuda_gl_interop.h>
#include "cuda/api.hpp"

namespace flow
{
OpenglTexture::OpenglTexture(unsigned int id)
{
  auto status =
      cudaGraphicsGLRegisterImage(&_resource, id, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsNone);
  throw_if_error_lazy(status, "failed allocating 2D CUDA array");
}

OpenglTextureView OpenglTexture::map() const
{
  return OpenglTextureView(_resource);
}

OpenglTextureView::OpenglTextureView(cudaGraphicsResource* resource) : _resource(resource)
{
  auto status = cudaGraphicsMapResources(1, &_resource, 0);
  throw_if_error_lazy(status, "failed to map resources");

  status = cudaGraphicsSubResourceGetMappedArray(&_array, _resource, 0, 0);
  throw_if_error_lazy(status, "failed to map array");

  cudaResourceDesc texRes{};
  texRes.resType = cudaResourceTypeArray;
  texRes.res.array.array = _array;

  status = cudaCreateSurfaceObject(&_surface_object, &texRes);
  throw_if_error_lazy(status, "failed to create surface object");
}

OpenglTextureView::~OpenglTextureView()
{
  cudaDestroySurfaceObject(_surface_object);
  cudaGraphicsUnmapResources(1, &_resource, 0);
}

cudaSurfaceObject_t OpenglTextureView::view() const
{
  return _surface_object;
}

cudaArray_t OpenglTextureView::array() const
{
  return _array;
}

}  // namespace flow