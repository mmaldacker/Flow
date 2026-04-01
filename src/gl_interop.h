#pragma once

#include "driver_types.h"
#include "surface_types.h"

namespace flow
{
class OpenglTexture;

class OpenglTextureView
{
public:
  ~OpenglTextureView();

  cudaSurfaceObject_t view() const;
  cudaArray_t array() const;

  friend class OpenglTexture;
private:
  OpenglTextureView(cudaGraphicsResource* resource);

  cudaGraphicsResource* _resource;
  cudaSurfaceObject_t _surface_object;
  cudaArray_t _array;
};

class OpenglTexture
{
public:
  OpenglTexture(unsigned int id);
  OpenglTextureView map() const;

private:
  cudaGraphicsResource* _resource;
};

}