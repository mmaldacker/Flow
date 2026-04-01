# Flow
CUDA based fluid simulator

## Building

The project is cmake based and will pull the required dependencies. The only requirement is to have the CUDA runtime installed.
Building is therefore the standard cmake commands:

```bash
cmake -DCMAKE_BUILD_TYPE=Release -S . -B build
cmake --build build
```

## Examples
### CUDA / OpenGL Interop

Simple example on how to write an OpenGL texture from a CUDA Kernel, using the Game of Life as example.