#pragma once

#include "cuda/api.hpp"
#include "span.hpp"

__device__ unsigned get_tid()
{
  return threadIdx.x + blockIdx.x * blockDim.x;
}

struct MurmurHash64A
{
  static constexpr std::uint64_t m = 0xc6a4a7935bd1e995LLU;
  static constexpr std::uint64_t r = 47;
  static constexpr std::uint64_t seed = 0x811c9dc1L;

  __device__ static std::uint64_t hash(std::uint64_t k)
  {
    uint64_t h = seed ^ m;

    k *= m;
    k ^= k >> r;
    k *= m;

    h ^= k;
    h *= m;

    return h;
  }
};

template<typename T>
tcb::span<T> to_span(const cuda::unique_span<T>& m)
{
  return {m.data(), m.size()};
}

template<typename T>
std::vector<T> from_mem(const cuda::unique_span<T>& input)
{
  std::vector<T> output(input.size());
  cuda::memory::copy(output, input);
  return output;
}

template<typename T>
cuda::unique_span<T> to_mem(const std::vector<T>& input)
{
  auto  output = cuda::memory::make_unique_span<T>(cuda::device::current::get(), input.size());
  cuda::memory::copy(output, input);
  return output;
}