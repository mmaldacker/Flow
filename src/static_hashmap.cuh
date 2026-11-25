#pragma once

#include "common.cuh"
#include "cuda/api.hpp"

class LinearProbing
{
public:
  __device__ LinearProbing(std::uint64_t index, std::uint64_t tableSize)
      : index_(index % tableSize), probingIndex_(0), tableSize_(tableSize)
  {
  }

  __device__ std::uint64_t index() const { return (index_ + probingIndex_) % tableSize_; }

  __device__ void next() { probingIndex_++; }

  __device__ bool isValid() const { return probingIndex_ < tableSize_; }

private:
  std::uint64_t index_;
  std::uint64_t probingIndex_;
  std::uint64_t tableSize_;
};

class QuadraticProbing
{
public:
  __device__ QuadraticProbing(std::uint64_t index, std::uint64_t tableSize)
      : index_(index % tableSize), probingIndex_(0), tableSize_(tableSize)
  {
  }

  __device__ std::uint64_t index() const
  {
    return (index_ + probingIndex_ * (probingIndex_ + 1) / 2) % tableSize_;
  }

  __device__ void next() { probingIndex_++; }

  __device__ bool isValid() const { return probingIndex_ < tableSize_; }

private:
  std::uint64_t index_;
  std::uint64_t probingIndex_;
  std::uint64_t tableSize_;
};

__global__ void clear_kernel(tcb::span<std::uint64_t> table)
{
  const auto index = get_tid();
  if (index < table.size())
  {
    table[index] = ~0;
  }
};

template <typename DeviceStaticHash, typename ValueType>
__global__ void insert_kernel(tcb::span<std::uint64_t> keys,
                              tcb::span<ValueType> values,
                              DeviceStaticHash table)
{
  auto index = get_tid();
  if (index >= keys.size())
    return;

  auto key = keys[index];
  auto& value = values[index];

  table.insert(key, value);
};

template < typename Hash, typename Probing,typename ValueType>
class DeviceStaticHash;

template <typename Hash, typename Probing, typename ValueType>
class StaticHash
{
public:
  static const std::uint64_t kEmpty = ~0;

  using DeviceStaticHash = DeviceStaticHash<Hash, Probing, ValueType>;

  StaticHash(const cuda::stream_t& stream, std::uint64_t size, float load_factor)
      : stream_(stream)
      , size_(size / load_factor)
      , table_(cuda::memory::make_unique_span<std::uint64_t>(cuda::device::current::get(), size_))
      , table_values_(
            cuda::memory::make_unique_span<std::uint64_t>(cuda::device::current::get(), size_))
  {
    clear();
  }

  void clear()
  {
    auto launch_config =
        cuda::launch_config_builder().overall_dimensions(size_).block_size(256).build();
    stream_.enqueue.kernel_launch(clear_kernel, launch_config, to_span(table_));
  }

  [[nodiscard]] std::uint64_t size() const { return size_; }

  void insert(cuda::unique_span<std::uint64_t>& device_keys,
              cuda::unique_span<ValueType>& device_values)
  {
    assert(device_keys.size() < table_.size());
    assert(device_keys.size() == device_values.size());

    auto keys = to_span(device_keys);
    auto values = to_span(device_values);

    auto device_hash = device_table();

    auto launch_config = cuda::launch_config_builder()
                             .overall_dimensions(device_keys.size())
                             .block_size(256)
                             .build();
    stream_.enqueue.kernel_launch(
        insert_kernel<DeviceStaticHash, ValueType>, launch_config, keys, values, device_hash);
  }

  [[nodiscard]] std::vector<std::uint64_t> local_table() const { return from_mem(table_); }

  [[nodiscard]] std::vector<ValueType> local_table_values() const
  {
    return from_mem(table_values_);
  }

  DeviceStaticHash device_table() { return DeviceStaticHash{table_, table_values_}; }

private:
  const cuda::stream_t& stream_;
  std::uint64_t size_;
  cuda::unique_span<std::uint64_t> table_;
  cuda::unique_span<ValueType> table_values_;
};

template <typename Hash, typename Probing, typename ValueType>
class DeviceStaticHash
{
public:
  DeviceStaticHash(cuda::unique_span<std::uint64_t>& table,
                   cuda::unique_span<ValueType>& table_values)
      : table_(to_span(table)), table_values_(to_span(table_values))
  {
  }

  using StaticHash = StaticHash<Hash, Probing, ValueType>;

  __device__ void insert(std::uint64_t key, ValueType& value)
  {
    std::uint64_t key_index = Hash::hash(key);
    Probing b(key_index, table_.size());
    for (; b.isValid(); b.next())
    {
      auto i = b.index();
      auto& k = table_[i];

      if (k != StaticHash::kEmpty)
      {
        continue;
      }

      auto old_key = atomicCAS(&k, StaticHash::kEmpty, key);
      if (old_key == StaticHash::kEmpty)
      {
        table_values_[i] = value;
        return;
      }
    }
  }

  __device__ bool has_key(std::uint64_t key)
  {
    Probing b(Hash::hash(key), table_.size());
    for (; b.isValid(); b.next())
    {
      auto& k = table_[b.index()];
      if (k == key)
      {
        return true;
      }
    }

    return false;
  }

  __device__ ValueType* get_value(std::uint64_t key)
  {
    Probing b(Hash::hash(key), table_.size());
    for (; b.isValid(); b.next())
    {
      auto i = b.index();
      auto& k = table_[i];
      if (k == key)
      {
        return &table_values_[i];
      }
    }

    return nullptr;
  }

private:
  tcb::span<std::uint64_t> table_;
  tcb::span<ValueType> table_values_;
};
