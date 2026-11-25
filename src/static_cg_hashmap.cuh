#pragma once

#include <cooperative_groups.h>
#include "common.cuh"

namespace cg = cooperative_groups;

template <unsigned groupSize = 1>
class GroupLinearProbing
{
public:
  static constexpr unsigned kGroupSize = groupSize;

  __device__ GroupLinearProbing(std::uint64_t keyIndex,
                                const cg::thread_block_tile<groupSize>& group,
                                std::uint64_t tableSize)
      : probingIndex_(0), groupSize_(group.size()), tableSize_(tableSize)
  {
    index_ = (keyIndex + group.thread_rank()) % tableSize_;
  }

  __device__ std::uint64_t index() const { return index_; }

  __device__ void next()
  {
    index_ = (index_ + groupSize_) % tableSize_;
    probingIndex_ += groupSize_;
  }

  __device__ bool isValid() const { return probingIndex_ < tableSize_; }

private:
  std::uint64_t index_;
  std::uint64_t probingIndex_;
  std::uint64_t groupSize_;
  std::uint64_t tableSize_;
};

__global__ void clear_group_kernel(tcb::span<std::uint64_t> table)
{
  const auto index = get_tid();
  if (index < table.size())
  {
    table[index] = ~0;
  }
};

template <typename DeviceStaticHash, typename ValueType, unsigned groupSize>
__global__ void insert_group_kernel(tcb::span<std::uint64_t> keys,
                                    tcb::span<ValueType> values,
                                    DeviceStaticHash table)
{
  auto index = get_tid() / groupSize;
  if (index >= keys.size())
    return;

  auto key = keys[index];
  auto& value = values[index];

  table.insert(key, value);
};

template <typename Hash, typename Probing, typename ValueType>
class GroupDeviceStaticHash;

template <typename Hash, typename Probing, typename ValueType>
class GroupStaticHash
{
public:
  static const std::uint64_t kEmpty = ~0;
  static constexpr unsigned kGroupSize = Probing::kGroupSize;

  using DeviceStaticHash = GroupDeviceStaticHash<Hash, Probing, ValueType>;

  GroupStaticHash(const cuda::stream_t& stream, std::uint64_t size, float load_factor)
      : stream_(stream)
      , size_(std::ceil(size / load_factor))
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
    stream_.enqueue.kernel_launch( clear_group_kernel, launch_config, to_span(table_));
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
                             .overall_dimensions(device_keys.size() * kGroupSize)
                             .block_size(256)
                             .build();
    stream_.enqueue.kernel_launch(insert_group_kernel<DeviceStaticHash, ValueType, kGroupSize>,
                 launch_config,
                 keys,
                 values,
                 device_hash);
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
class GroupDeviceStaticHash
{
public:
  static constexpr unsigned kGroupSize = Probing::kGroupSize;

  GroupDeviceStaticHash(cuda::unique_span<std::uint64_t>& table,
                        cuda::unique_span<ValueType>& table_values)
      : table_(to_span(table)), table_values_(to_span(table_values))
  {
  }

  using StaticHash = GroupStaticHash<Hash, Probing, ValueType>;

  __device__ void insert(std::uint64_t key, ValueType& value)
  {
    const cg::thread_block_tile<kGroupSize> group =
        cg::tiled_partition<kGroupSize>(cg::this_thread_block());

    Probing b(Hash::hash(key), group, table_.size());
    for (; b.isValid(); b.next())
    {
      auto i = b.index();
      auto& k = table_[i];

      // returns a mask of the result of k == kEmpty for each thread in the group
      auto empty_mask = group.ballot(k == StaticHash::kEmpty);
      while (empty_mask)
      {
        // ffs: find the position of the least significant bit set to 1
        const auto leader = __ffs(empty_mask) - 1;

        // leader tries to insert
        bool success = false;
        if (group.thread_rank() == leader)
        {
          auto old_key = atomicCAS(&k, StaticHash::kEmpty, key);
          success = (old_key == StaticHash::kEmpty);
        }

        // returns if success
        if (group.any(success))
        {
          if (group.thread_rank() == 0)
          {
            auto value_index = group.shfl(i, leader);
            table_values_[value_index] = value;
          }
          return;
        }

        // remove thread from mask and try again
        empty_mask ^= 1UL << leader;
      }
    }
  }

  __device__ bool has_key(std::uint64_t key)
  {
    const cg::thread_block_tile<kGroupSize> group =
        cg::tiled_partition<kGroupSize>(cg::this_thread_block());

    Probing b(Hash::hash(key), group, table_.size());
    for (; b.isValid(); b.next())
    {
      auto& k = table_[b.index()];

      // returns a mask of the result of k == kEmpty for each thread in the group
      if (group.ballot(k == key))
      {
        return true;
      }
    }

    return false;
  }

  __device__ ValueType* get_value(std::uint64_t key)
  {
    const cg::thread_block_tile<kGroupSize> group =
        cg::tiled_partition<kGroupSize>(cg::this_thread_block());

    Probing b(Hash::hash(key), group, table_.size());
    for (; b.isValid(); b.next())
    {
      auto& k = table_[b.index()];

      // returns a mask of the result of k == kEmpty for each thread in the group
      if (auto key_mask = group.ballot(k == key))
      {
        if (group.thread_rank() == 0)
        {
          // ffs: find the position of the least significant bit set to 1
          const auto leader = __ffs(key_mask) - 1;
          auto value_index = group.shfl(b.index(), leader);
          return &table_values_[value_index];
        }

        return nullptr;
      }

      if (group.any(k == StaticHash::kEmpty))
      {
        return nullptr;
      }
    }

    return nullptr;
  }

private:
  tcb::span<std::uint64_t> table_;
  tcb::span<ValueType> table_values_;
};
