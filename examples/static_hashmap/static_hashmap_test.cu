#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <vector>

#include "static_cg_hashmap.cuh"
#include "static_hashmap.cuh"

template <typename StaticHash, unsigned groupSize>
__global__ void insert_test(tcb::span<std::uint64_t> search_keys,
                            tcb::span<std::uint8_t> found_keys,
                            typename StaticHash::DeviceStaticHash table)
{
  const auto gid = get_tid() / groupSize;

  if (gid >= search_keys.size())
    return;

  auto key = search_keys[gid];
  auto* value = table.get_value(key);
  if (value != nullptr)
  {
    found_keys[gid] = true;
  }
}

template <typename StaticHash, unsigned groupSize>
void verify_hashmap()
{
  constexpr int table_size = 16;
  auto device = cuda::device::current::get();
  auto stream = device.default_stream();
  StaticHash hash(stream, table_size, 1.0);

  std::vector<std::uint64_t> keys = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  std::vector<std::uint64_t> values(10);

  auto device_keys = to_mem(keys);
  auto device_values = to_mem(values);

  hash.insert(device_keys, device_values);
  stream.synchronize();

  std::vector<std::uint64_t> search_keys = {0, 2, 4, 12, 43};
  std::vector<std::uint8_t> found_keys(search_keys.size(), 0);

  auto device_search_keys = to_mem(search_keys);
  auto device_found_keys = to_mem(found_keys);

  auto launch_config = cuda::launch_config_builder()
                           .overall_dimensions(device_search_keys.size() * groupSize)
                           .block_size(256)
                           .build();
  stream.enqueue.kernel_launch(insert_test<StaticHash, groupSize>,
                               launch_config,
                               to_span(device_search_keys),
                               to_span(device_found_keys),
                               hash.device_table());
  stream.synchronize();

  found_keys = from_mem(device_found_keys);

  assert(!found_keys[0]);
  assert(found_keys[1]);
  assert(found_keys[2]);
  assert(!found_keys[3]);
  assert(!found_keys[4]);
}

void verify_group_hashmap()
{
  constexpr int table_size = 1'000'000;
  auto device = cuda::device::current::get();
  auto stream = device.default_stream();

  StaticHash<MurmurHash64A, LinearProbing, std::uint64_t> hash1(stream, table_size, 0.8);
  GroupStaticHash<MurmurHash64A, GroupLinearProbing<1>, 1, std::uint64_t> hash2(
      stream, table_size, 0.8);
  GroupStaticHash<MurmurHash64A, GroupLinearProbing<4>, 4, std::uint64_t> hash3(
      stream, table_size, 0.8);

  std::vector<std::uint64_t> keys(table_size);
  std::vector<std::uint64_t> values(table_size, 100);

  std::generate(keys.begin(), keys.end(), std::rand);

  auto device_keys = to_mem(keys);
  auto device_values = to_mem(values);

  hash1.insert(device_keys, device_values);
  hash2.insert(device_keys, device_values);
  hash3.insert(device_keys, device_values);
  stream.synchronize();

  auto table1 = hash1.local_table();
  auto table2 = hash2.local_table();
  auto table3 = hash3.local_table();

  auto count1 = std::count_if(
      table1.begin(), table1.end(), [](const std::uint64_t& key) { return key != ~0; });
  auto count2 = std::count_if(
      table2.begin(), table2.end(), [](const std::uint64_t& key) { return key != ~0; });
  auto count3 = std::count_if(
      table3.begin(), table3.end(), [](const std::uint64_t& key) { return key != ~0; });

  assert(count1 != 0);
  assert(count1 == count2);
  assert(count1 == count3);
}

int main()
{
  try
  {
    std::cout << "Verify static hash linear" << std::endl;
    verify_hashmap<StaticHash<MurmurHash64A, LinearProbing, std::uint64_t>, 1>();

    std::cout << "Verify static hash quadratic" << std::endl;
    //verify_hashmap<StaticHash<MurmurHash64A, QuadraticProbing, std::uint64_t>, 1>();

    std::cout << "Verify group static hash size 1" << std::endl;
    verify_hashmap<GroupStaticHash<MurmurHash64A, GroupLinearProbing<1>, 1, std::uint64_t>, 1>();

    std::cout << "Verify group static hash size 2" << std::endl;
    verify_hashmap<GroupStaticHash<MurmurHash64A, GroupLinearProbing<2>, 2, std::uint64_t>, 2>();

    std::cout << "Verify group static hash size 4" << std::endl;
    verify_hashmap<GroupStaticHash<MurmurHash64A, GroupLinearProbing<4>, 4, std::uint64_t>, 4>();

    std::cout << "Verify group static hash vs linear" << std::endl;

    verify_group_hashmap();
  }
  catch (const std::exception& e)
  {
    std::cout << "Exception: " << e.what() << std::endl;
    return 0;
  }

  return 0;
}
