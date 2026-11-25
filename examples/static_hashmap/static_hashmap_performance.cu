#include <iostream>

#include "static_cg_hashmap.cuh"
#include "static_hashmap.cuh"

template <typename StaticHash>
__host__ double test_insert_hashmap(float load)
{
  auto device = cuda::device::current::get();
  auto stream = device.default_stream();

  constexpr int table_size = 10'000'000;
  StaticHash hash(stream, table_size, load);

  std::vector<std::uint64_t> keys(table_size);
  std::vector<std::uint64_t> values(table_size, 100);

  std::generate(keys.begin(), keys.end(), std::rand);

  auto device_keys = to_mem(keys);
  auto device_values = to_mem(values);

  auto start = cuda::event::create(device);
  auto end = cuda::event::create(device);

  int total = 5;
  double time = 0.0;
  for (int i = 0; i < total; i++)
  {
    stream.enqueue.event(start);
    hash.insert(device_keys, device_values);
    stream.enqueue.event(end);
    stream.synchronize();
    time += cuda::event::time_elapsed_between(start, end).count();
    hash.clear();
  }

  return time / total;
}


int main()
{
   std::vector<float> loads;
  float load = 0.5;
  while (load < 0.95)
  {
    loads.push_back(load);
    load += 0.05;
  }

  std::cout << "name,load,time" << std::endl;
  for (auto load : loads)
  {
    auto elapsed =
        test_insert_hashmap<StaticHash<MurmurHash64A, LinearProbing, std::uint64_t>>(load);
    std::cout << "linear," << load << "," << elapsed << std::endl;
  }

  for (auto load : loads)
  {
    auto elapsed = test_insert_hashmap<StaticHash<MurmurHash64A, QuadraticProbing, std::uint64_t>>(
        load);
    std::cout << "quadratic," << load << "," << elapsed << std::endl;
  }

  for (auto load : loads)
  {
    auto elapsed =
        test_insert_hashmap<GroupStaticHash<MurmurHash64A, GroupLinearProbing<1>, 1, std::uint64_t>>(
            load);
    std::cout << "grouplinear1," << load << "," << elapsed << std::endl;
  }

  for (auto load : loads)
  {
    auto elapsed =
        test_insert_hashmap<GroupStaticHash<MurmurHash64A, GroupLinearProbing<2>, 2, std::uint64_t>>(
            load);
    std::cout << "grouplinear2," << load << "," << elapsed << std::endl;
  }

  for (auto load : loads)
  {
    auto elapsed =
        test_insert_hashmap<GroupStaticHash<MurmurHash64A, GroupLinearProbing<4>, 4, std::uint64_t>>(
            load);
    std::cout << "grouplinear4," << load << "," << elapsed << std::endl;
  }

  for (auto load : loads)
  {
    auto elapsed =
        test_insert_hashmap<GroupStaticHash<MurmurHash64A, GroupLinearProbing<8>, 8, std::uint64_t>>(
            load);
    std::cout << "grouplinear8," << load << "," << elapsed << std::endl;
  }

  for (auto load : loads)
  {
    auto elapsed =
        test_insert_hashmap<GroupStaticHash<MurmurHash64A, GroupLinearProbing<16>, 16, std::uint64_t>>(
            load);
    std::cout << "grouplinear16," << load << "," << elapsed << std::endl;
  }

  for (auto load : loads)
  {
    auto elapsed =
        test_insert_hashmap<GroupStaticHash<MurmurHash64A, GroupLinearProbing<32>, 32, std::uint64_t>>(
            load);
    std::cout << "grouplinear32," << load << "," << elapsed << std::endl;
  }
}