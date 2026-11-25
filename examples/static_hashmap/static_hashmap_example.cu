#include "raylib.h"

#include <vector>

#ifdef _WIN32
// Windows defines CloseWindow and ShowCursor, which conflict with raylib's API.
#define CloseWindow Win32CloseWindow
#define ShowCursor Win32ShowCursor
#include <windows.h>
#undef ShowCursor
#undef CloseWindow
#endif

#include "static_hashmap.cuh"
#include "static_cg_hashmap.cuh"

namespace
{
constexpr int kGridSize = 50;
constexpr int kWindowSize = 800;
constexpr int kCellSize = (kWindowSize + kGridSize - 1) / kGridSize;
constexpr float kCircleDiameter = 20.0f;
constexpr float kCursorRadius = 4.0f;

using PositionHashMap = GroupStaticHash<MurmurHash64A, GroupLinearProbing<4>, std::uint64_t>;
using DevicePositionHash = PositionHashMap::DeviceStaticHash;

__host__ __device__ std::uint64_t position_to_key(int x, int y)
{
  return static_cast<std::uint64_t>(x * kGridSize + y);
}

__host__ __device__ std::tuple<int, int> position_from_key(int key)
{
  return  {key / kGridSize, key % kGridSize};
}

template <unsigned groupSize>
__global__ void lookup_positions_in_radius_kernel(DevicePositionHash table,
                                                  float mouse_grid_x,
                                                  float mouse_grid_y,
                                                  std::uint8_t* found)
{
  const int index = get_tid() / groupSize;
  constexpr int cell_count = kGridSize * kGridSize;
  if (index >= cell_count)
    return;

  auto [x, y] = position_from_key(index);
  const float dx = x + 0.5f - mouse_grid_x;
  const float dy = y + 0.5f - mouse_grid_y;
  const float radius_squared = kCursorRadius * kCursorRadius;
  found[index] = dx * dx + dy * dy <= radius_squared && table.has_key(index) ? 1 : 0;
}
} // namespace

int main()
{
  std::vector<std::uint64_t> circle_keys;
  const float center = kGridSize / 2.0f;
  const float radius = kCircleDiameter / 2.0f;
  const float radius_squared = radius * radius;

  for (int x = 0; x < kGridSize; ++x)
  {
    for (int y = 0; y < kGridSize; ++y)
    {
      // Grid coordinates refer to cell centers at x + 0.5, y + 0.5.
      const float dx = x + 0.5f - center;
      const float dy = y + 0.5f - center;
      if (dx * dx + dy * dy <= radius_squared)
      {
        circle_keys.push_back(position_to_key(x, y));
      }
    }
  }

  std::vector<std::uint64_t> values(circle_keys.size(), 1);
  auto device = cuda::device::current::get();
  auto stream = device.default_stream();
  auto device_keys = to_mem(circle_keys);
  auto device_values = to_mem(values);

  PositionHashMap positions(stream, circle_keys.size(), 0.7f);
  positions.insert(device_keys, device_values);
  stream.synchronize();

  constexpr int cell_count = kGridSize * kGridSize;
  auto found_positions = cuda::memory::make_unique_span<std::uint8_t>(device, cell_count);
  auto lookup_config =
      cuda::launch_config_builder().overall_dimensions(cell_count * PositionHashMap::kGroupSize).block_size(256).build();
  std::vector<std::uint8_t> found(cell_count);

  SetConfigFlags(FLAG_VSYNC_HINT | FLAG_MSAA_4X_HINT | FLAG_WINDOW_HIGHDPI);
  InitWindow(kWindowSize, kWindowSize, "static hashmap - hover the circle to query positions");
  SetTargetFPS(60);

  while (!WindowShouldClose())
  {
    const Vector2 mouse = GetMousePosition();
    cuda::launch(lookup_positions_in_radius_kernel<PositionHashMap::kGroupSize>,
                 lookup_config,
                 positions.device_table(),
                 mouse.x / kCellSize,
                 mouse.y / kCellSize,
                 found_positions.data());
    stream.synchronize();
    cuda::memory::copy(found, found_positions);

    BeginDrawing();
    ClearBackground(RAYWHITE);

    for (const auto key : circle_keys)
    {
      auto [x, y] = position_from_key(key);
      DrawRectangle(x * kCellSize,
                    y * kCellSize,
                    kCellSize,
                    kCellSize,
                    GREEN);
    }

    for (int index = 0; index < cell_count; ++index)
    {
      if (found[index] != 0)
      {
        auto [x, y] = position_from_key(index);
        DrawRectangle(x * kCellSize,
                      y * kCellSize,
                      kCellSize,
                      kCellSize,
                      RED);
      }
    }

    DrawCircleLines(mouse.x,
                    mouse.y,
                    kCursorRadius * kCellSize,
                    DARKGRAY);

    for (int line = 0; line <= kGridSize; ++line)
    {
      const int offset = line * kCellSize;
      DrawLine(offset, 0, offset, kWindowSize, LIGHTGRAY);
      DrawLine(0, offset, kWindowSize, offset, LIGHTGRAY);
    }

    EndDrawing();
  }

  CloseWindow();
  return 0;
}
