#include <random>
#include "cuda/api.hpp"
#include "gl_interop.h"
#include "raylib.h"
#include "surface_indirect_functions.h"

__global__ void game_of_life_step(std::uint8_t* prev,
                                  std::uint8_t* next,
                                  cudaSurfaceObject_t surface,
                                  int width,
                                  int height)
{
  const uchar4 lightgray  = make_uchar4(200, 200, 200, 255);
  const uchar4 darkgray = make_uchar4(80, 80, 80, 255);

  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height)
  {
    int left = (x - 1 + width) % width;
    int right = (x + 1) % width;
    int top = (y - 1 + height) % height;
    int bottom = (y + 1) % height;

    auto i = [=](int x, int y) { return x + y * width; };

    int total = prev[i(left, y)] + prev[i(right, y)] + prev[i(x, top)] + prev[i(x, bottom)] +
                prev[i(left, top)] + prev[i(right, top)] + prev[i(left, bottom)] +
                prev[i(right, bottom)];

    next[i(x, y)] = total == 3 || (total == 2 && prev[i(x, y)]) ? 1 : 0;

    auto colour = next[i(x, y)] ? lightgray : darkgray;
    surf2Dwrite(colour, surface, x * 4, y);
  }
}

std::vector<std::uint8_t> random_vector(std::size_t s)
{
  std::random_device rnd_device;
  std::mt19937 mersenne_engine{rnd_device()};
  std::uniform_int_distribution<int> dist{0, 3}; // 25% is alive
  auto gen = [&]() { return dist(mersenne_engine) > 0 ? 1 : 0; };

  std::vector<std::uint8_t> vec(s);
  std::generate(vec.begin(), vec.end(), gen);

  return vec;
}

int main()
{
  int window_width = 800;
  int window_height = 800;
 
  SetConfigFlags(FLAG_VSYNC_HINT | FLAG_MSAA_4X_HINT);
  InitWindow(window_width, window_height, "game of life - cuda opengl interop");
  SetTargetFPS(60);

  float scale = 2.0f;
  int width = window_width / scale;
  int height = window_height / scale;

  auto texture = LoadTextureFromImage({nullptr, width, height, 1, PIXELFORMAT_UNCOMPRESSED_R8G8B8A8});
  SetTextureFilter(texture , TEXTURE_FILTER_POINT);
  flow::OpenglTexture glTexture(texture.id);

  auto device = cuda::device::current::get();
  auto num_elements = width * height;

  auto a = cuda::memory::make_unique_span<std::uint8_t>(device, num_elements);
  auto b = cuda::memory::make_unique_span<std::uint8_t>(device, num_elements);

  auto r = random_vector(num_elements);
  cuda::memory::copy(a, r);
  cuda::memory::copy(b, r);

  auto* curr = a.data();
  auto* next = b.data();

  auto launch_config_2d =
      cuda::launch_config_builder().overall_dimensions(width, height, 1).block_size(256).build();

  while (!WindowShouldClose())
  {
    {
      auto view = glTexture.map();
      cuda::launch(game_of_life_step, launch_config_2d, curr, next, view.view(), width, height);
      std::swap(curr, next);
    }

    BeginDrawing();
    DrawTextureEx(texture, {0, 0}, 0.0f, scale, WHITE);
    EndDrawing();
  }

  CloseWindow();

  UnloadTexture(texture);

  return 0;
}