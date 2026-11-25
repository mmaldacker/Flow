add_executable(static_hashmap_example static_hashmap_test.cu)
target_compile_options(static_hashmap_example PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_FLAGS}>
)
target_link_libraries(static_hashmap_example flow)

add_executable(static_hashmap_performance static_hashmap_performance.cu)
target_compile_options(static_hashmap_performance PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_FLAGS}>
)
target_link_libraries(static_hashmap_performance flow)