# CMakeLists.txt pour les tests d'opérations C

cmake_minimum_required(VERSION 3.16)

# Trouver GGML (dans le sous-module llama.cpp)
set(LLAMA_GGML_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../llama.cpp/ggml)

# --- test_b1_linear ---
add_executable(test_b1_linear test_b1_linear.c)
target_link_libraries(test_b1_linear PRIVATE ggml ggml-cpu)
target_include_directories(test_b1_linear PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../llama.cpp/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../llama.cpp/ggml/include
)

# TODO: Add more test executables here as we implement them
# add_executable(test_rope_2d test_rope_2d.c)
# target_link_libraries(test_rope_2d PRIVATE ggml ggml-cpu)
# ...
