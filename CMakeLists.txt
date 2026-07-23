cmake_minimum_required(VERSION 3.16)
project(bonsai.cpp VERSION 0.1.0 LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Auto-detect and enable AVX2 and FMA flags if the compiler supports them
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-mavx2" COMPILER_SUPPORTS_AVX2)
check_cxx_compiler_flag("-mfma" COMPILER_SUPPORTS_FMA)

if(COMPILER_SUPPORTS_AVX2)
    message(STATUS "Compiler supports AVX2. Enabling -mavx2.")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mavx2")
endif()

if(COMPILER_SUPPORTS_FMA)
    message(STATUS "Compiler supports FMA. Enabling -mfma.")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mfma")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mfma")
endif()

# CUDA support disabled

# Disable unnecessary llama.cpp features to keep build minimal
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "Build llama.cpp examples" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "Build llama.cpp server" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "Build llama.cpp tests" FORCE)
set(LLAMA_BUILD_COMMON ON CACHE BOOL "Build llama.cpp common utils" FORCE)

# Add llama.cpp as a subdirectory
add_subdirectory(llama.cpp)

# Create the main encoder executable
add_executable(bonsai_encoder src/encoder.cpp)

# Link against llama and ggml
target_link_libraries(bonsai_encoder PRIVATE llama ggml)

# Include directories
target_include_directories(bonsai_encoder PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp/include)

# Install target
install(TARGETS bonsai_encoder RUNTIME DESTINATION bin)

# --- bonsai_vae: Flux VAE decoder, ggml-only (Step 3 of the pipeline) ---
add_executable(bonsai_vae
    src/vae_decoder.cpp
    src/vae_graph.cpp
)

target_link_libraries(bonsai_vae PRIVATE ggml ggml-cpu)
target_include_directories(bonsai_vae PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp/include
    ${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp/ggml/include
)

install(TARGETS bonsai_vae RUNTIME DESTINATION bin)

# --- bonsai_diffuser: B1_0 quantized Flux diffusion engine (Step 2 of the pipeline) ---
add_executable(bonsai_diffuser
    src/diffuser.cpp
    src/diffuser_graph.cpp
)

target_link_libraries(bonsai_diffuser PRIVATE ggml ggml-cpu)
target_include_directories(bonsai_diffuser PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp/include
    ${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp/ggml/include
)

install(TARGETS bonsai_diffuser RUNTIME DESTINATION bin)

# --- Tests unitaires des opérations diffuser ---
# Test b1_linear: multiplication matricielle quantifiée B1_0
add_executable(test_b1_linear
    tests/c_ops/test_b1_linear.cpp
)

target_link_libraries(test_b1_linear PRIVATE ggml ggml-cpu)
target_include_directories(test_b1_linear PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp/include
    ${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp/ggml/include
)

install(TARGETS test_b1_linear RUNTIME DESTINATION bin)

# Test RoPE 2D: position embedding rotatif 2D
add_executable(test_rope_2d
    tests/c_ops/test_rope_2d.cpp
)

target_link_libraries(test_rope_2d PRIVATE ggml ggml-cpu)
target_include_directories(test_rope_2d PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp/include
    ${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp/ggml/include
)

install(TARGETS test_rope_2d RUNTIME DESTINATION bin)

# Test input projection (x_embedder / context_embedder)
add_executable(test_input_proj
    tests/c_ops/test_input_proj.cpp
)

target_link_libraries(test_input_proj PRIVATE ggml ggml-cpu)
target_include_directories(test_input_proj PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp/include
    ${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp/ggml/include
)

# Test timestep embedding + modulation
add_executable(test_timestep
    tests/c_ops/test_timestep.cpp
)

target_link_libraries(test_timestep PRIVATE ggml ggml-cpu)
target_include_directories(test_timestep PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp/include
    ${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp/ggml/include
)

# Test double block (Flux2TransformerBlock)
add_executable(test_double_block
    tests/c_ops/test_double_block.cpp
)

target_link_libraries(test_double_block PRIVATE ggml ggml-cpu)
target_include_directories(test_double_block PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp/include
    ${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp/ggml/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

