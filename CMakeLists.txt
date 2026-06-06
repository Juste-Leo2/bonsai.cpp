cmake_minimum_required(VERSION 3.16)
project(bonsai.cpp VERSION 0.1.0 LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable CUDA support
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(LLAMA_CUDA ON CACHE BOOL "Enable CUDA support in llama.cpp" FORCE)
else()
    message(WARNING "CUDA compiler not found. Falling back to CPU only.")
    set(LLAMA_CUDA OFF CACHE BOOL "Enable CUDA support in llama.cpp" FORCE)
endif()

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


