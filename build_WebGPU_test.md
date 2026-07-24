# Build Bonsai.cpp + Dawn on Windows — Complete Guide

## 1. Build Dawn (WebGPU — Vulkan only)

```powershell
cd I:\bonsai_windows\dawn

# Clean if already built
Remove-Item -Recurse -Force out/Release

# Configure
cmake -B out/Release -S . -G "Visual Studio 17 2022" `
  -DCMAKE_BUILD_TYPE=Release `
  -DTINT_BUILD_TESTS=OFF `
  -DDAWN_BUILD_TESTS=OFF `
  -DDAWN_ENABLE_D3D11=OFF `
  -DDAWN_ENABLE_D3D12=OFF

# Build
cmake --build out/Release --config Release -j 8
```

---

## 2. Create DawnConfig.cmake (one-time setup)

Create the install directory and the CMake config file:

```powershell
mkdir I:\bonsai_windows\dawn\install
```

Create `I:\bonsai_windows\dawn\install\DawnConfig.cmake` with the following content:

```cmake
set(DAWN_ROOT "I:/bonsai_windows/dawn")
set(DAWN_BUILD "${DAWN_ROOT}/out/Release")

file(GLOB_RECURSE DAWN_LIBS   "${DAWN_BUILD}/src/dawn/**/*.lib")
file(GLOB_RECURSE TINT_LIBS   "${DAWN_BUILD}/tint*/**/*.lib")
file(GLOB_RECURSE THIRD_LIBS  "${DAWN_BUILD}/third_party/**/*.lib" "${DAWN_ROOT}/third_party/**/*.lib")
list(FILTER DAWN_LIBS EXCLUDE REGEX "dawn_proc\\.lib$")

add_library(dawn_all_libs INTERFACE)
target_include_directories(dawn_all_libs INTERFACE
    ${DAWN_ROOT}/include ${DAWN_ROOT}/src
    ${DAWN_BUILD}/gen/include ${DAWN_ROOT}/third_party/abseil-cpp)
target_link_libraries(dawn_all_libs INTERFACE ${DAWN_LIBS} ${TINT_LIBS} ${THIRD_LIBS})
add_library(dawn::webgpu_dawn ALIAS dawn_all_libs)
```

---

## 3. Sync code from Linux (if needed)

```powershell
copy \\wsl$\Ubuntu\home\leo\bonsai_v2\bonsai.cpp\tests\c_ops\test_webgpu_diffuser_ops.cpp tests\c_ops\
copy \\wsl$\Ubuntu\home\leo\bonsai_v2\bonsai.cpp\CMakeLists.txt .
copy \\wsl$\Ubuntu\home\leo\bonsai_v2\bonsai.cpp\src\b1_0_kernel.h src\
copy \\wsl$\Ubuntu\home\leo\bonsai_v2\bonsai.cpp\src\diffuser_graph.cpp src\
```

---

## 4. Configure Bonsai.cpp

```powershell
cd I:\bonsai_windows\bonsai.cpp

# Clean if already built
Remove-Item -Recurse -Force build

# Configure
cmake -B build -S . -G "Visual Studio 17 2022" `
  -DGGML_WEBGPU=ON `
  -DGGML_VULKAN=ON `
  -DGGML_AVX2=ON `
  -DDawn_DIR=I:/bonsai_windows/dawn/install `
  -DCMAKE_BUILD_TYPE=Release
```

---

## 5. Build

```powershell
# All targets
cmake --build build --config Release -j 8 `
  --target bonsai_diffuser `
  --target test_webgpu_add `
  --target test_webgpu_ops `
  --target test_webgpu_diffuser_ops

# Or a single target
cmake --build build --config Release -j 8 --target test_webgpu_diffuser_ops
```

---

## 6. Copy vulkan-1.dll next to the executables

```powershell
copy C:\Windows\System32\vulkan-1.dll build\Release\
```

> ⚠️ Must be done after every rebuild — MSBuild cleans the `Release/` output directory.

---

## 7. Run GPU tests

```powershell
.\build\Release\test_webgpu_add.exe           # Smoke test (ggml_add)
.\build\Release\test_webgpu_ops.exe           # Custom ops (b1_linear, rope_2d)
.\build\Release\test_webgpu_diffuser_ops.exe  # Diffuser components (4/5)
```

---

## Directory layout

```
I:\bonsai_windows\
├── dawn\
│   ├── include\webgpu\              ← WebGPU headers
│   ├── src\                         ← Dawn sources
│   ├── out\Release\                 ← Dawn build (Visual Studio)
│   │   ├── gen\include\dawn\        ← Generated headers
│   │   └── src\dawn\**\*.lib        ← Static .lib files
│   └── install\
│       └── DawnConfig.cmake         ← Hand-written (step 2)
└── bonsai.cpp\
    ├── build\Release\               ← Binaries (.exe)
    │   └── vulkan-1.dll             ← Copied from System32
    ├── src\                         ← Source code
    └── tests\c_ops\                 ← Tests
```
