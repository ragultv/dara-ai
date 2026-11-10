cmake_minimum_required(VERSION 3.24)
project(dara LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# CUDA is optional until the toolkit is installed (Phase 0 prerequisite).
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
  # Consumer NVIDIA GPUs supported by CUDA 13: Turing (sm_75), Ampere (86), Ada (89), Blackwell (120) + PTX for newer cards.
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "75-real;86-real;89-real;120-real;120-virtual")
  endif()
  enable_language(CUDA)
  set(DARA_HAS_CUDA ON)
else()
  message(STATUS "CUDA toolkit not found: building CPU-only")
endif()

add_library(dara_io src/io/gguf.cpp src/io/safetensors.cpp)
target_include_directories(dara_io PUBLIC src)
if(MSVC)
  target_compile_options(dara_io PRIVATE /W4 /permissive-)
endif()

# nlohmann/json is vendored (third_party/README.md) because vcpkg's tool downloads from GitHub
# releases were failing on the dev machine. Switch back to vcpkg once that works.
add_library(nlohmann_json INTERFACE)
target_include_directories(nlohmann_json SYSTEM INTERFACE third_party)
add_library(nlohmann_json::nlohmann_json ALIAS nlohmann_json)
add_library(dara_core src/core/budget.cpp src/core/placement.cpp src/core/session.cpp
                      src/core/manifest.cpp src/core/system.cpp src/core/nvml_provider.cpp)
target_include_directories(dara_core PUBLIC src)
target_link_libraries(dara_core PUBLIC nlohmann_json::nlohmann_json)
if(MSVC)
  target_compile_options(dara_core PRIVATE /W4 /permissive-)
endif()

add_library(dara_model src/model/deltanet.cpp src/model/ops.cpp src/model/attention.cpp src/model/layer.cpp)
target_include_directories(dara_model PUBLIC src)
target_link_libraries(dara_model PUBLIC dara_core)
if(MSVC)
  target_compile_options(dara_model PRIVATE /W4 /permissive-)
endif()

enable_testing()
add_executable(dara_tests tests/io_tests.cpp)
target_link_libraries(dara_tests PRIVATE dara_io)
add_test(NAME io COMMAND dara_tests)

add_executable(dara_core_tests tests/core_tests.cpp)
target_link_libraries(dara_core_tests PRIVATE dara_core)
target_compile_definitions(dara_core_tests PRIVATE DARA_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
add_test(NAME core COMMAND dara_core_tests)

add_executable(dara_deltanet_tests tests/deltanet_test.cpp)
target_link_libraries(dara_deltanet_tests PRIVATE dara_model)
target_include_directories(dara_deltanet_tests PRIVATE tests)
# Needs build/golden/deltanet_layer0.bin (tools/make_deltanet_golden.py); run manually, not part of the default ctest set.

add_executable(dara_ops_tests tests/ops_test.cpp)
target_link_libraries(dara_ops_tests PRIVATE dara_model)
target_include_directories(dara_ops_tests PRIVATE tests)

add_executable(dara_attention_tests tests/attention_test.cpp)
target_link_libraries(dara_attention_tests PRIVATE dara_model)
target_include_directories(dara_attention_tests PRIVATE tests)

add_executable(dara_layer_tests tests/layer_test.cpp)
target_link_libraries(dara_layer_tests PRIVATE dara_model)
target_include_directories(dara_layer_tests PRIVATE tests)

add_library(dara_tokenizer src/tokenizer/unicode.cpp src/tokenizer/tokenizer.cpp src/tokenizer/gemma_tokenizer.cpp src/tokenizer/gemma_chat.cpp)
target_include_directories(dara_tokenizer PUBLIC src)
target_link_libraries(dara_tokenizer PUBLIC dara_io)
if(MSVC)
  target_compile_options(dara_tokenizer PRIVATE /W4 /permissive- /utf-8)
endif()

add_executable(dara_tokenizer_tests tests/tokenizer_test.cpp)
target_link_libraries(dara_tokenizer_tests PRIVATE dara_tokenizer nlohmann_json::nlohmann_json)
# Needs the GGUF and build/golden/tokenizer_golden.json (tools/make_tokenizer_golden.py); run manually.

add_executable(dara_gemma_tokenizer_tests tests/gemma_tokenizer_test.cpp)
target_link_libraries(dara_gemma_tokenizer_tests PRIVATE dara_tokenizer nlohmann_json::nlohmann_json)
# Needs the Gemma GGUF and build/golden/gemma_tokenizer_golden.json (tools/make_gemma_tokenizer_golden.py); run manually.

add_executable(dara_gemma_chat_tests tests/gemma_chat_test.cpp)
target_link_libraries(dara_gemma_chat_tests PRIVATE dara_tokenizer nlohmann_json::nlohmann_json)
# Needs build/golden/gemma_chat_golden.json (tools/make_gemma_chat_golden.py); run manually.

add_executable(dara_tokenize tools/tokenize.cpp)
target_link_libraries(dara_tokenize PRIVATE dara_tokenizer)

add_executable(dara_inspect tools/inspect.cpp)
target_link_libraries(dara_inspect PRIVATE dara_io)
