cmake_minimum_required(VERSION 3.15)
project(geometry_web_demo CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (NOT EMSCRIPTEN)
  message(FATAL_ERROR "This project targets Emscripten only. Configure with:\n  emcmake cmake -B build\n  cmake --build build")
endif()

include(FetchContent)

# Header-only math library, pinned to the version the original project used.
FetchContent_Declare(
  glm
  GIT_REPOSITORY https://github.com/g-truc/glm.git
  GIT_TAG 0.9.9.7
)
FetchContent_MakeAvailable(glm)

# Dear ImGui, pinned to the version the original project used.
# GLFW itself is NOT fetched - Emscripten provides its own GLFW3 shim
# (enabled below via -sUSE_GLFW=3) that talks to the browser/canvas.
FetchContent_Declare(
  imgui
  GIT_REPOSITORY https://github.com/ocornut/imgui.git
  GIT_TAG v1.89.2
)
FetchContent_MakeAvailable(imgui)

add_executable(app
  main.cpp
  ${imgui_SOURCE_DIR}/imgui.cpp
  ${imgui_SOURCE_DIR}/imgui_draw.cpp
  ${imgui_SOURCE_DIR}/imgui_tables.cpp
  ${imgui_SOURCE_DIR}/imgui_widgets.cpp
  ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
  ${imgui_SOURCE_DIR}/backends/imgui_impl_opengl3.cpp
)

target_include_directories(app PRIVATE
  ${imgui_SOURCE_DIR}
  ${imgui_SOURCE_DIR}/backends
  ${glm_SOURCE_DIR}
)

target_compile_definitions(app PRIVATE IMGUI_IMPL_OPENGL_ES3)

set(EMS_FLAGS
  "-sUSE_GLFW=3"
  "-sUSE_WEBGL2=1"
  "-sMIN_WEBGL_VERSION=2"
  "-sMAX_WEBGL_VERSION=2"
  "-sALLOW_MEMORY_GROWTH=1"
  "-sNO_EXIT_RUNTIME=1"
)

target_compile_options(app PRIVATE ${EMS_FLAGS})
target_link_options(app PRIVATE ${EMS_FLAGS} "--shell-file=${CMAKE_SOURCE_DIR}/shell.html")

if (CMAKE_BUILD_TYPE STREQUAL "" OR NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set_target_properties(app PROPERTIES
  OUTPUT_NAME "index"
  SUFFIX ".html"
)
