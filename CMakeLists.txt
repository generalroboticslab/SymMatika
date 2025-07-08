cmake_minimum_required(VERSION 3.28)

if(WIN32)
   set(CMAKE_C_COMPILER   "C:/Program Files/LLVM/bin/clang-cl.exe"  CACHE FILEPATH "" FORCE)
   set(CMAKE_CXX_COMPILER "C:/Program Files/LLVM/bin/clang-cl.exe"  CACHE FILEPATH "" FORCE)
elseif (APPLE)
   set(CMAKE_C_COMPILER   "/opt/homebrew/opt/llvm/bin/clang"   CACHE FILEPATH "" FORCE)
   set(CMAKE_CXX_COMPILER "/opt/homebrew/opt/llvm/bin/clang++" CACHE FILEPATH "" FORCE)
endif()

project(SymMatika_Engine LANGUAGES CXX)

if(APPLE)
   execute_process(
           COMMAND xcrun --sdk macosx --show-sdk-path
           OUTPUT_VARIABLE _sdk_path
           OUTPUT_STRIP_TRAILING_WHITESPACE
   )
   set(CMAKE_OSX_SYSROOT   "${_sdk_path}"   CACHE PATH    ""   FORCE)
   set(CMAKE_C_FLAGS       "${CMAKE_C_FLAGS}   -isysroot ${_sdk_path}"   CACHE STRING "" FORCE)
   set(CMAKE_CXX_FLAGS     "${CMAKE_CXX_FLAGS} -isysroot ${_sdk_path}"   CACHE STRING "" FORCE)
endif()

set(ENGINE_ROOT "${CMAKE_CURRENT_LIST_DIR}")
list(APPEND CMAKE_MODULE_PATH "${ENGINE_ROOT}/cmake")

include(dependencies)
include(optimizations)

   add_executable(Backup___SymReg
           "${ENGINE_ROOT}/main.cpp"
   )
   target_link_libraries(Backup___SymReg
           PRIVATE
           project_dependencies
           data_processing
           computation
           OpenMP::OpenMP_CXX
           Eigen3::Eigen
   )
   target_compile_options(Backup___SymReg
           PRIVATE
           -fopenmp
           -g
   )
