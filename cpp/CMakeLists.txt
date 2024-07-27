cmake_minimum_required(VERSION 3.10)
project(similarity_search)
enable_language(CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# add the library and executables
add_library(SimilaritySearch STATIC search.cpp score.cpp score.cu heap.cpp)
add_executable(SimilaritySearchBenchmark benchmark.cpp)
target_link_libraries(SimilaritySearchBenchmark SimilaritySearch)

# CUDA
set_target_properties(SimilaritySearch
    PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# pybind
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

include_directories(${Python3_INCLUDE_DIRS})
link_directories(${Python3_LIBRARY_DIRS})

pybind11_add_module(similarity_search search.cpp)

target_link_libraries(SimilaritySearch PRIVATE ${Python3_LIBRARIES})
target_link_libraries(similarity_search PRIVATE SimilaritySearch ${Python3_LIBRARIES})
target_include_directories(SimilaritySearch PRIVATE ${pybind11_INCLUDE_DIRS})
target_include_directories(SimilaritySearchBenchmark PRIVATE ${pybind11_INCLUDE_DIRS})
target_include_directories(similarity_search PRIVATE ${pybind11_INCLUDE_DIRS})

# copy library to python
install(TARGETS similarity_search LIBRARY DESTINATION ${CMAKE_SOURCE_DIR}/../python)
