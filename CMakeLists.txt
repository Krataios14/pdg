cmake_minimum_required(VERSION 3.16)
project(pdg VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(PDG_BUILD_TESTS    "Build unit tests"  ON)
option(PDG_BUILD_EXAMPLES "Build examples"    ON)
option(PDG_WARNINGS       "Enable warnings"   ON)

add_library(pdg STATIC
  src/linalg.cpp
  src/socp.cpp
  src/lcvx.cpp
  src/scvx.cpp
  src/sim.cpp
  src/mc.cpp
)
add_library(pdg::pdg ALIAS pdg)
target_include_directories(pdg PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

if(PDG_WARNINGS)
  if(MSVC)
    target_compile_options(pdg PRIVATE /W4)
  else()
    target_compile_options(pdg PRIVATE -Wall -Wextra)
  endif()
endif()

find_package(Threads REQUIRED)
target_link_libraries(pdg PUBLIC Threads::Threads)

if(PDG_BUILD_TESTS)
  enable_testing()
  foreach(t linalg socp lcvx scvx sim)
    add_executable(test_${t} tests/test_${t}.cpp)
    target_link_libraries(test_${t} PRIVATE pdg)
    add_test(NAME ${t} COMMAND test_${t})
  endforeach()
endif()

if(PDG_BUILD_EXAMPLES)
  foreach(e 3dof_mars 6dof_landing monte_carlo)
    add_executable(ex_${e} examples/ex_${e}.cpp)
    target_link_libraries(ex_${e} PRIVATE pdg)
  endforeach()
endif()

install(TARGETS pdg EXPORT pdgTargets ARCHIVE DESTINATION lib)
install(DIRECTORY include/pdg DESTINATION include)
