cmake_minimum_required(VERSION 3.16)
project(fr3_policy_streamer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Franka REQUIRED)
find_package(Threads REQUIRED)
# The local libfranka build was configured with these private shared-library
# dependencies, but its generated development target does not re-export them.
find_package(Poco REQUIRED COMPONENTS Net Foundation)
find_package(pinocchio REQUIRED)
find_package(TinyXML2 REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(fr3_joint_velocity_streamer fr3_joint_velocity_streamer.cpp)
target_link_libraries(fr3_joint_velocity_streamer PRIVATE
  Franka::Franka
  Threads::Threads
  Poco::Foundation
  Poco::Net
  pinocchio::pinocchio
  tinyxml2::tinyxml2
  OpenMP::OpenMP_CXX
)
target_compile_options(fr3_joint_velocity_streamer PRIVATE -Wall -Wextra -Wpedantic)

# Controller-ablation executable. It shares only the UDP/state plumbing with
# the deployed streamer; its 1 kHz command tracking and predictive guard use
# libfranka's official joint-velocity rate limiter.
add_executable(fr3_reference_joint_velocity_streamer fr3_joint_velocity_streamer.cpp)
target_compile_definitions(fr3_reference_joint_velocity_streamer PRIVATE
  FR3_REFERENCE_CONTROLLER=1
)
target_link_libraries(fr3_reference_joint_velocity_streamer PRIVATE
  Franka::Franka
  Threads::Threads
  Poco::Foundation
  Poco::Net
  pinocchio::pinocchio
  tinyxml2::tinyxml2
  OpenMP::OpenMP_CXX
)
target_compile_options(fr3_reference_joint_velocity_streamer PRIVATE
  -Wall -Wextra -Wpedantic
)

enable_testing()
add_executable(fr3_reference_limiter_test test_reference_limiter.cpp)
target_link_libraries(fr3_reference_limiter_test PRIVATE
  Franka::Franka
  Threads::Threads
  Poco::Foundation
  Poco::Net
  pinocchio::pinocchio
  tinyxml2::tinyxml2
  OpenMP::OpenMP_CXX
)
target_compile_options(fr3_reference_limiter_test PRIVATE -Wall -Wextra -Wpedantic)
add_test(NAME fr3_reference_limiter_test COMMAND fr3_reference_limiter_test)
