cmake_minimum_required(VERSION 3.8)
project(igris_c_description)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(urdf REQUIRED)
find_package(xacro REQUIRED)

install(
  FILES igris_c_v2.xml igris_c_v2_.xml igris_c_v2_hand.xml igris_c_v2_cube.xml igris_c_v2_pos.xml igris_c_v2_converted.xml
  DESTINATION share/${PROJECT_NAME}
)

install(
  FILES scene.xml scene_.xml
  DESTINATION share/${PROJECT_NAME}
)

install(DIRECTORY meshes urdf usd hand launch
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
