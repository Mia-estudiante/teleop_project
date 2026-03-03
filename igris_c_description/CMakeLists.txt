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
  FILES igris_c_v2_.xml
  DESTINATION share/${PROJECT_NAME}
)

install(
  FILES scene_.xml
  DESTINATION share/${PROJECT_NAME}
)

install(
  FILES igris_c_v2.xml
  DESTINATION share/${PROJECT_NAME}
)

install(
  FILES igris_c_v2_hand.xml
  DESTINATION share/${PROJECT_NAME}
)

# install(
#   FILES igris_c_v2_hand_mimic.urdf
#   DESTINATION share/${PROJECT_NAME}
# )

install(
  FILES scene.xml
  DESTINATION share/${PROJECT_NAME}
)

install(DIRECTORY meshes
  DESTINATION share/${PROJECT_NAME}
)

install(DIRECTORY urdf
  DESTINATION share/${PROJECT_NAME}
)

install(DIRECTORY hand
  DESTINATION share/${PROJECT_NAME}
)

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
