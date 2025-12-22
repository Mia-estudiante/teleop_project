cmake_minimum_required(VERSION 3.5)
project(franka_description)

find_package(ament_cmake REQUIRED)
find_package(xacro REQUIRED)

install(DIRECTORY
  meshes
  robots
  rviz
  DESTINATION share/${PROJECT_NAME}
)

# 테스트 (선택적)
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()

  # pytest 또는 launch 테스트 등을 여기에 추가할 수 있음
endif()

ament_package()

