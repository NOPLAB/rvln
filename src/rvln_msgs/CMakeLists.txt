cmake_minimum_required(VERSION 3.8)
project(rvln_msgs)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  msg/GoalSpec.msg
  msg/ActionEmbedding.msg
  msg/Observation.msg
  srv/SetGoal.srv
  action/NavigateVLA.action
  DEPENDENCIES std_msgs geometry_msgs sensor_msgs nav_msgs
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # Skip copyright checks because these sources do not use license headers.
  # Skip C++ linters because this package has only generated C++ sources.
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cppcheck_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  set(ament_cmake_uncrustify_FOUND TRUE)
  # Use the repository-wide flake8 configuration.
  set(ament_cmake_flake8_CONFIG_FILE "${CMAKE_CURRENT_SOURCE_DIR}/../../.flake8")
  # Skip pep257 here: Humble does not pass the .pydocstyle ignore list to
  # the automatic hook. CI runs pep257 with that list for this package.
  set(ament_cmake_pep257_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_export_dependencies(rosidl_default_runtime)
ament_package()
