cmake_minimum_required(VERSION 3.8)
project(raspicat_vla_msgs)

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
  # ライセンスヘッダを付けない方針のため copyright はスキップ。
  # 自前の C++ ソースを持たない (rosidl 生成物のみ) ため C++ 系 linter もスキップ。
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cppcheck_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  set(ament_cmake_uncrustify_FOUND TRUE)
  # flake8 はリポジトリ共通の設定 (.flake8) を使う。
  set(ament_cmake_flake8_CONFIG_FILE "${CMAKE_CURRENT_SOURCE_DIR}/../../.flake8")
  # pep257 はスキップ: humble の自動フックには ignore リスト (.pydocstyle 参照)
  # を渡せず、find_package(ament_cmake_pep257) するとフックが無条件再登録されて
  # テスト名が重複する。このパッケージの pep257 は CI (ci-lint.yml) が同じ
  # ignore リスト付きでカバーする。
  set(ament_cmake_pep257_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_export_dependencies(rosidl_default_runtime)
ament_package()
