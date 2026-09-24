cmake_minimum_required(VERSION 3.8)
project(rvln_bringup)

find_package(ament_cmake REQUIRED)

install(DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME}/
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # ライセンスヘッダを付けない方針のため copyright はスキップ。
  # C++ ソースを持たないため C++ 系 linter もスキップ。
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

ament_package()
