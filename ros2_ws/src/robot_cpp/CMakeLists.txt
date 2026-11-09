cmake_minimum_required(VERSION 3.8)
project(robot_cpp)

# 打开常见警告，帮你在编译时发现问题
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# C++17
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

# 找依赖包
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)

# 可执行文件：从 src/command_node.cpp 编译出 command_node
add_executable(command_node src/command_node.cpp)

# 链接依赖库
ament_target_dependencies(command_node
  rclcpp
  sensor_msgs
)

# 安装到 lib/robot_cpp，这样 ros2 run 才能找到
install(TARGETS
  command_node
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
