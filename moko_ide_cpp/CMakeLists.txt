cmake_minimum_required(VERSION 3.10)
project(moko_ide VERSION 6.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

find_package(Qt5 REQUIRED COMPONENTS Core Widgets Network)

set(SOURCES
    main.cpp
    moko_window.cpp
    code_editor.cpp
    find_bar.cpp
    terminal_widget.cpp
    chat_widget.cpp
    helper_engine.cpp
    syntax_highlighter.cpp
    settings_dialog.cpp
    stress_test_dialog.cpp
    graphify_dialog.cpp
)

set(HEADERS
    moko_window.h
    code_editor.h
    find_bar.h
    terminal_widget.h
    chat_widget.h
    helper_engine.h
    syntax_highlighter.h
    settings_dialog.h
    stress_test_dialog.h
    graphify_dialog.h
)

add_executable(moko_ide ${SOURCES} ${HEADERS})

target_link_libraries(moko_ide
    Qt5::Core
    Qt5::Widgets
    Qt5::Network
    util
)

install(TARGETS moko_ide DESTINATION ${CMAKE_SOURCE_DIR}/..)
