set(MTS_PLUGIN_PREFIX "samplers")

add_plugin(independent  independent.cpp)
add_plugin(stratified   stratified.cpp)

# Register the test directory
add_tests(${CMAKE_CURRENT_SOURCE_DIR}/tests)
