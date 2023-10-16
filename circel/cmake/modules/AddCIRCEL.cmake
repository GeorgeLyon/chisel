include_guard()

function(add_circel_dialect dialect dialect_namespace)
  add_mlir_dialect(${ARGV})
  add_dependencies(circel-headers MLIR${dialect}IncGen)
endfunction()

function(add_circel_interface interface)
  add_mlir_interface(${ARGV})
  add_dependencies(circel-headers MLIR${interface}IncGen)
endfunction()

# Additional parameters are forwarded to tablegen.
function(add_circel_doc tablegen_file output_path command)
  set(LLVM_TARGET_DEFINITIONS ${tablegen_file}.td)
  string(MAKE_C_IDENTIFIER ${output_path} output_id)
  tablegen(MLIR ${output_id}.md ${command} ${ARGN})
  set(GEN_DOC_FILE ${CIRCEL_BINARY_DIR}/docs/${output_path}.md)
  add_custom_command(
          OUTPUT ${GEN_DOC_FILE}
          COMMAND ${CMAKE_COMMAND} -E copy
                  ${CMAKE_CURRENT_BINARY_DIR}/${output_id}.md
                  ${GEN_DOC_FILE}
          DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${output_id}.md)
  add_custom_target(${output_id}DocGen DEPENDS ${GEN_DOC_FILE})
  add_dependencies(circel-doc ${output_id}DocGen)
endfunction()

function(add_circel_dialect_doc dialect dialect_namespace)
  add_circel_doc(
    ${dialect} Dialects/${dialect}
    -gen-dialect-doc -dialect ${dialect_namespace})
endfunction()

function(add_circel_library name)
  add_mlir_library(${ARGV})
  add_circel_library_install(${name})
endfunction()

macro(add_circel_executable name)
  add_llvm_executable(${name} ${ARGN})
  set_target_properties(${name} PROPERTIES FOLDER "circel executables")
endmacro()

macro(add_circel_tool name)
  if (NOT CIRCEL_BUILD_TOOLS)
    set(EXCLUDE_FROM_ALL ON)
  endif()

  add_circel_executable(${name} ${ARGN})

  if (CIRCEL_BUILD_TOOLS)
    get_target_export_arg(${name} CIRCEL export_to_circeltargets)
    install(TARGETS ${name}
      ${export_to_circeltargets}
      RUNTIME DESTINATION "${CIRCEL_TOOLS_INSTALL_DIR}"
      COMPONENT ${name})

    if(NOT CMAKE_CONFIGURATION_TYPES)
      add_llvm_install_targets(install-${name}
        DEPENDS ${name}
        COMPONENT ${name})
    endif()
    set_property(GLOBAL APPEND PROPERTY CIRCEL_EXPORTS ${name})
  endif()
endmacro()

# Adds a CIRCEL library target for installation.  This should normally only be
# called from add_circel_library().
function(add_circel_library_install name)
  install(TARGETS ${name} COMPONENT ${name} EXPORT CIRCELTargets)
  set_property(GLOBAL APPEND PROPERTY CIRCEL_ALL_LIBS ${name})
  set_property(GLOBAL APPEND PROPERTY CIRCEL_EXPORTS ${name})
endfunction()

function(add_circel_dialect_library name)
  set_property(GLOBAL APPEND PROPERTY CIRCEL_DIALECT_LIBS ${name})
  add_circel_library(${ARGV} DEPENDS circel-headers)
endfunction()

function(add_circel_conversion_library name)
  set_property(GLOBAL APPEND PROPERTY CIRCEL_CONVERSION_LIBS ${name})
  add_circel_library(${ARGV} DEPENDS circel-headers)
endfunction()

function(add_circel_translation_library name)
  set_property(GLOBAL APPEND PROPERTY CIRCEL_TRANSLATION_LIBS ${name})
  add_circel_library(${ARGV} DEPENDS circel-headers)
endfunction()

function(add_circel_verification_library name)
  set_property(GLOBAL APPEND PROPERTY CIRCEL_VERIFICATION_LIBS ${name})
  add_circel_library(${ARGV} DEPENDS circel-headers)
endfunction()
