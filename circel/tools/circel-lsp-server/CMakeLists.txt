set(LIBS
  CIRCTFIRRTL
  MLIRLspServerLib
  )

add_llvm_tool(circel-lsp-server
  circel-lsp-server.cpp

  DEPENDS
  ${LIBS}
  )
target_link_libraries(circel-lsp-server PRIVATE ${LIBS})
llvm_update_compile_flags(circel-lsp-server)

mlir_check_all_link_libraries(circel-lsp-server)
