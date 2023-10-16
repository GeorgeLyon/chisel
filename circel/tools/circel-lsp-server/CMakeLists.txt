add_llvm_tool(circel-lsp-server
  circel-lsp-server.cpp)
target_link_libraries(circel-lsp-server PRIVATE 
  CIRCTFIRRTL
  MLIRLspServerLib)
llvm_update_compile_flags(circel-lsp-server)

mlir_check_all_link_libraries(circel-lsp-server)
