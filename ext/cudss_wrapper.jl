LS.DirectSolver(J::CuSparseMatrixCSR; options...) = ExaPF.LS.DirectSolver(lu(J))
