//------------------------------------------------------------------------------
// GB_kron: C<M> = accum (C, kron(A,B))
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// C<M> = accum (C, kron(A,B))

// The input matrices A and B are optionally transposed.

#define GB_FREE_WORKSPACE   \
{                           \
    GB_Matrix_free (&AT) ;  \
    GB_Matrix_free (&BT) ;  \
}

#define GB_FREE_ALL         \
{                           \
    GB_FREE_WORKSPACE ;     \
    GB_Matrix_free (&T) ;   \
}

#include "kronecker/GB_kron.h"
#include "mxm/GB_mxm.h"
#include "transpose/GB_transpose.h"
#include "mask/GB_accum_mask.h"

GrB_Info GB_kron                    // C<M> = accum (C, kron(A,B))
(
    GrB_Matrix C,                   // input/output matrix for results
    const bool C_replace,           // if true, clear C before writing to it
    const GrB_Matrix M,             // optional mask for C, unused if NULL
    const bool Mask_comp,           // if true, use !M
    const bool Mask_struct,         // if true, use the only structure of M
    const GrB_BinaryOp accum,       // optional accum for Z=accum(C,T)
    const GrB_BinaryOp op_in,       // defines '*' for kron(A,B)
    const GrB_Matrix A,             // input matrix
    bool A_transpose,               // if true, use A' instead of A
    const GrB_Matrix B,             // input matrix
    bool B_transpose,               // if true, use B' instead of B
    GB_Werk Werk
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    // C may be aliased with M, A, and/or B

    GrB_Info info ;
    struct GB_Matrix_opaque T_header, AT_header, BT_header ;
    GrB_Matrix T = NULL, AT = NULL, BT = NULL ;
    GrB_BinaryOp op = op_in ;

    GB_RETURN_IF_NULL_OR_FAULTY (op) ;
    GB_RETURN_IF_FAULTY_OR_POSITIONAL (accum) ;

    ASSERT_MATRIX_OK (C, "C input for GB_kron", GB0) ;
    ASSERT_MATRIX_OK_OR_NULL (M, "M for GB_kron", GB0) ;
    ASSERT_BINARYOP_OK_OR_NULL (accum, "accum for GB_kron", GB0) ;
    ASSERT_BINARYOP_OK (op, "op for GB_kron", GB0) ;
    ASSERT_MATRIX_OK (A, "A for GB_kron", GB0) ;
    ASSERT_MATRIX_OK (B, "B for GB_kron", GB0) ;

    // check domains and dimensions for C<M> = accum (C,T)
    GB_OK (GB_compatible (C->type, C, M, Mask_struct, accum, op->ztype,
        Werk)) ;

    // T=op(A,B) via op operator, so A and B must be compatible with z=op(a,b)
    GB_OK (GB_BinaryOp_compatible (op, NULL, A->type, B->type, GB_ignore_code,
        Werk)) ;

    // delete any lingering zombies and assemble any pending tuples in A and B,
    // so that cnz = nnz(A) * nnz(B) can be computed.  Updates of C and M are
    // done after this check.
    GB_MATRIX_WAIT (A) ;
    GB_MATRIX_WAIT (B) ;

    // check the dimensions of C
    int64_t anrows = (A_transpose) ? GB_NCOLS (A) : GB_NROWS (A) ;
    int64_t ancols = (A_transpose) ? GB_NROWS (A) : GB_NCOLS (A) ;
    int64_t bnrows = (B_transpose) ? GB_NCOLS (B) : GB_NROWS (B) ;
    int64_t bncols = (B_transpose) ? GB_NROWS (B) : GB_NCOLS (B) ;
    uint64_t cnrows, cncols, cnz = 0 ;
    bool ok = GB_int64_multiply (&cnrows, anrows,  bnrows) ;
    ok = ok && GB_int64_multiply (&cncols, ancols,  bncols) ;
    ok = ok && GB_int64_multiply (&cnz, GB_nnz (A), GB_nnz (B)) ;
    if (!ok || GB_NROWS (C) != cnrows || GB_NCOLS (C) != cncols)
    { 
        GB_ERROR (GrB_DIMENSION_MISMATCH, "%s:\n"
            "output is " GBd "-by-" GBd "; must be " GBu "-by-" GBu "\n"
            "first input is " GBd "-by-" GBd "%s with " GBd " entries\n"
            "second input is " GBd "-by-" GBd "%s with " GBd " entries",
            ok ? "Dimensions not compatible:" : "Problem too large:",
            GB_NROWS (C), GB_NCOLS (C), cnrows, cncols,
            anrows, ancols, A_transpose ? " (transposed)" : "", GB_nnz (A),
            bnrows, bncols, B_transpose ? " (transposed)" : "", GB_nnz (B)) ;
    }

    // quick return if an empty mask is complemented
    GB_RETURN_IF_QUICK_MASK (C, C_replace, M, Mask_comp, Mask_struct) ;

    // TODO: check if can apply matrix immediately in kron, select all non-null entries 
    // with values on corresponding indices on A and B, allocate proper T matrix
    // pass it to to kroner, return the result
    // TODO: MT has to have same CSC/CSR format as resulting C and be sparse or hypersparse

    GrB_Matrix MT;
    if (M != NULL && !Mask_comp && op->binop_function != NULL) {

        #include "../matrix/GB_matrix.h"
        GB_Matrix_new(&MT, op->ztype, GB_NROWS(C), GB_NCOLS(C));

            // leave nonzero entries only with needed A and B values present
            // extract M into tuples, iterate over it and find whether A and B are there and val is nonzero,
            // allocate updated tuples, build MT from them, pass MT to kroner

            GB_MATRIX_WAIT(M);

            #include "../nvals/GB_nvals.h"
            GrB_Index nvals = 0;
            GB_nvals(&nvals, M, Werk);

            #include "../print/GB_check.h"
            GB_Type_check(A->type, "A type", GxB_COMPLETE, stdout);
            GB_Type_check(B->type, "B type", GxB_COMPLETE, stdout);

            fflush(stdout);

            // allocate tuples with GB_malloc_memory
            size_t allocated = 0;
            GrB_Index* I_ind = GB_malloc_memory(nvals, sizeof(GrB_Index), &allocated);
            printf("%s\n", "write this way");
            GrB_Index* J_ind = GB_malloc_memory(1, sizeof(GrB_Index) * nvals, &allocated);
            bool* vals = GB_malloc_memory(1, sizeof(bool) * nvals, &allocated);

            // array to store elements of C (use calloc)
            GB_void* c_elems = GB_calloc_memory(1, op->ztype->size * nvals, &allocated);

            // array to indicate presence of value (use calloc)
            bool* c_pres = GB_calloc_memory(1, sizeof(bool) * nvals, &allocated);
            printf("%s\n", "fall");

            // extract into I_ind, J_ind, vals

            #include "../extractTuples/GB_extractTuples.h"
            GB_extractTuples(I_ind, false, J_ind, false, vals, &nvals, M->type, M, Werk);

            // get B sizes to compute indices

            GrB_Index brows = 0;
            GrB_Index bcols = 0;

            if (B_transpose) {
                bcols = GB_NROWS(B); 
                brows = GB_NCOLS(B);
            }
            else {
                brows = GB_NROWS(B);
                bcols = GB_NCOLS(B);
            }

            // iterate over tuples to count nonzeros with present vals

            GrB_Index nvals_upd = 0;

            #include "GraphBLAS.h"
            printf("Type size A: %zu\n", A->type->size);
            //int32_t* aaa = calloc(1, A->type->size);
            GrB_Scalar aaa;
            GrB_Scalar_new(&aaa, A->type);
            //memset(aaa, 0, A->type->size);
            printf("%d\n", GrB_Matrix_extractElement(aaa, A, 0, 0));
            //memcpy(aaa, ((GB_void*)A->x), 4);
            //int32_t aaa_val;
            //memcpy(&aaa_val, aaa, A->type->size);
            //printf("%d\n", aaa_val);
            GB_void rrr [1];
            printf("%s\n", "one byte only");
            printf("%d\n", GrB_Matrix_extractElement(rrr, A, 0, 0));
            GxB_Scalar_fprint(aaa, "value", GxB_COMPLETE, stdout);
            GrB_Scalar_free(&aaa);

            #pragma omp parallel for reduction(+:nvals_upd)
            for (GrB_Index mval = 0; mval < nvals; mval++) {
                if (!Mask_struct && !vals[mval]) {
                    continue;
                }

                size_t thread_allocated = 0;
                GrB_Info info_elem = 0;
                GB_void extractCheckA[A->type->size];
                GrB_Scalar a_elem;
                GrB_Scalar_new(&a_elem, op->xtype);

                GrB_Scalar b_elem;
                GrB_Scalar_new(&b_elem, op->ytype);

                GrB_Index arow = 0;
                GrB_Index acol = 0;
                if (A_transpose) {
                    arow = J_ind[mval] / bcols;
                    acol = I_ind[mval] / brows;
                }
                else {
                    arow = I_ind[mval] / brows;
                    acol = J_ind[mval] / bcols;
                }

                info_elem = GrB_Matrix_extractElement(extractCheckA, A, arow, acol);
                if (info_elem == GrB_NO_VALUE) {
                    GrB_Scalar_free(&a_elem);
                    GrB_Scalar_free(&b_elem);
                    continue;
                }
                GrB_Matrix_extractElement(a_elem, A, arow, acol);

                GrB_Index browindex = 0;
                GrB_Index bcolindex = 0;

                if (B_transpose) {
                    browindex = J_ind[mval] % bcols;
                    bcolindex = I_ind[mval] % brows;
                }
                else {
                    browindex = I_ind[mval] % brows;
                    bcolindex = J_ind[mval] % bcols;
                }

                GB_void extractCheckB[B->type->size];
                info_elem = GrB_Matrix_extractElement(extractCheckB, B, browindex, bcolindex);
                if (info_elem == GrB_NO_VALUE) {
                    GrB_Scalar_free(&a_elem);
                    GrB_Scalar_free(&b_elem);
                    continue;
                }

                GrB_Matrix_extractElement(b_elem, B, browindex, bcolindex);

                GxB_Scalar_fprint(a_elem, "a_elem", GxB_COMPLETE, stdout);
                GxB_Scalar_fprint(b_elem, "b_elem", GxB_COMPLETE, stdout);

                GrB_Scalar c_elem;
                GrB_Scalar_new(&c_elem, op->ztype);
                c_elem->x = GB_calloc_memory(1, op->ztype->size, &thread_allocated);

                int32_t val_a, val_b;
                memcpy(&val_a, a_elem->x, sizeof(int32_t)); 
                memcpy(&val_b, b_elem->x, sizeof(int32_t)); 

                printf("%d %d\n", val_a, val_b);

                op->binop_function(c_elem->x, a_elem->x, b_elem->x);

                c_pres[mval] = true;

                memcpy(c_elems + (mval * op->ztype->size), c_elem->x, op->ztype->size);


                nvals_upd++;
                GrB_Scalar_free(&a_elem);
                GrB_Scalar_free(&b_elem);
                GB_free_memory(&c_elem->x, op->ztype->size);
                GrB_Scalar_free(&c_elem);
            }

            printf("%s\n", "did first loop");

            // allocate updated tuples of size nvals_upd for building resulting matrix

            GrB_Index* I_ind_upd = GB_malloc_memory(1, sizeof(GrB_Index) * nvals_upd, &allocated);
            GrB_Index* J_ind_upd = GB_malloc_memory(1, sizeof(GrB_Index) * nvals_upd, &allocated);
            GB_void* vals_upd = GB_calloc_memory(1,  op->ztype->size * nvals_upd, &allocated);

            // copy present vals into updated tuples

            GrB_Index next_pos = 0;
            #pragma omp parallel for
            for (GrB_Index resval = 0; resval < nvals; resval++) {
                if (!c_pres[resval]) {
                    continue;
                }

                GrB_Index pos = 0;
                #pragma omp atomic capture 
                {
                    pos = next_pos;
                    next_pos++;
                }

                I_ind_upd[pos] = I_ind[resval];
                J_ind_upd[pos] = J_ind[resval];
                memcpy(vals_upd + (pos * op->ztype->size), c_elems + (resval * op->ztype->size), op->ztype->size);
            }

            // call GrB_build to build MT from tuples

            for (int i = 0; i < nvals_upd; i++) {
                printf("%d %d %d\n", I_ind_upd[i], J_ind_upd[i], ((int32_t *)vals_upd)[i]);
                        }
            
            GxB_Matrix_fprint(MT, "MT", GxB_COMPLETE, stdout);

            //#include "../builder/GB_build.h"
            GrB_Info buildres = GrB_Matrix_build(MT, I_ind_upd, J_ind_upd, (void *)vals_upd, nvals_upd, NULL);

            printf("%d\n", buildres);

            GxB_Matrix_fprint(MT, "MT", GxB_COMPLETE, stdout);

            // transpose MT and set CSC if C->is_csc

            if (MT->is_csc != C->is_csc) {
                GB_transpose(MT, MT->type, MT->is_csc, MT, NULL, NULL, false, false, Werk);
                MT->is_csc = C->is_csc;
            }

            // if MT is not sparse or needs to be hypersparse, convert it

            if (GB_IS_HYPERSPARSE(A) || GB_IS_HYPERSPARSE(B)) {
                GB_convert_any_to_hyper(MT, Werk);
            }

            if (GB_IS_BITMAP(MT)) {
                GB_convert_any_to_sparse(MT, Werk);
            }
            
            printf("%s\n", "did everything");
            GxB_Matrix_fprint(MT, "MT", GxB_COMPLETE, stdout);
            GB_free_memory(&I_ind, nvals * sizeof(GrB_Index));
            GB_free_memory(&J_ind, nvals * sizeof(GrB_Index));
            GB_free_memory(&vals, nvals * sizeof(bool));
            GB_free_memory(&I_ind_upd, nvals_upd * sizeof(GrB_Index));
            GB_free_memory(&J_ind_upd, nvals_upd * sizeof(GrB_Index));
            GB_free_memory(&vals_upd, nvals_upd * op->ztype->size);
            GB_free_memory(&c_elems, nvals * op->ztype->size);
            GB_free_memory(&c_pres, nvals * sizeof(bool));
            GrB_Info kron_info = GB_accum_mask (C, M, NULL, accum, &MT, C_replace, Mask_comp,
        Mask_struct, Werk);

            if (MT != NULL) {
                GrB_Matrix_free(&MT);
            }
            return kron_info;
    }

    //--------------------------------------------------------------------------
    // transpose A and B if requested
    //--------------------------------------------------------------------------

    bool T_is_csc = C->is_csc ;
    if (T_is_csc != A->is_csc)
    { 
        // Negate A_transpose
        A_transpose = !A_transpose ;
    }
    if (T_is_csc != B->is_csc)
    { 
        // Negate B_transpose
        B_transpose = !B_transpose ;
    }

    // do not flipij the builtin positional ops (FIRSTI, and friends);
    // this is no longer needed with the new index binary ops.
    bool flipij = (!T_is_csc) ;

    bool A_is_pattern, B_is_pattern ;
    GB_binop_pattern (&A_is_pattern, &B_is_pattern, false, op->opcode) ;
    if (A_transpose)
    { 
        // AT = A' and typecast to op->xtype
        GBURBLE ("(A transpose) ") ;
        GB_CLEAR_MATRIX_HEADER (AT, &AT_header) ;
        GB_OK (GB_transpose_cast (AT, op->xtype, T_is_csc, A, A_is_pattern,
            Werk)) ;
        ASSERT_MATRIX_OK (AT, "AT kron", GB0) ;
    }

    if (B_transpose)
    { 
        // BT = B' and typecast to op->ytype
        GBURBLE ("(B transpose) ") ;
        GB_CLEAR_MATRIX_HEADER (BT, &BT_header) ;
        GB_OK (GB_transpose_cast (BT, op->ytype, T_is_csc, B, B_is_pattern,
            Werk)) ;
        ASSERT_MATRIX_OK (BT, "BT kron", GB0) ;
    }

    //--------------------------------------------------------------------------
    // T = kron(A,B)
    //--------------------------------------------------------------------------

    GB_CLEAR_MATRIX_HEADER (T, &T_header) ;
    GB_OK (GB_kroner (T, T_is_csc, op, flipij,
        A_transpose ? AT : A, A_is_pattern,
        B_transpose ? BT : B, B_is_pattern, M, Mask_comp, Mask_struct, Werk)) ;

    GB_FREE_WORKSPACE ;
    ASSERT_MATRIX_OK (T, "T = kron(A,B)", GB0) ;

    //--------------------------------------------------------------------------
    // C<M> = accum (C,T): accumulate the results into C via the mask
    //--------------------------------------------------------------------------

    return (GB_accum_mask (C, M, NULL, accum, &T, C_replace, Mask_comp,
        Mask_struct, Werk)) ;
}

