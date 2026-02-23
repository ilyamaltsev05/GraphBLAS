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
        #include "stdio.h"
        GB_convert_any_to_sparse (M, Werk) ;

        // iterate over mask, count how many elements will be present in MT
        // determine number of entries in MT (MT->p basically)

        GB_MATRIX_WAIT(M);
        GB_MATRIX_WAIT(A);
        GB_MATRIX_WAIT(B);

        GxB_Matrix_fprint (A, "A", GxB_COMPLETE, stdout) ;
        GxB_Matrix_fprint (B, "B", GxB_COMPLETE, stdout) ;

        size_t allocated = 0 ;
        bool MT_hypersparse = (A->h != NULL) || (B->h != NULL);
        uint64_t centries, nvecs;
        centries = 0 ;
        nvecs = 0 ;
        int32_t* MTp32 = NULL ; int64_t* MTp64 = NULL ;
        MTp32 = M->p_is_32 ? GB_calloc_memory (M->vdim + 2, sizeof(int32_t), &allocated) : NULL ;
        MTp64 = M->p_is_32 ? NULL : GB_calloc_memory (M->vdim + 2, sizeof(int64_t), &allocated) ;

        // declare needed pointers
        GB_Mp_DECLARE(Mp, ) ;
        GB_Mp_PTR(Mp, M) ;

        GB_Mh_DECLARE(Mh, ) ;
        GB_Mh_PTR(Mh, M) ;

        GB_Mi_DECLARE(Mi, ) ;
        GB_Mi_PTR(Mi, M) ;

        #define GBI(Ai,p,avlen) ((Ai == NULL) ? ((p) % (avlen)) : Ai [p])
        #define GBB(Ab,p)       ((Ab == NULL) ? 1 : Ab [p])
        #define GBP(Ap,k,avlen) ((Ap == NULL) ? ((k) * (avlen)) : Ap [k])
        #define GBH(Ah,k)       ((Ah == NULL) ? (k) : Ah [k])

        #include "../element/GrB_Matrix_extractElement.c"

        GxB_Matrix_fprint (M, "M", GxB_COMPLETE, stdout) ;

        int64_t vlen = M->vlen ;
        #pragma omp parallel
        {
            GrB_Scalar a_elem ;
            GrB_Matrix_new(&a_elem, A->type, 1, 1) ;
            GrB_Scalar b_elem ;
            GrB_Matrix_new(&b_elem, B->type, 1, 1) ;

            #pragma omp for reduction(+:nvecs)
            for (GrB_Index k = 0 ; k < M->nvec ; k++)
            {
                GrB_Index j = Mh32 ? GBH (Mh32, k) : GBH (Mh64, k) ;
                // operate on column A(:,j)
                int64_t pA_start = Mp32 ? GBP (Mp32, k, vlen) : GBP(Mp64, k, vlen) ;
                int64_t pA_end   = Mp32 ? GBP (Mp32, k+1, vlen) : GBP(Mp64, k+1, vlen) ;
                //if (pA_start != pA_end) nvecs++ ;
                bool nonempty = false ;
                for (GrB_Index p = pA_start ; p < pA_end ; p++)
                {
                    if (!GBB (M->b, p)) continue ;
                    // entry A(i,j) with row index i and value aij
                    int64_t i = Mi32 ? GBI (Mi32, p, vlen) : GBI (Mi64, p, vlen) ;
                    printf("i and j: %d %d\n", i, j) ;
                    GrB_Index Mrow = M->is_csc ? i : j ; GrB_Index Mcol = M->is_csc ? j : i ;
                    printf("%s: %d %d\n", "row and col", Mrow, Mcol);

                    // extract elements from A and B, increment MTp
                    if (Mask_struct || (M->iso ? ((int8_t*)M->x)[0] : ((int8_t*)M->x)[p])) {
                        GrB_Index bh = B_transpose ? bncols : bnrows ;
                        GrB_Index bw = B_transpose ? bnrows : bncols ;

                        GrB_Index arow = A_transpose ? (Mcol / bw) : (Mrow / bh);
                        GrB_Index acol = A_transpose ? (Mrow / bh) : (Mcol / bw);

                        GrB_Index brow = B_transpose ? (Mcol % bw) : (Mrow % bh);
                        GrB_Index bcol = B_transpose ? (Mrow % bh) : (Mcol % bw);


                        GrB_Index code = GrB_Matrix_extractElement_Scalar(a_elem, A, arow, acol) ;
                        printf("a elem code nvals: %d %d\n", code, a_elem->nvals) ;
                        if (code != GrB_SUCCESS || a_elem->nvals != 1) {
                            continue;
                        }

                        code = GrB_Matrix_extractElement_Scalar(b_elem, B, brow, bcol) ;
                        printf("b elem code: %d\n", code) ;
                        if (code != GrB_SUCCESS || b_elem->nvals != 1) {
                            continue;
                        }

                        printf("incrementing: %d\n", k + 1) ;

                        if (M->h == NULL) {
                            if (M->p_is_32) {
                                #pragma omp atomic update
                                (MTp32[k + 1])++ ;
                            }
                            else {
                                #pragma omp atomic update
                                (MTp64[k + 1])++ ;
                            }
                        }
                        else {
                            int64_t vec = M->j_is_32 ? ((int32_t *)M->h)[k] : ((int64_t *)M->h)[k] ;
                            if (M->p_is_32)
                                #pragma omp atomic update
                                (MTp32[vec + 1])++ ;
                            else {
                                #pragma omp atomic update
                                (MTp64[vec + 1])++ ;
                            }
                        }

                        nonempty = true ;
                    }
                }
                if (nonempty) nvecs++ ;
            }

            GB_Matrix_free (&a_elem) ;
            GB_Matrix_free (&b_elem) ;
        }

        // GB_cumsum for MT->p
        int cumsum_threads = (M->vdim / 2) > 0 ? (M->vdim / 2) : 1 ;
        M->p_is_32 ? GB_cumsum(MTp32, M->p_is_32, M->vdim + 1, NULL, cumsum_threads, Werk) : GB_cumsum(MTp64, M->p_is_32, 
            M->vdim + 1, NULL, cumsum_threads, Werk) ;

        // another iteration over M to determine MT->i and MT->x

        if (MTp32) {
            memmove(MTp32, MTp32 + 1, (M->vdim) * sizeof(int32_t)) ;
        }
        else {
            memmove(MTp64, MTp64 + 1, (M->vdim) * sizeof(int64_t));
        }

        printf("\nPointers array:\n") ;
        for (int64_t i = 0; i < M->vdim + 1; i++) M->p_is_32 ? printf("%d ", MTp32[i]) : 
        printf("%d ", MTp64[i]) ;
        printf("\n") ;

        centries = M->p_is_32 ? MTp32[M->vdim] : MTp64[M->vdim] ;
        int32_t* MTi32 = NULL ; int64_t* MTi64 = NULL;
        MTi32 = M->i_is_32 ? GB_calloc_memory (centries, sizeof(int32_t), &allocated) : NULL ;
        MTi64 = M->i_is_32 ? NULL : GB_calloc_memory (centries, sizeof(int64_t), &allocated) ;

        void* MTx = GB_calloc_memory (centries, sizeof(op->ztype->size), &allocated) ;

        #pragma omp parallel
        {
            GrB_Scalar a_elem ;
            GrB_Matrix_new(&a_elem, A->type, 1, 1) ;
            GrB_Scalar b_elem ;
            GrB_Matrix_new(&b_elem, B->type, 1, 1) ;
            GrB_Scalar c_elem ;
            GrB_Matrix_new(&c_elem, op->ztype, 1, 1) ;
            size_t allocated = 0 ;
            c_elem->x = GB_calloc_memory (1, op->ztype->size, &allocated) ;
            #pragma omp for
            for (GrB_Index k = 0 ; k < M->nvec ; k++)
            {
                GrB_Index j = Mh32 ? GBH (Mh32, k) : GBH (Mh64, k) ;
                // operate on column A(:,j)
                int64_t pA_start = Mp32 ? GBP (Mp32, k, vlen) : GBP(Mp64, k, vlen) ;
                int64_t pA_end   = Mp32 ? GBP (Mp32, k+1, vlen) : GBP(Mp64, k+1, vlen) ;
                GrB_Index pos = M->p_is_32 ? MTp32[k] : MTp64[k] ;
                printf("k and pos: %d %d\n", k, pos) ;
                for (GrB_Index p = pA_start ; p < pA_end ; p++)
                {
                    if (!GBB (M->b, p)) continue ;
                    // entry A(i,j) with row index i and value aij
                    int64_t i = Mi32 ? GBI (Mi32, p, vlen) : GBI (Mi64, p, vlen) ;
                    GrB_Index Mrow = M->is_csc ? i : j ; GrB_Index Mcol = M->is_csc ? j : i ;
                    printf("%s: %d %d\n", "row and col", Mrow, Mcol);

                    // extract elements from A and B, initialize offset of MTi, get result of op, 
                    // place it in MTx
                    if (Mask_struct || ((int8_t*)M->x)[p]) {

                        GrB_Index bh = B_transpose ? bncols : bnrows ;
                        GrB_Index bw = B_transpose ? bnrows : bncols ;

                        GrB_Index arow = A_transpose ? (Mcol / bw) : (Mrow / bh);
                        GrB_Index acol = A_transpose ? (Mrow / bh) : (Mcol / bw);

                        GrB_Index brow = B_transpose ? (Mcol % bw) : (Mrow % bh);
                        GrB_Index bcol = B_transpose ? (Mrow % bh) : (Mcol % bw);


                        GrB_Index code = GrB_Matrix_extractElement_Scalar(a_elem, A, arow, acol) ;
                        printf("a elem code: %d\n", code) ;
                        if (code != GrB_SUCCESS || a_elem->nvals != 1) {
                            continue;
                        }

                        code = GrB_Matrix_extractElement_Scalar(b_elem, B, brow, bcol) ;
                        printf("b elem code: %d\n", code) ;
                        if (code != GrB_SUCCESS || b_elem->nvals != 1) {
                            continue;
                        }

                        op->binop_function(c_elem->x, a_elem->x, b_elem->x) ;

                        printf("pos: %d\n", pos) ;

                        memcpy(MTx + op->ztype->size * pos, c_elem->x, op->ztype->size) ;

                        if (M->i_is_32) { MTi32[pos] = i ; } else { MTi64[pos] = i ; }
                        pos++ ;
                    }
                }
            }
            GB_Matrix_free (&a_elem) ;
            GB_Matrix_free (&b_elem) ;
            GB_free_memory(&c_elem->x, op->ztype->size) ;
            GB_Matrix_free (&c_elem) ;
        }

        // initialize other fields of MT properly

        // take stuff from kroner, call GB_new_bix and replace needed stuff (pointers)
        // take case of iso in account as well
        // think of index binary operations

        printf("\nPointers array:\n") ;
        for (int64_t i = 0; i < M->vdim + 1; i++) M->p_is_32 ? printf("%d ", MTp32[i]) : 
        printf("%d ", MTp64[i]) ;
        printf("\n") ;

        printf("\nColumns array:\n") ;
        for (int64_t i = 0; i < centries; i++) M->i_is_32 ? printf("%d ", MTi32[i]) : 
        printf("%d ", MTi64[i]) ;
        printf("\n") ;

        printf("\nValues array:\n") ;
        for (int64_t i = 0 ; i < centries ; i++) printf("%d ", ((int32_t *)MTx)[i]) ;
        printf("\n") ;

        //printf("centries: %d\n", centries) ;
        MT = NULL ;
        GB_OK (GB_new_bix (&MT, C->type, vlen, M->vdim, GB_ph_malloc, M->is_csc, 
        GxB_SPARSE, true, M->hyper_switch, M->vdim + 1, centries, true, false, M->p_is_32, 
        M->j_is_32, M->i_is_32)) ;

        MT->h = NULL ;
        MT->nvals = centries ;
        MT->magic = GB_MAGIC ;
        MT->nvec = M->vdim ;
        MT->nvec_nonempty = nvecs ;

        if (MT->p_is_32) {
            MT->p = (void*)(MTp32) ;
        }
        else {
            MT->p = (void*)(MTp64) ;
        }
        MT->i = MT->i_is_32 ? MTi32 : MTi64 ;
        MT->x = MTx ;

        GB_MATRIX_WAIT(MT) ;

        GxB_Matrix_fprint (MT, "MT", GxB_COMPLETE, stdout) ;

        // GB_hyper_prune and transpose and cast if needed

        if (MT->is_csc != C->is_csc) 
        {
            GB_transpose_in_place (MT, true, Werk) ;
        }

        if (MT_hypersparse) {
            int32_t* MTh32 = NULL ; int64_t* MTh64 =  NULL ;
            if (M->j_is_32) {
                MTh32 = GB_calloc_memory (M->vdim, sizeof(int32_t), &allocated) ;
            }
            else {
                MTh64 = GB_calloc_memory (M->vdim, sizeof(int64_t), &allocated) ;
            }

            #pragma omp parallel for
            for (GrB_Index i = 0; i < M->vdim; i++) {
                if (M->j_is_32) { MTh32[i] = i ; } else { MTh64[i] = i ; } 
            }

            MT->h = MTh32 ? MTh32 : MTh64 ;

            GB_hyper_prune (MT, Werk) ;

            GxB_Matrix_fprint (MT, "hyper MT", GxB_COMPLETE, stdout) ;
        }

        // return GB_accum_mask

        return (GB_accum_mask (C, M, NULL, accum, &MT, C_replace, Mask_comp, Mask_struct, Werk)) ;
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

