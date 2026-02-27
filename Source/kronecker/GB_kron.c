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

static bool GB_lookup_xoffset (
    GrB_Index* p,
    GrB_Matrix A,
    GrB_Index row,
    GrB_Index col
)
{
    GrB_Index vector = A->is_csc ? col : row ;
    GrB_Index coord  = A->is_csc ? row : col ;

    if (A->p == NULL) {
        GrB_Index offset = vector * A->vlen + coord ;
        if (A->b == NULL || ((int8_t*)A->b)[offset]) {
            *p = offset ;
            return true ;
        }
        return false ;
    }

    int64_t start, end ;
    bool res ;

    if (A->h == NULL) {
        start = A->p_is_32 ? ((uint32_t*)A->p)[vector] : ((uint64_t*)A->p)[vector] ;
        end = A->p_is_32 ? ((uint32_t*)A->p)[vector + 1] : ((uint64_t*)A->p)[vector + 1] ;
        end-- ;
        if (start > end) return false ;
        res = GB_binary_search(coord, A->i, A->i_is_32, &start, &end) ;
        if (res) { *p = start ; }
        return res ;
    } 
    else 
    {
        start = 0 ; end = A->plen - 1 ;
        res = GB_binary_search(vector, A->h, A->j_is_32, &start, &end) ;
        if (!res) return false ;
        int64_t k = start ;
        start = A->p_is_32 ? ((uint32_t*)A->p)[k] : ((uint64_t*)A->p)[k] ;
        end = A->p_is_32 ? ((uint32_t*)A->p)[k+1] : ((uint64_t*)A->p)[k+1] ;
        end-- ;
        if (start > end) return false ;
        res = GB_binary_search(coord, A->i, A->i_is_32, &start, &end) ;
        if (res) { *p = start ; }
        return res ;
    }
}

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

        // iterate over mask, count how many elements will be present in MT
        // determine number of entries in MT (MT->p basically)

        GB_MATRIX_WAIT(M);

        size_t allocated = 0 ;
        bool MT_hypersparse = (A->h != NULL) || (B->h != NULL);
        uint64_t centries, nvecs;
        centries = 0 ;
        nvecs = 0 ;
        uint32_t* MTp32 = NULL ; uint64_t* MTp64 = NULL ;
        MTp32 = M->p_is_32 ? GB_calloc_memory (M->vdim + 2, sizeof(uint32_t), &allocated) : NULL ;
        MTp64 = M->p_is_32 ? NULL : GB_calloc_memory (M->vdim + 2, sizeof(uint64_t), &allocated) ;
        bool MTiso = A->iso && B->iso ;

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

        GB_cast_function cast_A = NULL ;
        GB_cast_function cast_B = NULL ;
        
        cast_A = GB_cast_factory (op->xtype->code, A->type->code) ;
        cast_B = GB_cast_factory (op->ytype->code, B->type->code) ;

        int64_t vlen = M->vlen ;
        #pragma omp parallel
        {
            GrB_Index offset ;

            #pragma omp for reduction(+:nvecs)
            for (GrB_Index k = 0 ; k < M->nvec ; k++)
            {
                GrB_Index j = Mh32 ? GBH (Mh32, k) : GBH (Mh64, k) ;
                // operate on column A(:,j)
                int64_t pA_start = Mp32 ? GBP (Mp32, k, vlen) : GBP(Mp64, k, vlen) ;
                int64_t pA_end   = Mp32 ? GBP (Mp32, k+1, vlen) : GBP(Mp64, k+1, vlen) ;
                bool nonempty = false ;
                for (GrB_Index p = pA_start ; p < pA_end ; p++)
                {
                    if (!GBB (M->b, p)) continue ;
                    // entry A(i,j) with row index i and value aij
                    int64_t i = Mi32 ? GBI (Mi32, p, vlen) : GBI (Mi64, p, vlen) ;
                    GrB_Index Mrow = M->is_csc ? i : j ; GrB_Index Mcol = M->is_csc ? j : i ;

                    // extract elements from A and B, increment MTp
                    if (Mask_struct || (M->iso ? ((int8_t*)M->x)[0] : ((int8_t*)M->x)[p])) {

                        GrB_Index arow = A_transpose ? (Mcol / bncols) : (Mrow / bnrows);
                        GrB_Index acol = A_transpose ? (Mrow / bnrows) : (Mcol / bncols);

                        GrB_Index brow = B_transpose ? (Mcol % bncols) : (Mrow % bnrows);
                        GrB_Index bcol = B_transpose ? (Mrow % bnrows) : (Mcol % bncols);

                        bool code = GB_lookup_xoffset(&offset, A, arow, acol) ;
                        if (!code) {
                            continue;
                        }

                        code = GB_lookup_xoffset(&offset, B, brow, bcol) ;
                        if (!code) {
                            continue;
                        }

                        if (M->h == NULL) {
                            if (M->p_is_32) {
                                (MTp32[k + 1])++ ;
                            }
                            else {
                                (MTp64[k + 1])++ ;
                            }
                        }
                        else {
                            uint64_t vec = M->j_is_32 ? ((uint32_t *)M->h)[k] : ((uint64_t *)M->h)[k] ;
                            if (M->p_is_32)
                                (MTp32[vec + 1])++ ;
                            else {
                                (MTp64[vec + 1])++ ;
                            }
                        }

                        nonempty = true ;
                    }
                }
                if (nonempty) nvecs++ ;
            }
        }

        // GB_cumsum for MT->p
        int cumsum_threads = (M->vdim / 2) > 0 ? (M->vdim / 2) : 1 ;
        M->p_is_32 ? GB_cumsum(MTp32, M->p_is_32, M->vdim + 1, NULL, cumsum_threads, Werk) : GB_cumsum(MTp64, M->p_is_32, 
            M->vdim + 1, NULL, cumsum_threads, Werk) ;

        // another iteration over M to determine MT->i and MT->x

        if (MTp32) {
            memmove(MTp32, MTp32 + 1, (M->vdim + 1) * sizeof(uint32_t)) ;
        }
        else {
            memmove(MTp64, MTp64 + 1, (M->vdim + 1) * sizeof(uint64_t));
        }

        centries = M->p_is_32 ? MTp32[M->vdim] : MTp64[M->vdim] ;
        uint32_t* MTi32 = NULL ; uint64_t* MTi64 = NULL;
        MTi32 = M->i_is_32 ? GB_calloc_memory (centries, sizeof(uint32_t), &allocated) : NULL ;
        MTi64 = M->i_is_32 ? NULL : GB_calloc_memory (centries, sizeof(uint64_t), &allocated) ;

        void* MTx = NULL ;
        if (!MTiso) {
            MTx = GB_calloc_memory (centries, op->ztype->size, &allocated) ;
        }
        else {
            GB_void a_elem[A->type->size] ;
            GB_void b_elem[B->type->size] ;

            cast_A (a_elem, A->x, A->type->size) ;
            cast_B (b_elem, B->x, B->type->size) ;

            MTx = GB_calloc_memory (1, op->ztype->size, &allocated) ;
            op->binop_function(MTx, a_elem, b_elem) ;
        }

        #pragma omp parallel
        {
            GrB_Index offset ;
            GB_void a_elem[op->xtype->size] ;
            GB_void b_elem[op->ytype->size] ;

            #pragma omp for
            for (GrB_Index k = 0 ; k < M->nvec ; k++)
            {
                GrB_Index j = Mh32 ? GBH (Mh32, k) : GBH (Mh64, k) ;
                // operate on column A(:,j)
                int64_t pA_start = Mp32 ? GBP (Mp32, k, vlen) : GBP(Mp64, k, vlen) ;
                int64_t pA_end   = Mp32 ? GBP (Mp32, k+1, vlen) : GBP(Mp64, k+1, vlen) ;
                GrB_Index pos = M->p_is_32 ? MTp32[k] : MTp64[k] ;
                for (GrB_Index p = pA_start ; p < pA_end ; p++)
                {
                    if (!GBB (M->b, p)) continue ;
                    // entry A(i,j) with row index i and value aij
                    int64_t i = Mi32 ? GBI (Mi32, p, vlen) : GBI (Mi64, p, vlen) ;
                    GrB_Index Mrow = M->is_csc ? i : j ; GrB_Index Mcol = M->is_csc ? j : i ;

                    // extract elements from A and B, initialize offset of MTi, get result of op, 
                    // place it in MTx
                    if (Mask_struct || (M->iso ? ((int8_t*)M->x)[0] : ((int8_t*)M->x)[p])) {

                        GrB_Index arow = A_transpose ? (Mcol / bncols) : (Mrow / bnrows);
                        GrB_Index acol = A_transpose ? (Mrow / bnrows) : (Mcol / bncols);

                        GrB_Index brow = B_transpose ? (Mcol % bncols) : (Mrow % bnrows);
                        GrB_Index bcol = B_transpose ? (Mrow % bnrows) : (Mcol % bncols);

                        bool code = GB_lookup_xoffset (&offset, A, arow, acol) ;
                        if (!code) {
                            continue;
                        }
                        cast_A (a_elem, A->x + offset * A->type->size, A->type->size) ;

                        code = GB_lookup_xoffset (&offset, B, brow, bcol) ;
                        if (!code) {
                            continue;
                        }
                        cast_B (b_elem, B->x + offset * B->type->size, B->type->size) ;

                        if (!MTiso) {
                            op->binop_function(MTx + op->ztype->size * pos, a_elem, b_elem) ;
                        }

                        if (M->i_is_32) { MTi32[pos] = i ; } else { MTi64[pos] = i ; }
                        pos++ ;
                    }
                }
            }
        }

        // initialize other fields of MT properly

        MT = NULL ;
        GB_OK (GB_new_bix (&MT, op->ztype, vlen, M->vdim, GB_ph_null, M->is_csc, 
        GxB_SPARSE, true, M->hyper_switch, M->vdim + 1, centries, true, false, 
        M->p_is_32, M->j_is_32, M->i_is_32)) ;

        GB_free_memory (&MT->i, MT->i_size) ;
        GB_free_memory (&MT->x, MT->x_size) ;


        MT->p = M->p_is_32 ? (void*)MTp32 : (void*)MTp64 ;
        MT->i = M->i_is_32 ? (void*)MTi32 : (void*)MTi64 ;
        MT->x = MTx ;
        MT->iso = MTiso ;

        MT->p_size = (M->p_is_32 ? sizeof(int32_t) : sizeof(int64_t)) * (M->vdim + 2) ;
        MT->i_size = centries ? ((M->i_is_32 ? sizeof(int32_t) : sizeof(int64_t)) * centries) : (M->i_is_32 ? sizeof(int32_t) : sizeof(int64_t)) ;
        MT->x_size = centries? op->ztype->size * centries : op->ztype->size ;
        MT->magic = GB_MAGIC ;
        MT->nvals = centries ;
        MT->nvec_nonempty = nvecs ;

        GB_MATRIX_WAIT(MT) ;

        // GB_hyper_prune and transpose and cast if needed

        if (MT->is_csc != C->is_csc) 
        {
            GB_transpose_in_place (MT, true, Werk) ;
        }

        if (MT_hypersparse) {
            uint32_t* MTh32 = NULL ; uint64_t* MTh64 =  NULL ;
            if (MT->j_is_32) {
                MTh32 = GB_calloc_memory (MT->vdim, sizeof(uint32_t), &allocated) ;
            }
            else {
                MTh64 = GB_calloc_memory (MT->vdim, sizeof(uint64_t), &allocated) ;
            }

            #pragma omp parallel for
            for (GrB_Index i = 0; i < MT->vdim; i++) {
                if (MT->j_is_32) { MTh32[i] = i ; } else { MTh64[i] = i ; } 
            }

            MT->h = MTh32 ? (void*)MTh32 : (void*)MTh64 ;

            GB_hyper_prune (MT, Werk) ;
        }

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

