#include "matrix.h"
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

// Include SSE intrinsics
#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
#include <immintrin.h>
#include <x86intrin.h>
#endif

/* Below are some intel intrinsics that might be useful
 * void _mm256_storeu_pd (double * mem_addr, __m256d a)
 * __m256d _mm256_set1_pd (double a)
 * __m256d _mm256_set_pd (double e3, double e2, double e1, double e0)
 * __m256d _mm256_loadu_pd (double const * mem_addr)
 * __m256d _mm256_add_pd (__m256d a, __m256d b)
 * __m256d _mm256_sub_pd (__m256d a, __m256d b)
 * __m256d _mm256_fmadd_pd (__m256d a, __m256d b, __m256d c)
 * __m256d _mm256_mul_pd (__m256d a, __m256d b)
 * __m256d _mm256_cmp_pd (__m256d a, __m256d b, const int imm8)
 * __m256d _mm256_and_pd (__m256d a, __m256d b)
 * __m256d _mm256_max_pd (__m256d a, __m256d b)
*/

/* Generates a random double between low and high */
double rand_double(double low, double high) {
    double range = (high - low);
    double div = RAND_MAX / range;
    return low + (rand() / div);
}

/* Generates a random matrix */
void rand_matrix(matrix *result, unsigned int seed, double low, double high) {
    srand(seed);
    for (int i = 0; i < result->rows; i++) {
        for (int j = 0; j < result->cols; j++) {
            set(result, i, j, rand_double(low, high));
        }
    }
}

/*
 * Allocates space for a matrix struct pointed to by the double pointer mat with
 * `rows` rows and `cols` columns. You should also allocate memory for the data array
 * and initialize all entries to be zeros. `parent` should be set to NULL to indicate that
 * this matrix is not a slice. You should also set `ref_cnt` to 1.
 * You should return -1 if either `rows` or `cols` or both have invalid values. Return -2 if any
 * call to allocate memory in this function fails. Remember to set the error messages in numc.c.
 * Return 0 upon success.
 */
int allocate_matrix(matrix **mat, int rows, int cols) {
    /* TODO: YOUR CODE HERE */
    if(rows<=0 || cols <=0) return -1;
    *mat = malloc(sizeof(matrix) * 1);
    if(*mat == NULL) return -2;
    (*mat)->data = calloc(rows*cols, sizeof(double));
    if((*mat)->data == NULL) return -2;
    (*mat)->rows = rows;
    (*mat)->cols = cols;
    (*mat)->ref_cnt = 1;
    (*mat)->parent = NULL;
    return 0;
}

/*
 * Allocates space for a matrix struct pointed to by `mat` with `rows` rows and `cols` columns.
 * Its data should point to the `offset`th entry of `from`'s data (you do not need to allocate memory)
 * for the data field. `parent` should be set to `from` to indicate this matrix is a slice of `from`.
 * You should return -1 if either `rows` or `cols` or both have invalid values. Return -2 if any
 * call to allocate memory in this function fails.
 * Remember to set the error messages in numc.c.
 * Return 0 upon success.
 */
int allocate_matrix_ref(matrix **mat, matrix *from, int offset, int rows, int cols) {
    /* TODO: YOUR CODE HERE */
    if(rows<=0 || cols <=0) return -1;
    *mat = malloc(sizeof(matrix) * 1);
    if(*mat == NULL) return -2;
    (*mat)->data = from->data + offset;
    (*mat)->rows = rows;
    (*mat)->cols = cols;
    from->ref_cnt += 1;
    (*mat)->ref_cnt = -1; //this is a slice and you can't create another slice from this slice
    (*mat)->parent = from;
    return 0;
}

/*
 * You need to make sure that you only free `mat->data` if `mat` is not a slice and has no existing slices,
 * or that you free `mat->parent->data` if `mat` is the last existing slice of its parent matrix and its parent matrix has no other references
 * (including itself). You cannot assume that mat is not NULL.
 */
void deallocate_matrix(matrix *mat) {
    /* TODO: YOUR CODE HERE */
    if(mat != NULL){
        if(mat->ref_cnt==-1){
            if(mat->parent->ref_cnt == 1){
                free(mat->parent->data);
                free(mat->parent);
                free(mat);
            }else{
                mat->parent->ref_cnt -= 1;
                free(mat);
            }
        }else{
            if(mat->ref_cnt == 1){
                free(mat->data);
                free(mat);
            }else{
                mat->ref_cnt -= 1;
            }

        }

    //     if(mat->ref_cnt == 1){
    //         if(mat->parent == NULL){
    //             free(mat->data);
    //         } else { 
    //             if(mat->parent->ref_cnt == 1){
    //                 free(mat->parent->data);
    //                 free(mat->parent);
    //             } else{
    //                 mat->parent->ref_cnt -= 1;
    //             }
    //         }
    //         free(mat);
    //     }
    }
}

/*
 * Returns the double value of the matrix at the given row and column.
 * You may assume `row` and `col` are valid.
 */
double get(matrix *mat, int row, int col) {
    /* TODO: YOUR CODE HERE */
    return mat->data[(mat->cols * row) + col];
}

/*
 * Sets the value at the given row and column to val. You may assume `row` and
 * `col` are valid
 */
void set(matrix *mat, int row, int col, double val) {
    /* TODO: YOUR CODE HERE */
    mat->data[(mat->cols * row) + col] = val;
}

/*
 * Sets all entries in mat to val
 */
void fill_matrix(matrix *mat, double val) {
    /* TODO: YOUR CODE HERE */
    __m256d value = _mm256_set1_pd(val);
    int len = mat->rows * mat->cols;
    #pragma omp parallel for
    for(int i = 0; i<len/4*4; i+=4) {
        _mm256_storeu_pd(mat->data + i, value);
    }
    #pragma omp parallel for
    for(int i = len/4*4;i<len;i++){
        mat->data[i] = val;
    }
}

/*
 * Store the result of adding mat1 and mat2 to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 */
int add_matrix(matrix *result, matrix *mat1, matrix *mat2) {
    /* TODO: YOUR CODE HERE */
    if(mat1->rows!=mat2->rows || mat1->cols != mat2->cols || result->cols != mat1->cols || result->rows != mat2->rows) return -1;
    // for(int i = 0; i<mat1->rows; i++){
    //     for(int j = 0; j<mat1->cols; j++){
    //         result->data[(result->cols) * i + j] = mat1->data[(result->cols) * i + j] + mat2->data[(result->cols) * i + j];
    //     }
    // }
    // return 0;
    int len = mat1->rows * mat1->cols;
    omp_set_num_threads(8);
    #pragma omp parallel for 
        for(int i =0;i < len / 4 * 4; i+=4){
            __m256d _1 = _mm256_loadu_pd(mat1->data + i);
            __m256d _2 = _mm256_loadu_pd(mat2->data + i);
            __m256d _sum = _mm256_add_pd(_1,_2);
            _mm256_storeu_pd(result->data + i, _sum);
        }
    #pragma omp parallel for
    for(int i = len/4*4;i<len;i++){
        result->data[i] = mat1->data[i] + mat2->data[i];
    }
    return 0;

    
}

matrix* transpose(matrix *mat){
    matrix *ret = NULL;
    allocate_matrix(&ret, mat->cols, mat->rows);
    int len = mat->rows * mat->cols;
    int cols = mat->cols;
    int rows = mat->rows;
    #pragma omp parallel for
    for(int i = 0;i<len; i++){
        ret->data[i] = mat->data[(cols * (i%rows)) + (i/rows)]; //i%cols = col; i/cols = row
    }
    return ret;
}

/*
 * Store the result of multiplying mat1 and mat2 to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 * Remember that matrix multiplication is not the same as multiplying individual elements.
 */
int mul_matrix(matrix *result, matrix *mat1, matrix *mat2) {
    /* TODO: YOUR CODE HERE */
    if(mat1->cols!=mat2->rows || result->cols != mat2->cols || result->rows != mat1->rows) return -1;
    matrix *mat2_trans = transpose(mat2);

    omp_set_num_threads(8);
    // matrix *mat2_trans = NULL;
    // allocate_matrix(&mat2_trans, mat2->cols, mat2->rows);
    // int len = mat2->rows * mat2->cols;
    // int cols = mat2->cols;
    // int rows = mat2->rows;
    // #pragma omp parallel for
    // for(int i = 0;i<len; i++){
    //     mat2_trans->data[i] = mat2->data[(cols * (i%rows)) + (i/rows)]; //i%cols = col; i/cols = row
    // }

    int rows1 = mat1->rows;
    int cols1 = mat1->cols;
    int rows2 = mat2_trans->rows;
    int cols2 = mat2_trans->cols;
    #pragma omp parallel for
    for(int i = 0; i<rows1; i++){
        #pragma omp parallel for
        for(int j = 0; j<rows2; j++){
            __m256d sum = _mm256_set1_pd(0);
            for(int z = 0; z<cols2/4*4; z+=4){
                __m256d _mat1vals = _mm256_loadu_pd(mat1->data + (cols1*i) + z);
                __m256d _mat2vals = _mm256_loadu_pd(mat2_trans->data + (cols2*j) + z);
                sum = _mm256_add_pd(sum, _mm256_mul_pd(_mat1vals,_mat2vals));
            }
            double t[4];
            _mm256_storeu_pd(t, sum);
            double tsum = t[0]+t[1]+t[2]+t[3];
            //#pragma omp parallel for reduction(+:tsum)
                for(int z = cols2/4*4; z<cols2; z+=1){
                    tsum += mat1->data[cols1 * i + z] * mat2_trans->data[cols2 * j + z];
                }
            // double tsum = 0;
            // for(int z = 0; z<mat2_trans->cols; z++){
            //     tsum += mat1->data[(mat1->cols) * i + z] * mat2_trans->data[(mat2_trans->cols) * j + z];
            // }
            result->data[(result->cols) * i + j] = tsum;
        }
    }
    //deallocate_matrix(mat2_trans);
    free(mat2_trans->data);
    free(mat2_trans);
    return 0;
}



/*
 * Store the result of raising mat to the (pow)th power to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 * Remember that pow is defined with matrix multiplication, not element-wise multiplication.
 */
int pow_matrix(matrix *result, matrix *mat, int pow) {
    /* TODO: YOUR CODE HERE */
    if(mat->rows != mat->cols || result->rows != result->cols || result->rows != mat->rows) return -1;
    omp_set_num_threads(8);
    fill_matrix(result, 0);
    int cols = result->cols;
    int rows = result->rows;
    #pragma omp parallel for
    for(int i = 0; i<rows; i++){
        result->data[cols * i + i] = 1;
        //set(result,i,i,1);
    }
    if(pow>0){
        matrix *temp = NULL;
        allocate_matrix(&temp, mat->rows, mat->rows);
        int k = 1;
        int s = 1;
        mul_matrix(temp, result, mat);
        // #pragma omp parallel for
        // for(int i = 0; i<(rows*cols)/4*4;i+=4){
        //     _mm256_storeu_pd(temp->data + i, _mm256_loadu_pd(mat->data + i));
        // }
        // #pragma omp parallel for
        // for(int i = (rows*cols)/4*4; i<rows*cols; i++){
        //     temp->data[i] = mat->data[i];
        // }
        while(k<pow){
            if( k+k <= pow){
                if(s){
                    mul_matrix(result,temp,temp);
                }else{
                    mul_matrix(temp,result,result);
                }
                k += k;
            }else{
                if(s){
                    mul_matrix(result,temp,mat);
                }else{
                    mul_matrix(temp,result,mat);
                }
                k += 1;
            }
            s = !s;
        }
        // for(int i = 0; i<pow; i++){
        //     if(i%2 == 0){
        //         mul_matrix(temp, result, mat);
        //     } else{
        //         mul_matrix(result, temp, mat);
        //     }
        // }
        if(s){
            #pragma omp parallel for
            for(int i = 0; i<(rows*cols)/4*4;i+=4){
                _mm256_storeu_pd(result->data + i, _mm256_loadu_pd(temp->data + i));
            }
            #pragma omp parallel for
            for(int i = (rows*cols)/4*4; i<rows*cols; i++){
                result->data[i] = temp->data[i];
            }
        }
        deallocate_matrix(temp);
    }
    return 0;
}



/*
 * Store the result of taking the absolute value element-wise to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 */
int abs_matrix(matrix *result, matrix *mat) {
    if(mat->cols != result->cols || result->rows != mat->rows) return -1;
    // for(int i = 0; i<mat->rows; i++){
    //     for(int j = 0; j<mat->cols; j++){
    //         int val = mat->data[(result->cols) * i + j];
    //         result->data[(result->cols) * i + j] = (val >= 0 ? val : -1*val)    ;
    //     }
    // }
    // return 0;
    int len = mat->rows * mat->cols;
    __m256d _neg = _mm256_set1_pd(-1);
    #pragma omp parallel for
    for(int i = 0;i < len / 4 * 4; i+=4){
        __m256d _vals = _mm256_loadu_pd(mat->data + i);
        __m256d _negated = _mm256_mul_pd(_vals, _neg);
        _mm256_storeu_pd(result->data + i, _mm256_max_pd(_negated, _vals));
    }

    #pragma omp parallel for
    for(int i = len / 4 * 4;i < len; i++){
            double val = mat->data[i];
            result->data[i] = (val >= 0 ? val : -1*val);
    }
    return 0;
}





/*
 * (OPTIONAL)
 * Store the result of element-wise negating mat's entries to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 */
int neg_matrix(matrix *result, matrix *mat) {
    // if(mat->rows != mat->cols || result->rows != result->cols || result->rows != mat->rows) return -1;
    // for(int i = 0; i<mat->rows; i++){
    //     for(int j = 0; j<mat->cols; j++){
    //         result->data[(result->cols) * i + j] = -1 * (mat->data[(result->cols) * i + j]);
    //     }
    // }
    return 0;
}

/*
 * (OPTIONAL)
 * Store the result of subtracting mat2 from mat1 to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 */
int sub_matrix(matrix *result, matrix *mat1, matrix *mat2) {
    /* TODO: YOUR CODE HERE */
    // if(mat1==NULL || mat2==NULL || result==NULL) return -1;
    // if(mat1->rows!=mat2->rows || mat1->cols != mat2->cols || result->cols != mat1->cols || result->rows != mat2->rows) return -1;
    // for(int i = 0; i<mat1->rows; i++){
    //     for(int j = 0; j<mat1->cols; j++){
    //         result->data[(result->cols) * i + j] = mat1->data[(result->cols) * i + j] - mat2->data[(result->cols) * i + j];
    //     }
    // }
    return 0;
}
