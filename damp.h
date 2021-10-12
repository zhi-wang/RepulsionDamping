#pragma once

/* This is a C header. */
#ifdef __cplusplus
extern "C"
{
#endif

   /* Fortran real*8 damprep() */
   void dampd_(double* r, double* r2, double* rr1, double* rr3, double* rr5,
               double* rr7, double* rr9, double* rr11, int* rorder,
               double* dmpi, double* dmpk, double* dmpik);

   /* Fortran real*4 damprep() */
   void damps_(float* r, float* r2, float* rr1, float* rr3, float* rr5,
               float* rr7, float* rr9, float* rr11, int* rorder, float* dmpi,
               float* dmpk, float* dmpik);

   /* Fortran real*8 Gordon1 */
   void dampg1d_(double* r, int* rorder, double* dmpi, double* dmpk,
                 double* dmpik);

   /* Fortran real*4 Gordon1 */
   void dampg1s_(float* r, int* rorder, float* dmpi, float* dmpk, float* dmpik);

   /* Run tests in double precision. */
   void rund(char c, double arr[3]);

   /* Run tests in single precision. */
   void runs(char c, float arr[3]);

#ifdef __cplusplus
}
#endif
