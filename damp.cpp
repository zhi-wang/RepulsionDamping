#include "damp.h"
#include <cmath>
#include <cstdio>

using std::exp;
using std::fabs;

/* ------ Tinker9 macros ------ */

#define REAL_ABS fabs
#define REAL_EXP exp
#define CONSTEXPR
#define restrict __restrict__
#define SEQ_CUDA

namespace {

/* ------ Helper functions ------ */

#pragma acc routine seq
template <int N, class real>
SEQ_CUDA
inline real fsinhc_analyt(real d, real d2, real d3, real d4,
                          real y /* exp(-d) */, real z /* exp(+d) */)
{
   real cy, cz;
   if CONSTEXPR (N == 7) {
      cy = d * (d * (d * (d * (d * (d + 21) + 210) + 1260) + 4725) + 10395) +
         10395;
      cy = -cy;
      cz = d * (d * (d * (d * (d * (d - 21) + 210) - 1260) + 4725) - 10395) +
         10395;
      real d13 = d3 * d3 * d3 * d4;
      return (cy * y + cz * z) / (2 * d13);
   } else if CONSTEXPR (N == 6) {
      cy = d * (d * (d * (d * (d + 15) + 105) + 420) + 945) + 945;
      cz = d * (d * (d * (d * (d - 15) + 105) - 420) + 945) - 945;
      real d11 = d3 * d4 * d4;
      return (cy * y + cz * z) / (2 * d11);
   } else if CONSTEXPR (N == 5) {
      cy = d * (d * (d * (d + 10) + 45) + 105) + 105;
      cy = -cy;
      cz = d * (d * (d * (d - 10) + 45) - 105) + 105;
      real d9 = d3 * d3 * d3;
      return (cy * y + cz * z) / (2 * d9);
   } else if CONSTEXPR (N == 4) {
      cy = d * (d * (d + 6) + 15) + 15;
      cz = d * (d * (d - 6) + 15) - 15;
      real d7 = d3 * d4;
      return (cy * y + cz * z) / (2 * d7);
   } else if CONSTEXPR (N == 3) {
      cy = d2 + 3 * d + 3;
      cy = -cy;
      cz = d2 - 3 * d + 3;
      real d5 = d2 * d3;
      return (cy * y + cz * z) / (2 * d5);
   } else if CONSTEXPR (N == 2) {
      cy = d + 1;
      cz = d - 1;
      return (cy * y + cz * z) / (2 * d3);
   } else /* if CONSTEXPR (N == 1) */ {
      cy = -1;
      cz = 1;
      return (cy * y + cz * z) / (2 * d);
   }
}

#pragma acc routine seq
template <int N, class real>
SEQ_CUDA
inline real fsinhc_taylor(real x2)
{
   constexpr real c[][5] = {
      {1 / 1., 1 / 6., 1 / 20., 1 / 42., 1 / 72.},        // 1
      {1 / 3., 1 / 10., 1 / 28., 1 / 54., 1 / 88.},       // 2
      {1 / 15., 1 / 14., 1 / 36., 1 / 66., 1 / 104.},     // 3
      {1 / 105., 1 / 18., 1 / 44., 1 / 78., 1 / 120.},    // 4
      {1 / 945., 1 / 22., 1 / 52., 1 / 90., 1 / 136.},    // 5
      {1 / 10395., 1 / 26., 1 / 60., 1 / 102., 1 / 152.}, // 6
      {1 / 135135., 1 / 30., 1 / 68., 1 / 114., 1 / 168.} // 7
   };
   constexpr int M = N - 1;
   // clang-format off
   return c[M][0]*(1+x2*c[M][1]*(1+x2*c[M][2]*(1+x2*c[M][3]*(1+x2*c[M][4]))));
   // clang-format on
}

//*
#pragma acc routine seq
template <int N, class real>
SEQ_CUDA
inline real fsinhc_pade44(real x2, real x4)
{
   // Pade coefficients
   constexpr real c[][2][3] = {
      {
         {1., 53. / 396, 551. / 166320}, // 1
         {1., -13. / 396, 5. / 11088}    // 1
      },
      {
         {1. / 3, 211. / 8580, 2647. / 6486480}, // 2
         {1., -15. / 572, 17. / 61776}           // 2
      },
      {
         {1. / 15, 271. / 81900, 7. / 171600}, // 3
         {1., -17. / 780, 19. / 102960}        // 3
      },
      {
         {1. / 105, 113. / 321300, 1889. / 551350800}, // 4
         {1., -19. / 1020, 7 / 53040}                  // 4
      },
      {
         {1. / 945, 83. / 2686068, 7789. / 31426995600}, // 5
         {1., -21. / 1292, 23. / 232560}                 // 5
      },
      {
         {1. / 10395, 499. / 215675460, 3461. / 219988969200}, // 6
         {1., -23. / 1596, 25. / 325584}                       // 6
      },
      {
         {1. / 135135, 197. / 1305404100, 409. / 459976935600}, // 7
         {1., -25. / 1932, 3. / 48944}                          // 7
      }};
   constexpr int M = N - 1;
   return (c[M][0][0] + x2 * c[M][0][1] + x4 * c[M][0][2]) /
      (c[M][1][0] + x2 * c[M][1][1] + x4 * c[M][1][2]);
}
// */

#pragma acc routine seq
template <int N, class real>
SEQ_CUDA
inline real fsinhc(real d, real d2, real d3, real d4, real expmd /* exp(-d) */,
                   real exppd /* exp(+d) */)
{
   constexpr int M = N - 1;
   /**
    * (x, Approx. |Analyt - Taylor|)
    *
    * f1   (0.90, 1e-8)   (0.35, 1e-12)   (0.28, 1e-13)
    * f2   (1.15, 1e-8)   (0.45, 1e-12)   (0.37, 1e-13)
    * f3   (1.5,  1e-8)   (0.6,  1e-12)   (0.48, 1e-13)
    * f4   (2.0,  1e-8)   (0.8,  1e-12)   (0.64, 1e-13)
    * f5   (2.7,  1e-8)   (1.1,  1e-12)   (0.87, 1e-13)
    * f6   (3.7,  1e-8)   (1.5,  1e-12)   (1.16, 1e-13)
    * f7   (5.0,  1e-8)   (2.0,  1e-12)   (1.60, 1e-13)
    */
   // double epsd[] = {0.35, 0.45, 0.6, 0.8, 1.1, 1.5, 2.0};
   double epsd[] = {0.28, 0.37, 0.48, 0.64, 0.87, 1.16, 1.60};
   float epsf[] = {0.9, 1.15, 1.5, 2.0, 2.7, 3.7, 5.0};
   real absd = REAL_ABS(d), eps;
   if CONSTEXPR (sizeof(real) == sizeof(double))
      eps = epsd[M];
   else
      eps = epsf[M];
   if CONSTEXPR (N == 7) {
      if (absd > eps)
         return fsinhc_analyt<7>(d, d2, d3, d4, expmd, exppd);
      else
         return fsinhc_taylor<7>(d2);
   } else if CONSTEXPR (N == 6) {
      if (absd > eps)
         return fsinhc_analyt<6>(d, d2, d3, d4, expmd, exppd);
      else
         return fsinhc_taylor<6>(d2);
   } else if CONSTEXPR (N == 5) {
      if (absd > eps)
         return fsinhc_analyt<5>(d, d2, d3, d4, expmd, exppd);
      else
         return fsinhc_taylor<5>(d2);
   } else if CONSTEXPR (N == 4) {
      if (absd > eps)
         return fsinhc_analyt<4>(d, d2, d3, d4, expmd, exppd);
      else
         return fsinhc_taylor<4>(d2);
   } else if CONSTEXPR (N == 3) {
      if (absd > eps)
         return fsinhc_analyt<3>(d, d2, d3, d4, expmd, exppd);
      else
         return fsinhc_taylor<3>(d2);
   } else if CONSTEXPR (N == 2) {
      if (absd > eps)
         return fsinhc_analyt<2>(d, d2, d3, d4, expmd, exppd);
      else
         return fsinhc_taylor<2>(d2);
   } else /* if CONSTEXPR (N == 1) */ {
      if (absd > eps)
         return fsinhc_analyt<1>(d, d2, d3, d4, expmd, exppd);
      else
         return fsinhc_taylor<1>(d2);
   }
}

/* ------ New Implementation ------ */

#pragma acc routine seq
template <int order, class real>
SEQ_CUDA
inline void damp_rep(real* restrict dmpik, real r, real rr1, real r2, real rr3,
                     real rr5, real rr7, real rr9, real rr11, real ai, real aj)
{
   real pfac = 2 / (ai + aj);
   pfac = pfac * pfac;
   pfac = pfac * ai * aj;
   pfac = pfac * pfac * pfac;
   pfac *= r2;

   real a = ai * r / 2, b = aj * r / 2;
   real c = (a + b) / 2, d = (b - a) / 2;
   real expmc = REAL_EXP(-c);
   real expmd = REAL_EXP(-d);
   real exppd = REAL_EXP(d);

   real c2 = c * c;
   real c3 = c2 * c;
   real c4 = c2 * c2;
   real d2 = d * d;
   real d3 = d2 * d;
   real d4 = d2 * d2;
   real c2d2 = (c * d) * (c * d);

   real f1d, f2d, f3d, f4d, f5d, f6d, f7d;
   f1d = fsinhc<1>(d, d2, d3, d4, expmd, exppd);
   f2d = fsinhc<2>(d, d2, d3, d4, expmd, exppd);
   f3d = fsinhc<3>(d, d2, d3, d4, expmd, exppd);
   f4d = fsinhc<4>(d, d2, d3, d4, expmd, exppd);
   f5d = fsinhc<5>(d, d2, d3, d4, expmd, exppd);
   f6d = fsinhc<6>(d, d2, d3, d4, expmd, exppd);
   if CONSTEXPR (order > 9)
      f7d = fsinhc<7>(d, d2, d3, d4, expmd, exppd);

   real inv3 = 1. / 3, inv15 = 1. / 15, inv105 = 1. / 105, inv945 = 1. / 945;

   // compute
   // clang-format off
   real s;
   s = f1d * (c+1)
     + f2d * c2;
   s *= rr1;
   s *= expmc;
   dmpik[0] = pfac * s * s;

   real ds;
   ds = f1d * c2
      + f2d * ((c-2)*c2 - (c+1)*d2)
      - f3d * c2d2;
   ds *= rr3;
   ds *= expmc;
   dmpik[1] = pfac * 2 * s * ds;

   real d2s = 0;
   d2s += f1d * c3
        + f2d * c2*((c-3)*c - 2*d2);
   d2s += d2*(f3d * (2*(2-c)*c2 + (c+1)*d2)
            + f4d * c2d2);
   d2s *= rr5 * inv3;
   d2s *= expmc;
   dmpik[2] = pfac * 2 * (s * d2s + ds * ds);

   real d3s = 0;
   d3s += f1d * c3*(c+1)
        + f2d * c3*(c*(c-3) - 3*(d2+1));
   d3s -= d2*(f3d * 3*c2*((c-3)*c - d2)
         + d2*(f4d * (3*(2-c)*c2 + (c+1)*d2)
             + f5d * c2d2));
   d3s *= rr7 * inv15;
   d3s *= expmc;
   dmpik[3] = pfac * 2 * (s * d3s + 3 * ds * d2s);

   real d4s = 0;
   d4s += f1d * c3*(3 + c*(c+3))
        + f2d * c3*(c3 - 9*(c+1) - 2*c2 - 4*(c+1)*d2);
   d4s += d2*(f3d * 2*c3*(6*(c+1) - 2*c2 + 3*d2)
            + d2*(f4d * 2*c2*(3*(c-3)*c - 2*d2)
                + f5d * d2*(4*(2-c)*c2 + (c+1)*d2)
                + f6d * c2*d4));
   d4s *= rr9 * inv105;
   d4s *= expmc;
   dmpik[4] = pfac * 2 * (s * d4s + 4 * ds * d3s + 3 * d2s * d2s);

   if CONSTEXPR (order > 9) {
      real d5s = 0;
      d5s += f1d * c3*(15 + c*(15 + c*(c+6)));
      d5s += f2d * c3*(c4 - 15*c2 - 45*(c+1) - 5*(3+c*(c+3))*d2);
      d5s -= d2*(f3d * 5*c3*(c3 - 9*(c+1) - 2*c2 - 2*(c+1)*d2)
               + d2*(f4d * 10*c3*(3 - (c-3)*c + d2)
                   + d2*(f5d * 5*c2*(2*(c-3)*c - d2)
                       + f6d * d2*((c+1)*d2 - 5*(c-2)*c2)
                       + f7d * c2*d4)));
      d5s *= rr11 * inv945;
      d5s *= expmc;
      dmpik[5] = pfac * 2 * (s * d5s + 5 * ds * d4s + 10 * d2s * d3s);
   }
   // clang-format on
}

#pragma acc routine seq
template <int order, class real>
SEQ_CUDA
inline void damp_gordon1(real* restrict dmpik, real r, real ai, real aj)
{
   real a = ai * r, b = aj * r;
   real c = (b + a) / 2, d = (b - a) / 2;
   real expmc = REAL_EXP(-c);
   real expmd = REAL_EXP(-d);
   real exppd = REAL_EXP(d);

   real t = (ai + aj) * r;
   real x = a / t, y = 1 - x;

   real c2 = c * c, c3 = c * c * c;
   real d2 = d * d, d3 = d * d * d;
   real d4 = d2 * d2;
   real c2d2 = c2 * d2;

   real f1d, f2d, f3d, f4d, f5d, f6d, f7d;
   f1d = fsinhc<1>(d, d2, d3, d4, expmd, exppd);
   f2d = fsinhc<2>(d, d2, d3, d4, expmd, exppd);
   f3d = fsinhc<3>(d, d2, d3, d4, expmd, exppd);
   f4d = fsinhc<4>(d, d2, d3, d4, expmd, exppd);
   f5d = fsinhc<5>(d, d2, d3, d4, expmd, exppd);
   f6d = fsinhc<6>(d, d2, d3, d4, expmd, exppd);
   if CONSTEXPR (order > 9)
      f7d = fsinhc<7>(d, d2, d3, d4, expmd, exppd);

   real iC2 = 1. / 3, iC3 = 1. / 15, iC4 = 1. / 105, iC5 = 1. / 945;
   real ec = expmc / 16, ea = REAL_EXP(-a) / 32, eb = REAL_EXP(-b) / 32;

#define TINKER_GORDON1_L00(X) (4 * ((X) * ((X) * (2 * (X)-3) - 3) + 6))
#define TINKER_GORDON1_L01(X) ((X) * (4 * ((X)-3) * (X) + 11) - 2)
#define TINKER_GORDON1_M0(a)  (1)
#define TINKER_GORDON1_M1(a)  ((a) + 1)
#define TINKER_GORDON1_M2(a)  ((a) * ((a) + 3) + 3)
#define TINKER_GORDON1_M3(a)  ((a) * ((a) * ((a) + 6) + 15) + 15)
#define TINKER_GORDON1_M4(a)  ((a) * ((a) * ((a) * ((a) + 10) + 45) + 105) + 105)
#define TINKER_GORDON1_M5(a)                                                   \
   ((a) * ((a) * ((a) * ((a) * ((a) + 15) + 105) + 420) + 945) + 945)

   // [0]
   real k01, k02, l00x, l01x, l00y, l01y;
   k01 = 3 * c * (c + 3);
   k02 = c3;
   l00x = TINKER_GORDON1_L00(x), l01x = TINKER_GORDON1_L01(x) * t;
   l00y = TINKER_GORDON1_L00(y), l01y = TINKER_GORDON1_L01(y) * t;
   dmpik[0] = 1 -
      ((k01 * f1d + k02 * f2d) * ec + (l00x + l01x) * ea + (l00y + l01y) * eb);

   // [1]
   real k11, k12, k13, l10x, l11x, l10y, l11y;
   k11 = 3 * c2 * (c + 2);
   k12 = c * ((c - 2) * c2 - 3 * (c + 3) * d2);
   k13 = -c3 * d2;
   l10x = TINKER_GORDON1_M1(a) * l00x, l11x = a * TINKER_GORDON1_M0(a) * l01x;
   l10y = TINKER_GORDON1_M1(b) * l00y, l11y = b * TINKER_GORDON1_M0(b) * l01y;
   dmpik[1] = 1 -
      ((k11 * f1d + k12 * f2d + k13 * f3d) * ec + (l10x + l11x) * ea +
       (l10y + l11y) * eb);


   // [2]
   real k21, k22, k23, k24, l20x, l21x, l20y, l21y;
   k21 = 3 * c2 * (c * (c + 2) + 2);
   k22 = c2 * ((c - 3) * c2 - 6 * (c + 2) * d2);
   k23 = 3 * (c + 3) * d2 - 2 * (c - 2) * c2;
   k24 = c2d2;
   l20x = TINKER_GORDON1_M2(a) * l00x, l21x = a * TINKER_GORDON1_M1(a) * l01x;
   l20y = TINKER_GORDON1_M2(b) * l00y, l21y = b * TINKER_GORDON1_M1(b) * l01y;
   dmpik[2] = 1 -
      iC2 *
         ((k21 * f1d + k22 * f2d + c * d2 * (k23 * f3d + k24 * f4d)) * ec +
          (l20x + l21x) * ea + (l20y + l21y) * eb);

   // [3]
   real k31, k32, k33, k34, k35, l30x, l31x, l30y, l31y;
   k31 = 3 * c2 * (c * (c * (c + 3) + 6) + 6);
   k32 = c2 * (c2 * ((c - 3) * c - 3) - 9 * (c * (c + 2) + 2) * d2);
   k33 = c2 * (9 * (c + 2) * d2 - 3 * (c - 3) * c2);
   k34 = 3 * (c - 2) * c3 - 3 * c * (c + 3) * d2;
   k35 = -c3 * d2;
   l30x = TINKER_GORDON1_M3(a) * l00x, l31x = a * TINKER_GORDON1_M2(a) * l01x;
   l30y = TINKER_GORDON1_M3(b) * l00y, l31y = b * TINKER_GORDON1_M2(b) * l01y;
   dmpik[3] = 1 -
      iC3 *
         ((k31 * f1d + k32 * f2d +
           d2 * (k33 * f3d + d2 * (k34 * f4d + k35 * f5d))) *
             ec +
          (l30x + l31x) * ea + (l30y + l31y) * eb);

   // [4]
   if CONSTEXPR (order > 7) {
      real k41, k42, k43, k44, k45, k46, l40x, l41x, l40y, l41y;
      k41 = 3 * c2 * (c * (c * (c * (c + 5) + 15) + 30) + 30);
      k42 = c2 *
         (c2 * (c * ((c - 2) * c - 9) - 9) -
          12 * (c * (c * (c + 3) + 6) + 6) * d2);
      k43 = c2 * (18 * (c * (c + 2) + 2) * d2 - 4 * c2 * ((c - 3) * c - 3));
      k44 = c2 * (6 * (c - 3) * c2 - 12 * (c + 2) * d2);
      k45 = c * (3 * (c + 3) * d2 - 4 * (c - 2) * c2);
      k46 = c3 * d2;
      l40x = TINKER_GORDON1_M4(a) * l00x,
      l41x = a * TINKER_GORDON1_M3(a) * l01x;
      l40y = TINKER_GORDON1_M4(b) * l00y,
      l41y = b * TINKER_GORDON1_M3(b) * l01y;
      dmpik[4] = 1 -
         iC4 *
            ((k41 * f1d + k42 * f2d +
              d2 *
                 (k43 * f3d +
                  d2 * (k44 * f4d + d2 * (k45 * f5d + k46 * f6d)))) *
                ec +
             (l40x + l41x) * ea + (l40y + l41y) * eb);
   }

   // [5]
   if CONSTEXPR (order > 9) {
      real k51 = 0, k52 = 0, k53 = 0, k54 = 0, k55 = 0, k56 = 0, k57 = 0, l50x,
           l51x, l50y, l51y;
      k51 = 3 * c2 * (c * (c * (c * (c * (c + 8) + 35) + 105) + 210) + 210);
      k52 = c2 *
         (c2 * (c * (c3 - 15 * c - 45) - 45) -
          15 * (c * (c * (c * (c + 5) + 15) + 30) + 30) * d2);
      k53 = c2 *
         (5 * (c * (9 - (c - 2) * c) + 9) * c2 +
          30 * (c * (c * (c + 3) + 6) + 6) * d2);
      k54 = c2 * (10 * c2 * ((c - 3) * c - 3) - 30 * (c * (c + 2) + 2) * d2);
      k55 = c2 * (15 * (c + 2) * d2 - 10 * (c - 3) * c2);
      k56 = c * (5 * (c - 2) * c2 - 3 * (c + 3) * d2);
      k57 = -c3 * d2;
      l50x = TINKER_GORDON1_M5(a) * l00x,
      l51x = a * TINKER_GORDON1_M4(a) * l01x;
      l50y = TINKER_GORDON1_M5(b) * l00y,
      l51y = b * TINKER_GORDON1_M4(b) * l01y;
      dmpik[5] = 1 -
         iC5 *
            ((k51 * f1d + k52 * f2d +
              d2 *
                 (k53 * f3d +
                  d2 *
                     (k54 * f4d +
                      d2 * (k55 * f5d + d2 * (k56 * f6d + k57 * f7d))))) *
                ec +
             (l50x + l51x) * ea + (l50y + l51y) * eb);
   }

#undef TINKER_GORDON1_L00
#undef TINKER_GORDON1_L01
#undef TINKER_GORDON1_M0
#undef TINKER_GORDON1_M1
#undef TINKER_GORDON1_M2
#undef TINKER_GORDON1_M3
#undef TINKER_GORDON1_M4
#undef TINKER_GORDON1_M5
}

/* ------ Output ------ */

template <class RE>
void prtdmp(RE* dmpik)
{
   for (int i = 0; i < 6; ++i) {
      int j = 2 * i;
      printf("%16.8lf", (double)dmpik[j]);
   }
   printf("\n");
}

template <class RE>
void run(char c, RE arr[3])
{
   int rorder = 11;
   RE r = arr[0], dmpi = arr[1], dmpk = arr[2];
   RE r2, rr1, rr3, rr5, rr7, rr9, rr11, dmpik[11];
   r2 = r * r;
   rr1 = 1 / r;
   rr3 = rr1 / r2;
   rr5 = 3 * rr3 / r2;
   rr7 = 5 * rr5 / r2;
   rr9 = 7 * rr7 / r2;
   rr11 = 9 * rr9 / r2;

   printf("%s\n", "Current Impl");
   // clean
   for (int i = 0; i < 11; ++i)
      dmpik[i] = 0;
   if (c == 'R' or c == 'r') {
      if (sizeof(RE) == sizeof(double)) {
         dampd_((double*)&r, (double*)&r2, (double*)&rr1, (double*)&rr3,
                (double*)&rr5, (double*)&rr7, (double*)&rr9, (double*)&rr11,
                &rorder, (double*)&dmpi, (double*)&dmpk, (double*)dmpik);
      } else if (sizeof(RE) == sizeof(float)) {
         damps_((float*)&r, (float*)&r2, (float*)&rr1, (float*)&rr3,
                (float*)&rr5, (float*)&rr7, (float*)&rr9, (float*)&rr11,
                &rorder, (float*)&dmpi, (float*)&dmpk, (float*)dmpik);
      }
   } else if (c == 'G' or c == 'g') {
      if (sizeof(RE) == sizeof(double)) {
         dampg1d_((double*)&r, &rorder, (double*)&dmpi, (double*)&dmpk,
                  (double*)dmpik);
      } else if (sizeof(RE) == sizeof(float)) {
         dampg1s_((float*)&r, &rorder, (float*)&dmpi, (float*)&dmpk,
                  (float*)dmpik);
      }
   }
   prtdmp(dmpik);
   printf("%s\n", "New Impl");
   // clean
   for (int i = 0; i < 11; ++i)
      dmpik[i] = 0;
   if (c == 'R' or c == 'r') {
      if (rorder == 9)
         damp_rep<9>(dmpik, r, rr1, r2, rr3, rr5, rr7, rr9, rr11, dmpi, dmpk);
      else if (rorder == 11)
         damp_rep<11>(dmpik, r, rr1, r2, rr3, rr5, rr7, rr9, rr11, dmpi, dmpk);
   } else if (c == 'G' or c == 'g') {
      if (rorder == 9)
         damp_gordon1<9>(dmpik, r, dmpi, dmpk);
      else if (rorder == 11)
         damp_gordon1<11>(dmpik, r, dmpi, dmpk);
   }
   // copy
   dmpik[2 * 5] = dmpik[5];
   dmpik[2 * 4] = dmpik[4];
   dmpik[2 * 3] = dmpik[3];
   dmpik[2 * 2] = dmpik[2];
   dmpik[2 * 1] = dmpik[1];
   prtdmp(dmpik);
}
}

extern "C"
{
   void rund(char c, double a[3])
   {
      run<double>(c, a);
   }

   void runs(char c, float a[3])
   {
      run<float>(c, a);
   }
}
