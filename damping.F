#define RE real*8
#define FLT_8

#define DAMPREP dampd
#include "fsrc.fh"
#undef DAMPREP

#define DAMPG1 dampg1d
#define DAMPPOLE damppoled
#include "fgordon.fh"
#undef DAMPG1
#undef DAMPPOLE

#undef FLT_8
#undef RE



#define RE real*4
#define FLT_4

#define DAMPREP damps
#include "fsrc.fh"
#undef DAMPREP

#define DAMPG1 dampg1s
#define DAMPPOLE damppoles
#include "fgordon.fh"
#undef DAMPG1
#undef DAMPPOLE

#undef FLT_4
#undef RE
