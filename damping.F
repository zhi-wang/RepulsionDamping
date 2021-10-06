#define RE real*8
#define DAMPREP dampd
#include "fsrc.fh"
#undef RE
#undef DAMPREP

#define RE real*4
#define DAMPREP damps
#include "fsrc.fh"
#undef RE
#undef DAMPREP
