#include "damp.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

static void drive(double darr[3])
{
   float sarr[3] = {darr[0], darr[1], darr[2]};
   printf("(r,ai,aj)       %16.8lf%16.4lf%16.4lf\n", darr[0], darr[1], darr[2]);
   printf("%s\n", "SINGLE");
   runs(sarr);
   printf("%s\n", "DOUBLE");
   rund(darr);
   printf("%s\n\n", "------------");
}

#define FILENAME "input.txt"
#define DEFAULT_DATA                                                           \
   "4.2680   4.3189   8.5908271422   -0.5636998917   -0.8034106213   "         \
   "7.3384501040   -0.5663455549   0.5436405963"
#define LINE_LEN 2048

int main(int argc, char** argv)
{
   char* filename = NULL;
   if (argc == 1) {
      if (access(FILENAME, F_OK) != 0) {
         FILE* fileHandle = fopen(FILENAME, "w");
         fprintf(fileHandle, "%s\n", DEFAULT_DATA);
         fclose(fileHandle);
      }
      filename = FILENAME;
   } else {
      filename = argv[1];
   }

   FILE* inputFile = fopen(filename, "r");
   char line[LINE_LEN];
   while (fgets(line, LINE_LEN, inputFile)) {
      double p[8] = {0}, dist = 0;
      double darr[3] = {0};
      int cnt;
      cnt = sscanf(line, "%lf%lf%lf%lf%lf%lf%lf%lf", p, p + 1, p + 2, p + 3,
                   p + 4, p + 5, p + 6, p + 7);
      if (cnt == 8 || cnt == 3) {
         if (cnt == 8) {
            double s[3] = {p[2] - p[5], p[3] - p[6], p[4] - p[7]};
            dist = sqrt(s[0] * s[0] + s[1] * s[1] + s[2] * s[2]);
         } else {
            dist = p[2];
         }
         darr[0] = dist, darr[1] = p[0], darr[2] = p[1];
         drive(darr);
      }
   }
   fclose(inputFile);
   return 0;
}
