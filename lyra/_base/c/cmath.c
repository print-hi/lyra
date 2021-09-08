#include "cmath.h"

int fastfactorial(int n){
 if(n<=1)
 return 1;
 else
 return n * fastfactorial(n-1);
}