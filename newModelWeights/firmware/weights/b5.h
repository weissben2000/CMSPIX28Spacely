//Numpy array shape [3]
//Min -0.375000000000
//Max 0.375000000000
//Number of zeros 1

#ifndef B5_H_
#define B5_H_

#ifndef __SYNTHESIS__
bias5_t b5[3];
#else
bias5_t b5[3] = {0.000, -0.375, 0.375};
#endif

#endif
