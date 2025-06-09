#ifndef DEFINES_H_
#define DEFINES_H_

#include "ap_int.h"
#include "ap_fixed.h"
#include "nnet_utils/nnet_types.h"
#include <cstddef>
#include <cstdio>

//hls-fpga-machine-learning insert numbers
#define N_INPUT_1_1 16
#define N_LAYER_2 58
#define N_LAYER_5 3

//hls-fpga-machine-learning insert layer-precision
typedef ap_fixed<16,6> input_t;
typedef ap_fixed<24,11> layer2_accum_t;
typedef ap_fixed<24,11> layer2_t;
typedef ap_fixed<4,1> weight2_t;
typedef ap_fixed<4,1> bias2_t;
typedef ap_uint<1> layer2_index;
typedef ap_ufixed<8,0,AP_RND,AP_SAT> layer4_t;
typedef ap_fixed<18,8> relu1_table_t;
typedef ap_fixed<19,8> layer5_accum_t;
typedef ap_fixed<19,8> result_t;
typedef ap_fixed<4,1> weight5_t;
typedef ap_fixed<4,1> bias5_t;
typedef ap_uint<1> layer5_index;

#endif
