#ifndef CXXNET_LAYER_CLSTM_LAYER_INL_HPP_
#define CXXNET_LAYER_CLSTM_LAYER_INL_HPP_

#include <mshadow/tensor.h>
#include "./layer.h"
#include "./param.h"
#include "./op.h"
#include "../utils/utils.h"

namespace cxxnet {
namespace layer {

template<typename xpu>
class CLSTMLayer : public ConvolutionLayer<xpu> {
 public:
  CLSTMLayer(mshadow::Random<xpu> *p_rnd) : ConvolutionLayer<xpu>(p_rnd) {
    this->parallel_size = 1;
  }



  size_t parallel_size, seq_length;

};
}  // namespace layer
}  // namespace cxxnet
#endif  // LAYER_CLSTM_LAYER_INL_HPP_
