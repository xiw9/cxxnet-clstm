#ifndef CXXNET_LAYER_SOFTMAX_LAYER_INL_HPP_
#define CXXNET_LAYER_SOFTMAX_LAYER_INL_HPP_

#include <dmlc/logging.h>
#include <mshadow/tensor.h>
#include "../layer.h"
#include "./loss_layer_base-inl.hpp"

namespace cxxnet {
namespace layer {
/*! \brief loss function layer */
template<typename xpu>
class SoftmaxLayer: public LossLayerBase<xpu> {
 public:
  SoftmaxLayer(const LabelInfo *label_info)
      : LossLayerBase<xpu>(label_info) {
     skipzero = 0;
     kk = 1.0f;
  }
  virtual ~SoftmaxLayer(void) {
  }
  virtual void SetParam(const char *name, const char *val) {
    if (!strcmp(name, "skipzero")) skipzero = atoi(val);
    if (!strcmp(name, "k")) kk = atof(val);
    LossLayerBase<xpu>::SetParam(name, val);
  }
 protected:
  virtual void Forward_(mshadow::Tensor<xpu, 2> inout_data,
                        mshadow::Stream<xpu> *stream) {
    mshadow::Softmax(inout_data, inout_data);
  }
  virtual void SetGradCPU(mshadow::Tensor<cpu, 2> inout_data,
                          const LabelRecord &label) {
    mshadow::Tensor<cpu, 2> lb = label.label;
    CHECK(lb.size(0) == inout_data.size(0) && lb.size(1) == 1)
        << "SoftmaxLayer: label size mismatch";
    for (mshadow::index_t i = 0; i < inout_data.size(0); ++i) {
      index_t k = static_cast<index_t>(lb[i][0]);
      inout_data[i][k] -= 1.0f;
      inout_data[i] *= kk;
      if (skipzero && k==0)
        inout_data[i] = 0.0f;
    }
  }
  int skipzero;
  float kk;
};
}  // namespace layer
}  // namespace cxxnet
#endif  // LAYER_SOFTMAX_LAYER_INL_HPP_
