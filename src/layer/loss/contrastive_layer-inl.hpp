#ifndef CXXNET_LAYER_Contrastive_LOSS_LAYER_INL_HPP_
#define CXXNET_LAYER_Contrastive_LOSS_LAYER_INL_HPP_

#include <dmlc/logging.h>
#include <mshadow/tensor.h>
#include "../layer.h"
#include "./loss_layer_base-inl.hpp"

namespace cxxnet {
namespace layer {
/*! \brief loss function layer */
template<typename xpu>
class ContrastiveLossLayer: public LossLayerBase<xpu> {
 public:
  ContrastiveLossLayer(const LabelInfo *label_info)
      : LossLayerBase<xpu>(label_info) {
    p = 1024;
    m = 0.5477f;
    k = 1.0f;
  }
  virtual ~ContrastiveLossLayer(void) {
  }
  virtual void SetParam(const char *name, const char *val) {
    if (!strcmp(name, "width")) p = atoi(val);
    if (!strcmp(name, "m")) m = atof(val);
    if (!strcmp(name, "k")) k = atof(val);
    LossLayerBase<xpu>::SetParam(name, val);
  }
 protected:
  virtual void Forward_(mshadow::Tensor<xpu, 2> inout_data,
                        mshadow::Stream<xpu> *stream) {
    // Do Nothing
  }
  virtual void SetGradCPU(mshadow::Tensor<cpu, 2> inout_data,
                          const LabelRecord &label) {
    mshadow::Tensor<cpu, 2> lb = label.label;
    CHECK(lb.size(0) == inout_data.size(0) && lb.size(1) == 1)
      << "ContrastiveLayer: label size mismatch";
    for (index_t i = 1; i < inout_data.size(0) ; i += 2) {
      float d = 0.0f;
      for (int j = 0; j < p; ++j)
        d += (inout_data[i][j] - inout_data[i][j + p]) * (inout_data[i][j] - inout_data[i][j + p]);

      if (lb[i][0] > 0.5f){ //postive pair
     	for (int j = 0; j < p; ++j){
	  inout_data[i][j] = 2.0f * (inout_data[i][j] - inout_data[i][j + p]);
          if (d > m * m)
            inout_data[i][j] = 0; //inout_data[i][j] * 0.01f;
	  inout_data[i][j + p] = -1.0f * inout_data[i][j];
	}
      }else{ //neg
	for (int j = 0; j < p; ++j){
	  if (d < m * m) {
	    inout_data[i][j] = -2.0f * (inout_data[i][j] - inout_data[i][j + p]);
	  } else {
	    inout_data[i][j] = 0;
	  }
	  inout_data[i][j + p] = -1.0f * inout_data[i][j];
	}
      }
      for (index_t j = 0; j < inout_data.size(1); ++j){
	inout_data[i - 1][j] = inout_data[i][j] / 2.0f * k;
	inout_data[i][j] = inout_data[i][j] / 2.0f * k;
      }
    }
  }
 private:
  // feature width
  int p;
  float m, k;
};
}  // namespace layer
}  // namespace cxxnet
#endif  // LAYER_Contrastive_LOSS_LAYER_INL_HPP_
