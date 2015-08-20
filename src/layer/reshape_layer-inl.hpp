#ifndef CXXNET_LAYER_RESHAPE_LAYER_INL_HPP_
#define CXXNET_LAYER_FESHAPE_LAYER_INL_HPP_

#include "./layer.h"
#include "./op.h"

namespace cxxnet {
namespace layer {

template<typename xpu>
class ReshapeLayer : public ILayer<xpu> {
 public:
  virtual ~ReshapeLayer(void) {}
  ReshapeLayer(void){
    this->ratio = 20;
    this->clip = 0;
    this->reverse = 0;
  }
  virtual void SetParam(const char *name, const char* val) {
    if (!strcmp(name, "ratio")) this->ratio = atoi(val);
    if (!strcmp(name, "clip")) this->clip = atoi(val);
    if (!strcmp(name, "reverse")) this->reverse = atoi(val);
  }
  virtual void InitConnection(const std::vector<Node<xpu>*> &nodes_in,
                              const std::vector<Node<xpu>*> &nodes_out,
                              ConnectState<xpu> *p_cstate) {
    utils::Check(nodes_in.size() == 1 && nodes_out.size() == 1,
                 "ReshapeLayer: only support 1-1 connection");
    mshadow::Shape<4> ishape = nodes_in[0]->data.shape_;
    if (clip) {
      nodes_out[0]->data.shape_ =
          mshadow::Shape4(ishape[0] / ratio, 1, ishape[2] , ishape[3]);
    }
    if (reverse){
      nodes_out[0]->data.shape_ =
          mshadow::Shape4(ishape[0] * ratio, 1, ishape[2], ishape[3]);
    }
    if (!clip && !reverse)
      nodes_out[0]->data.shape_ = 
          mshadow::Shape4(ishape[0] / ratio, ishape[1] * ratio, ishape[2], ishape[3]);
  }
  virtual void Forward(bool is_train,
                       const std::vector<Node<xpu>*> &nodes_in,
                       const std::vector<Node<xpu>*> &nodes_out,
                       ConnectState<xpu> *p_cstate) {
    using namespace mshadow::expr;
    mshadow::Tensor<xpu, 4> &node_in = nodes_in[0]->data;
    mshadow::Tensor<xpu, 4> &node_out = nodes_out[0]->data;

    if (clip){
      for (index_t i = 0; i < node_out.shape_[0]; i++) {
        node_out[i] = reshape(node_in[i * ratio], node_out[i].shape_);
      }
    }
    if (reverse){
      for (index_t i = 0; i < node_out.shape_[0]; i++)
        node_out[i] = reshape(node_in[i / ratio], node_out[i].shape_);
    }
    if (!clip && !reverse) 
      node_out = reshape(node_in, node_out.shape_);
  }
  virtual void Backprop(bool prop_grad,
                        const std::vector<Node<xpu>*> &nodes_in,
                        const std::vector<Node<xpu>*> &nodes_out,
                        ConnectState<xpu> *p_cstate) {
    using namespace mshadow::expr;
    mshadow::Tensor<xpu, 4> &node_in = nodes_in[0]->data;
    mshadow::Tensor<xpu, 4> &node_out = nodes_out[0]->data;
    if (prop_grad) {
      if (clip){
        // NOT implemented        
      }
      if (reverse){
        for (index_t i = 0; i < nodes_in[0]->data.shape_[0]; i++) {
          node_in[i] = reshape(node_out[i * ratio], node_in[i].shape_);
        }
      }
      if (!clip && !reverse)
        node_in= reshape(node_out, node_in.shape_);
    }    
  }
 private:
  index_t ratio, clip, reverse;
};
}  // namespace layer
}  // namespace cxxnet
#endif  // LAYER_RESHAPE_LAYER_INL_HPP_

