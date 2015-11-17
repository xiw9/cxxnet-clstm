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
  }
  virtual void SetParam(const char *name, const char* val) {
  }
  virtual void InitConnection(const std::vector<Node<xpu>*> &nodes_in,
                              const std::vector<Node<xpu>*> &nodes_out,
                              ConnectState<xpu> *p_cstate) {
    utils::Check(nodes_in.size() == 1 && nodes_out.size() == 1,
                 "ReshapeLayer: only support 1-1 connection");
    mshadow::Shape<4> ishape = nodes_in[0]->data.shape_;
    nodes_out[0]->data.shape_ = mshadow::Shape4(ishape[0], 2, 1, ishape[3]);
  }
  virtual void Forward(bool is_train,
                       const std::vector<Node<xpu>*> &nodes_in,
                       const std::vector<Node<xpu>*> &nodes_out,
                       ConnectState<xpu> *p_cstate) {
    using namespace mshadow::expr;
    mshadow::Tensor<xpu, 4> &node_in = nodes_in[0]->data;
    mshadow::Tensor<xpu, 4> &node_out = nodes_out[0]->data;
    for (index_t i = 0; i < node_out.shape_[0]; i++) {
      node_out[i][0] = reshape(node_in[(i / 2) * 2][0], node_out[i][0].shape_);
      node_out[i][1] = reshape(node_in[(i / 2) * 2 + 1][0], node_out[i][1].shape_);
    }
  }
  virtual void Backprop(bool prop_grad,
                        const std::vector<Node<xpu>*> &nodes_in,
                        const std::vector<Node<xpu>*> &nodes_out,
                        ConnectState<xpu> *p_cstate) {
    using namespace mshadow::expr;
    mshadow::Tensor<xpu, 4> &node_in = nodes_in[0]->data;
    mshadow::Tensor<xpu, 4> &node_out = nodes_out[0]->data;
    if (prop_grad) {
       for (index_t i = 0; i < nodes_in[0]->data.shape_[0]; i++) {
         node_in[i][0] = reshape(node_out[i][i % 2],node_out[i][0].shape_);
       }
    }    
  }
};
}  // namespace layer
}  // namespace cxxnet
#endif  // LAYER_RESHAPE_LAYER_INL_HPP_

