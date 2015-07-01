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
  CLSTMLayer(mshadow::Random<xpu> *p_rnd) : Conv(p_rnd) {
    this->parallel_size = 1;
  }
  virtual ~CLSTMLayer(void) {
    conv_node_in.FreeSpace();
    conv_node_out.FreeSpace();
  }

  virtual void SetParam(const char *name, const char* val) {
    Conv::SetParam(name, val);
    if (!strcmp(name, "parallel_size")) this->parallel_size = atoi(val);
    if (Conv::param_.num_channel % 4 != 0)
      utils::Error("num_channel mod 4 should be zero");
  }

  virtual void SetStream(mshadow::Stream<xpu> *stream) {
    Conv::SetStream(stream);

    it.set_stream(stream);
    ft.set_stream(stream);
    ot.set_stream(stream);
    gt.set_stream(stream);
    ct.set_stream(stream);
    c_tanht.set_stream(stream);
    ht.set_stream(stream);

    flush.set_stream(stream);
    t.set_stream(stream);
    xhprev.set_stream(stream);
    d_xhprev.set_stream(stream);
    lifog.set_stream(stream);
    d_lifog.set_stream(stream);
    d_c.set_stream(stream);
    d_cprev.set_stream(stream);

    conv_node_in.data.set_stream(stream);
    conv_node_out.data.set_stream(stream);
  }

  virtual void InitConnection(const std::vector<Node<xpu>*> &nodes_in,
                              const std::vector<Node<xpu>*> &nodes_out,
                              ConnectState<xpu> *p_cstate) {
    conv_node_in.data.shape_ = mshadow::Shape4(
        this->parallel_size, nodes_in[0]->data.size(1) + Conv::param_.num_channel / 4,
        nodes_in[0]->data.size(2), nodes_in[0]->data.size(3));
    conv_node_in.must_contiguous = true;
    conv_node_out.must_contiguous = true;
    conv_nodes_in.push_back(&conv_node_in);
    conv_nodes_out.push_back(&conv_node_out);
    conv_node_in.AllocSpace();
    Conv::InitConnection(conv_nodes_in, conv_nodes_out, p_cstate);
    conv_node_out.AllocSpace();
    if (conv_node_out.data.size(2) != conv_node_in.data.size(2) ||
        conv_node_out.data.size(3) != conv_node_in.data.size(3)) {
      utils::Error("Conv output should be the same size as the input");
    }

    nodes_out[0]->data.shape_ =
        mshadow::Shape4(nodes_in[0]->data.size(0), Conv::param_.num_channel / 4, conv_node_out.data.size(2), conv_node_out.data.size(3));
    nodes_in[0]->must_contiguous = true;
    nodes_in[1]->must_contiguous = true;
    nodes_out[0]->must_contiguous = true;

    this->seq_length = nodes_in[0]->data.size(0);
    this->num_hidden_in = nodes_in[0]->data.size(1) * nodes_in[0]->data.size(2) * nodes_in[0]->data.size(3);
    this->num_hidden_out = nodes_out[0]->data.size(1) * nodes_out[0]->data.size(2) * nodes_out[0]->data.size(3);
    this->initTemp();


  }

/*  virtual void OnBatchSizeChanged(const std::vector<Node<xpu>*> &nodes_in,
                                  const std::vector<Node<xpu>*> &nodes_out,
                                  ConnectState<xpu> *p_cstate) {
    Conv::OnBatchSizeChanged(nodes_in, nodes_out, p_cstate);
    this->seq_length = nodes_in[0]->data.size(0);
    this->num_hidden_out = nodes_out[0]->data.size(1) * nodes_out[0]->data.size(2) * nodes_out[0]->data.size(3) / 4;
    this->initTemp();
  }*/
  virtual void Forward(bool is_train,
                       const std::vector<Node<xpu>*> &nodes_in,
                       const std::vector<Node<xpu>*> &nodes_out,
                       ConnectState<xpu> *p_cstate) {
    mshadow::Tensor<xpu, 4> &node_out = nodes_out[0]->data;
    mshadow::Tensor<xpu, 4> xt = nodes_in[0]->data;
    mshadow::Tensor<xpu, 4> seq_label = nodes_in[1]->data;
    
    index_t n_seq = seq_length / parallel_size;
    xt.shape_ = mshadow::Shape4(n_seq,1, parallel_size, num_hidden_in);
    xt.stride_ = num_hidden_in;
    seq_label.shape_ = mshadow::Shape4(n_seq, 1, 1, parallel_size);
    seq_label.stride_ = parallel_size;
    
    for (index_t i = 0; i < n_seq; i++){
      flush = mshadow::expr::broadcast<0>(seq_label[i][0][0], flush.shape_);
      if (i != 0)
        t = flush * ht[i-1][0];
      else
        t = flush * ht[n_seq-1][0];
      concat2D(xhprev, xt[i][0], t);
      conv_node_in.data = mshadow::expr::reshape(xhprev, conv_node_in.data.shape_);
      Conv::Forward(is_train, conv_nodes_in, conv_nodes_out, p_cstate);
      lifog = mshadow::expr::reshape(conv_node_out.data.T(), lifog.shape_);
      if (i != 0)
        t = flush * ct[i-1][0];
      else
        t = flush * ct[n_seq-1][0];
      LSTM_Forward(lifog, t, ht[i][0], ct[i][0], it[i][0], ft[i][0], ot[i][0], gt[i][0], c_tanht[i][0]); 
    }
    node_out = mshadow::expr::reshape(ht, node_out.shape_);

  }

  virtual void Backprop(bool prop_grad,
                        const std::vector<Node<xpu>*> &nodes_in,
                        const std::vector<Node<xpu>*> &nodes_out,
                        ConnectState<xpu> *p_cstate) {
    mshadow::Tensor<xpu, 4> &node_in = nodes_in[0]->data;
    mshadow::Tensor<xpu, 4> &node_out = nodes_out[0]->data;
    mshadow::Tensor<xpu, 4> d_xt = node_in;
    mshadow::Tensor<xpu, 4> d_ht = node_out;
    mshadow::Tensor<xpu, 4> seq_label = nodes_in[1]->data;

    index_t n_seq = seq_length / parallel_size;
    d_xt.shape_ = mshadow::Shape4(n_seq,1,parallel_size,num_hidden_in);
    d_xt.stride_ = num_hidden_in;
    d_ht.shape_ = mshadow::Shape4(n_seq,1,parallel_size,num_hidden_out);
    d_ht.stride_ = num_hidden_out;
    seq_label.shape_ = mshadow::Shape4(n_seq, 1, 1, parallel_size);
    seq_label.stride_ = parallel_size;
    d_cprev = 0.0f;
    
    for (index_t i = n_seq - 1; i < n_seq; i--){ //unsigned int >=0
      mshadow::Copy(d_c, d_cprev, d_cprev.stream_);
      if (i == 0){
        flush = 0.0f;
        concat2D(xhprev, d_xt[i][0], flush);
        LSTM_Backprop(d_ht[i][0], flush, c_tanht[i][0], it[i][0], ft[i][0], ot[i][0], gt[i][0], d_lifog, d_c, d_cprev);
        conv_node_in.data = mshadow::expr::reshape(xhprev, conv_node_in.data.shape_);
        conv_node_out.data = mshadow::expr::reshape(d_lifog.T(), conv_node_out.data.shape_);
        Conv::Backprop(prop_grad, conv_nodes_in, conv_nodes_out, p_cstate);
        d_xhprev = mshadow::expr::reshape(conv_node_in.mat().T(), d_xhprev.shape_);        
      }else{
        flush = mshadow::expr::broadcast<0>(seq_label[i][0][0], flush.shape_);
        t = flush * ht[i-1][0];
        concat2D(xhprev, d_xt[i][0], t);
        t = flush * ct[i-1][0];
        LSTM_Backprop(d_ht[i][0], t, c_tanht[i][0], it[i][0], ft[i][0], ot[i][0], gt[i][0], d_lifog, d_c, d_cprev);
        conv_node_in.data = mshadow::expr::reshape(xhprev, conv_node_in.data.shape_);
        conv_node_out.data = mshadow::expr::reshape(d_lifog.T(), conv_node_out.data.shape_);
        Conv::Backprop(prop_grad, conv_nodes_in, conv_nodes_out, p_cstate);
        d_xhprev = mshadow::expr::reshape(conv_node_in.mat().T(), d_xhprev.shape_);        
        t = d_xhprev.Slice(num_hidden_in, num_hidden_in + num_hidden_out).T();
        d_ht[i-1][0] += flush * t;
        d_cprev *= flush;
      }
      if (prop_grad) {
        d_xt[i][0] = d_xhprev.Slice(0, num_hidden_in).T();
      }
    }
  }

 protected: 
  void LSTM_Forward(mshadow::Tensor<xpu, 2> lifog,
                    mshadow::Tensor<xpu, 2> cprev, 
                    mshadow::Tensor<xpu, 2> h,      
                    mshadow::Tensor<xpu, 2> c,
                    mshadow::Tensor<xpu, 2> i,
                    mshadow::Tensor<xpu, 2> f,
                    mshadow::Tensor<xpu, 2> o,
                    mshadow::Tensor<xpu, 2> g,
                    mshadow::Tensor<xpu, 2> c_tanh){
    using namespace cxxnet::op;
    using namespace mshadow::expr;
    /*
      li_t = w_ix $ x_t + w_ih $ h_t-1 + b_i
      lf_t = w_fx $ x_t + w_fh $ h_t-1 + b_f
      lo_t = w_ox $ x_t + w_oh $ h_t-1 + b_o
      lg_t = w_gx $ x_t + w_gh $ h_t-1 + b_g
      lifog = [li_t, lf_t, lo_t, lg_t]
     */
    mshadow::Tensor<xpu, 2> li, lf, lo, lg;
    li = lifog.Slice(0 * num_hidden_out, 1 * num_hidden_out);
    lf = lifog.Slice(1 * num_hidden_out, 2 * num_hidden_out);
    lo = lifog.Slice(2 * num_hidden_out, 3 * num_hidden_out);
    lg = lifog.Slice(3 * num_hidden_out, 4 * num_hidden_out);
    /*
      i_t = sigmoid(li_t)
      f_t = sigmoid(lf_t)
      o_t = sigmoid(lo_t)
      g_t = tanh(lg_t)
     */
    i = F<sigmoid>(li.T());
    f = F<sigmoid>(lf.T());
    o = F<sigmoid>(lo.T());
    g = F<tanh>(lg.T());
    /*
      c_t = f_t * c_t-1 + i_t * g_t
      h_t = o_t * tanh(c_t)
     */
    c = f * cprev + i * g;
    c_tanh = F<tanh>(c);
    h = o * c_tanh;
  }

  void LSTM_Backprop(mshadow::Tensor<xpu, 2> d_h,
                     mshadow::Tensor<xpu, 2> cprev,
                     mshadow::Tensor<xpu, 2> c_tanh,
                     mshadow::Tensor<xpu, 2> i,
                     mshadow::Tensor<xpu, 2> f,
                     mshadow::Tensor<xpu, 2> o,
                     mshadow::Tensor<xpu, 2> g,
                     mshadow::Tensor<xpu, 2> d_lifog,
                     mshadow::Tensor<xpu, 2> d_c,
                     mshadow::Tensor<xpu, 2> d_cprev){
    using namespace cxxnet::op;
    using namespace mshadow::expr;
    
    d_c += F<tanh_grad>(c_tanh) * o * d_h;
    d_cprev = f * d_c;
    
    mshadow::Tensor<xpu, 2> d_li, d_lf, d_lo, d_lg;
    d_li = d_lifog.Slice(0 * num_hidden_out, 1 * num_hidden_out);
    d_lf = d_lifog.Slice(1 * num_hidden_out, 2 * num_hidden_out);
    d_lo = d_lifog.Slice(2 * num_hidden_out, 3 * num_hidden_out);
    d_lg = d_lifog.Slice(3 * num_hidden_out, 4 * num_hidden_out);
    
    d_li = F<sigmoid_grad>(i.T()) * g.T() * d_c.T();
    d_lf = F<sigmoid_grad>(f.T()) * cprev.T() * d_c.T();
    d_lo = F<sigmoid_grad>(o.T()) * c_tanh.T() * d_h.T();
    d_lg = F<tanh_grad>(g.T()) * i.T() * d_c.T();


  }

  inline void initTemp(){
    it.Resize(mshadow::Shape4(seq_length / parallel_size, 1, parallel_size, num_hidden_out));
    ft.Resize(mshadow::Shape4(seq_length / parallel_size, 1, parallel_size, num_hidden_out));
    ot.Resize(mshadow::Shape4(seq_length / parallel_size, 1, parallel_size, num_hidden_out));
    gt.Resize(mshadow::Shape4(seq_length / parallel_size, 1, parallel_size, num_hidden_out));
    ct.Resize(mshadow::Shape4(seq_length / parallel_size, 1, parallel_size, num_hidden_out));
    c_tanht.Resize(mshadow::Shape4(seq_length / parallel_size, 1, parallel_size, num_hidden_out));
    ht.Resize(mshadow::Shape4(seq_length / parallel_size, 1, parallel_size, num_hidden_out));
  
    flush.Resize(mshadow::Shape2(parallel_size, num_hidden_out));
    t.Resize(mshadow::Shape2(parallel_size, num_hidden_out));
    d_c.Resize(mshadow::Shape2(parallel_size, num_hidden_out));
    d_cprev.Resize(mshadow::Shape2(parallel_size, num_hidden_out));
    xhprev.Resize(mshadow::Shape2(parallel_size, num_hidden_in + num_hidden_out));
    d_xhprev.Resize(mshadow::Shape2(num_hidden_in + num_hidden_out, parallel_size));
    lifog.Resize(mshadow::Shape2(4 * num_hidden_out, parallel_size));
    d_lifog.Resize(mshadow::Shape2(4 * num_hidden_out, parallel_size));
  }

  inline void tensor2To4(mshadow::Tensor<xpu, 2> a, mshadow::Tensor<xpu, 4> *a4){
    CHECK(a.CheckContiguous());
    a4->set_stream(a.stream_);
    a4->dptr_ = a.dptr_;
    a4->stride_ = a.stride_;
    a4->shape_ = mshadow::Shape4(1,1,a.size(0),a.size(1));
    CHECK(a4->CheckContiguous());
  }

  inline void concat2D(mshadow::Tensor<xpu, 2> dst, mshadow::Tensor<xpu, 2> a, mshadow::Tensor<xpu, 2> b){
    utils::Check(a.size(0) == b.size(0) && b.size(0) == dst.size(0), "CLSTMLayer: concat size[0] mismatch");
    utils::Check(a.size(1) + b.size(1) == dst.size(1), "CLSTMLayer: concat size[1] mismatch");
    mshadow::Tensor<xpu, 4> dst4, a4, b4;
    tensor2To4(dst, &dst4);
    tensor2To4(a, &a4);
    tensor2To4(b, &b4);
    dst4 = mshadow::expr::concat<3>(a4, b4);
    CHECK(dst.CheckContiguous());
  }

  typedef ConvolutionLayer<xpu> Conv;
  /*! \brief batched BPTT */
  size_t parallel_size, seq_length, num_hidden_in, num_hidden_out;
  /*! \brief var in LSTM layer */
  mshadow::TensorContainer<xpu, 4> it, ft, ot, gt, ct, c_tanht, ht;
  mshadow::TensorContainer<xpu, 2> flush, t;
  mshadow::TensorContainer<xpu, 2> xhprev;
  mshadow::TensorContainer<xpu, 2> d_xhprev;
  mshadow::TensorContainer<xpu, 2> lifog;
  mshadow::TensorContainer<xpu, 2> d_lifog;
  mshadow::TensorContainer<xpu, 2> d_c;
  mshadow::TensorContainer<xpu, 2> d_cprev;
  /*! \brief io of the embeded conv layer */
  std::vector<Node<xpu>*> conv_nodes_in, conv_nodes_out;
  Node<xpu> conv_node_in, conv_node_out;
};
}  // namespace layer
}  // namespace cxxnet
#endif  // LAYER_CLSTM_LAYER_INL_HPP_
