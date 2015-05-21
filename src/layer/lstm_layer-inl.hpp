#ifndef CXXNET_LAYER_LSTM_LAYER_INL_HPP_
#define CXXNET_LAYER_LSTM_LAYER_INL_HPP_

#include <mshadow/tensor.h>
#include "./layer.h"
#include "./param.h"
#include "./op.h"
#include "../utils/utils.h"

namespace cxxnet {
namespace layer {

template<typename xpu>
class LSTMLayer : public ILayer<xpu> {
 public:
  LSTMLayer(mshadow::Random<xpu> *p_rnd) : prnd_(p_rnd) {
    this->parallel_size = 1;
  }
  virtual ~LSTMLayer(void) {}
  virtual void SetParam(const char *name, const char* val) {
    param_.SetParam(name, val);
    if (!strcmp(name, "parallel_size")) this->parallel_size = atoi(val);
  }
  virtual void ApplyVisitor(typename ILayer<xpu>::IVisitor *pvisitor) {
    pvisitor->Visit("wmat", wmat_, gwmat_);
    pvisitor->Visit("bias", bias_, gbias_);
  }
  virtual void InitModel(void) {
    //ifog weights: input, forget, output and cell gate * input vector and hidden state
    //ifog bias: input, forget, output and cell gate
    wmat_.Resize(mshadow::Shape2(param_.num_hidden * 4, param_.num_hidden + param_.num_input_node));
    bias_.Resize(mshadow::Shape1(param_.num_hidden * 4));
    param_.RandInitWeight(this->prnd_, wmat_, wmat_.size(1), wmat_.size(0));
    bias_ = param_.init_bias;
    
    gwmat_.Resize(wmat_.shape_);
    gbias_.Resize(bias_.shape_);
    gwmat_ = 0.0f; gbias_ = 0.0f; 
  }
  virtual void SaveModel(utils::IStream &fo) const {
    fo.Write(&param_, sizeof(LayerParam));
    wmat_.SaveBinary(fo);
    bias_.SaveBinary(fo);
  }
  virtual void LoadModel(utils::IStream &fi) {
    utils::Check(fi.Read(&param_, sizeof(LayerParam)) != 0,
                 "LSTMLayer:LoadModel invalid model file");    
    wmat_.LoadBinary(fi);
    bias_.LoadBinary(fi); 
    gwmat_.Resize(wmat_.shape_);
    gbias_.Resize(bias_.shape_);
    gwmat_ = 0.0f; gbias_ = 0.0f;
  }
  virtual void SetStream(mshadow::Stream<xpu> *stream) {
    wmat_.set_stream(stream);
    bias_.set_stream(stream);
    gwmat_.set_stream(stream);
    gbias_.set_stream(stream);

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
    lifog.set_stream(stream);
    d_xhprev.set_stream(stream);
    d_lifog.set_stream(stream);
    d_c.set_stream(stream);
    d_cprev.set_stream(stream);
  }
  virtual void InitConnection(const std::vector<Node<xpu>*> &nodes_in,
                              const std::vector<Node<xpu>*> &nodes_out,
                              ConnectState<xpu> *p_cstate) {
    utils::Check((nodes_in.size() == 1 || nodes_in.size() == 2) && nodes_out.size() == 1,
                 "LSTMLayer: Layer only support 2(w/sequence label)-1 connection");
    utils::Check(param_.num_hidden > 0, "LSTMLayer: must set nhidden correctly");
    nodes_out[0]->data.shape_ =
        mshadow::Shape4(nodes_in[0]->data.size(0), 1, 1, param_.num_hidden);
    if (param_.num_input_node == 0) {
      param_.num_input_node = static_cast<int>(nodes_in[0]->data.size(3));
    } else {
      utils::Check(param_.num_input_node == static_cast<int>(nodes_in[0]->data.size(3)),
                   "LSTMLayer: input hidden nodes is not consistent");
    }
    this->seq_length = nodes_in[0]->data.size(0);
    this->initTemp();
  }

  virtual void OnBatchSizeChanged(const std::vector<Node<xpu>*> &nodes_in,
				  const std::vector<Node<xpu>*> &nodes_out,
				  ConnectState<xpu> *p_cstate) {
    this->seq_length = nodes_in[0]->data.size(0);
    this->initTemp();
  }
  
  /*
    nodes_in[0] size: [batch_size][1][1][input_width]
    nodes_in[1] size: [batch_size][1][1][1]
    nodes_out[0] size: [batch_size][1][1][hidden_size]
    
    The input sequence nodes_in[0] should be:
    Seq[0][i], Seq[1][j], ... , Seq[parallel_size][k], Seq[0][i + 1], Seq[1][j + 1], ... , Seq[parallel_size][k + 1], ...
    The correspond sequence label (in nodes_in[1]) should be '1' when it is the beginning of a sequence.
  */
  virtual void Forward(bool is_train,
                       const std::vector<Node<xpu>*> &nodes_in,
                       const std::vector<Node<xpu>*> &nodes_out,
                       ConnectState<xpu> *p_cstate) {
    mshadow::Tensor<xpu, 4> &node_in = nodes_in[0]->data;
    mshadow::Tensor<xpu, 4> &node_out = nodes_out[0]->data;
    mshadow::Tensor<xpu, 4> xt = node_in;
    mshadow::Tensor<xpu, 4> seq_label = nodes_in[1]->data;
    
    index_t n_seq = seq_length / parallel_size;
    xt.shape_ = mshadow::Shape4(n_seq,1,parallel_size,node_in.size(3));
    seq_label.shape_ = mshadow::Shape4(n_seq, 1, 1, parallel_size);
    
    for (index_t i = 0; i < n_seq; i++){
      if (i == 0){
	concat2D(xhprev, xt[i][0], ht[n_seq-1][0]);
	LSTM_Forward(xhprev, ct[n_seq-1][0], ht[i][0], ct[i][0], it[i][0], ft[i][0], ot[i][0], gt[i][0], c_tanht[i][0]);
      }else{
	flush = 1.0f - mshadow::expr::broadcast<0>(seq_label[i][0][0], ht[i-1][0].shape_);
	t = flush * ht[i-1][0];
	concat2D(xhprev, xt[i][0], t);
	t = flush * ct[i-1][0];
	LSTM_Forward(xhprev, t, ht[i][0], ct[i][0], it[i][0], ft[i][0], ot[i][0], gt[i][0], c_tanht[i][0]);
      }
    }
    ht.shape_ = node_out.shape_;
    mshadow::Copy(node_out, ht, ht.stream_);
    ht.shape_ = ct.shape_;
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
    d_xt.shape_ = mshadow::Shape4(n_seq,1,parallel_size,node_in.size(3));
    d_ht.shape_ = mshadow::Shape4(n_seq,1,parallel_size,node_out.size(3));
    seq_label.shape_ = mshadow::Shape4(n_seq, 1, 1, parallel_size);
    d_cprev = 0.0f;

    for (index_t i = n_seq - 1; i < n_seq; i--){ //unsigned int >=0
      mshadow::Copy(d_c, d_cprev, d_cprev.stream_);
      if (i == 0){
	flush = 0.0f;
        concat2D(xhprev, d_xt[i][0], flush);
	LSTM_Backprop(d_ht[i][0], xhprev, flush, c_tanht[i][0], it[i][0], ft[i][0], ot[i][0], gt[i][0], d_xhprev, d_c, d_cprev);
      }else{
	flush = 1.0f - mshadow::expr::broadcast<0>(seq_label[i][0][0], ht[i-1][0].shape_);
   	t = flush * ht[i-1][0];
	concat2D(xhprev, d_xt[i][0], t);
	t = flush * ct[i-1][0];
	LSTM_Backprop(d_ht[i][0], xhprev, t, c_tanht[i][0], it[i][0], ft[i][0], ot[i][0], gt[i][0], d_xhprev, d_c, d_cprev);
	t = d_xhprev.Slice(param_.num_input_node, param_.num_input_node + param_.num_hidden).T();
	d_ht[i-1][0] += flush * t;
	d_cprev *= flush;
      }
      if (prop_grad) {
	d_xt[i][0] = d_xhprev.Slice(0, param_.num_input_node).T();
      }
    }
  }

 protected:  
  void LSTM_Forward(mshadow::Tensor<xpu, 2> xhprev,
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
      li_t = w_ix * x_t + w_ih * h_t-1 + b_i
      lf_t = w_fx * x_t + w_fh * h_t-1 + b_f
      lo_t = w_ox * x_t + w_oh * h_t-1 + b_o
      lg_t = w_gx * x_t + w_gh * h_t-1 + b_g
      lifog = [li_t, lf_t, lo_t, lg_t]
     */
    lifog = broadcast<0>(bias_, lifog.shape_);
    lifog += dot(wmat_, xhprev.T());
    mshadow::Tensor<xpu, 2> li, lf, lo, lg;
    li = lifog.Slice(0 * param_.num_hidden, 1 * param_.num_hidden);
    lf = lifog.Slice(1 * param_.num_hidden, 2 * param_.num_hidden);
    lo = lifog.Slice(2 * param_.num_hidden, 3 * param_.num_hidden);
    lg = lifog.Slice(3 * param_.num_hidden, 4 * param_.num_hidden);
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
                     mshadow::Tensor<xpu, 2> xhprev,
                     mshadow::Tensor<xpu, 2> cprev,
                     mshadow::Tensor<xpu, 2> c_tanh,
                     mshadow::Tensor<xpu, 2> i,
                     mshadow::Tensor<xpu, 2> f,
                     mshadow::Tensor<xpu, 2> o,
                     mshadow::Tensor<xpu, 2> g,
                     mshadow::Tensor<xpu, 2> d_xhprev,
                     mshadow::Tensor<xpu, 2> d_c,
                     mshadow::Tensor<xpu, 2> d_cprev){
    using namespace cxxnet::op;
    using namespace mshadow::expr;
    
    d_c += F<tanh_grad>(c_tanh) * o * d_h;
    d_cprev = f * d_c;
    
    mshadow::Tensor<xpu, 2> d_li, d_lf, d_lo, d_lg;
    d_li = d_lifog.Slice(0 * param_.num_hidden, 1 * param_.num_hidden);
    d_lf = d_lifog.Slice(1 * param_.num_hidden, 2 * param_.num_hidden);
    d_lo = d_lifog.Slice(2 * param_.num_hidden, 3 * param_.num_hidden);
    d_lg = d_lifog.Slice(3 * param_.num_hidden, 4 * param_.num_hidden);
    
    d_li = F<sigmoid_grad>(i.T()) * g.T() * d_c.T();
    d_lf = F<sigmoid_grad>(f.T()) * cprev.T() * d_c.T();
    d_lo = F<sigmoid_grad>(o.T()) * c_tanh.T() * d_h.T();
    d_lg = F<tanh_grad>(g.T()) * i.T() * d_c.T();

    gwmat_ += dot(d_lifog, xhprev);
    gbias_ += sum_rows(d_lifog.T());
    d_xhprev = dot(wmat_.T(), d_lifog);
  }

  inline void tensor2To4(mshadow::Tensor<xpu, 2> a, mshadow::Tensor<xpu, 4> *a4){
    a4->set_stream(a.stream_);
    a4->dptr_ = a.dptr_;
    a4->stride_ = a.stride_;
    a4->shape_ = mshadow::Shape4(1,1,a.size(0),a.size(1));
  }

  inline void concat2D(mshadow::Tensor<xpu, 2> dst, mshadow::Tensor<xpu, 2> a, mshadow::Tensor<xpu, 2> b){
    utils::Check(a.size(0) == b.size(0) && b.size(0) == dst.size(0), "LSTMLayer: concat size[0] mismatch");
    utils::Check(a.size(1) + b.size(1) == dst.size(1), "LSTMLayer: concat size[1] mismatch");
    mshadow::Tensor<xpu, 4> dst4, a4, b4;
    tensor2To4(dst, &dst4);
    tensor2To4(a, &a4);
    tensor2To4(b, &b4);
    dst4 = mshadow::expr::concat<3>(a4, b4);
  }

  inline void initTemp(){
    it.Resize(mshadow::Shape4(seq_length / parallel_size, 1, parallel_size, param_.num_hidden));
    ft.Resize(mshadow::Shape4(seq_length / parallel_size, 1, parallel_size, param_.num_hidden));
    ot.Resize(mshadow::Shape4(seq_length / parallel_size, 1, parallel_size, param_.num_hidden));
    gt.Resize(mshadow::Shape4(seq_length / parallel_size, 1, parallel_size, param_.num_hidden));
    ct.Resize(mshadow::Shape4(seq_length / parallel_size, 1, parallel_size, param_.num_hidden));
    c_tanht.Resize(mshadow::Shape4(seq_length / parallel_size, 1, parallel_size, param_.num_hidden));
    ht.Resize(mshadow::Shape4(seq_length / parallel_size, 1, parallel_size, param_.num_hidden));
  
    flush.Resize(mshadow::Shape2(parallel_size, param_.num_hidden));
    t.Resize(mshadow::Shape2(parallel_size, param_.num_hidden));
    xhprev.Resize(mshadow::Shape2(parallel_size, param_.num_input_node + param_.num_hidden));
    d_xhprev.Resize(mshadow::Shape2(param_.num_input_node + param_.num_hidden, parallel_size));
    d_c.Resize(mshadow::Shape2(parallel_size, param_.num_hidden));
    d_cprev.Resize(mshadow::Shape2(parallel_size, param_.num_hidden));
    lifog.Resize(mshadow::Shape2(4 * param_.num_hidden, parallel_size));
    d_lifog.Resize(mshadow::Shape2(4 * param_.num_hidden, parallel_size));
  }
  
  /*! \brief random number generator */
  mshadow::Random<xpu> *prnd_;
  /*! \brief parameters that potentially be useful */
  LayerParam param_;
  /*! \brief weight matrix */
  mshadow::TensorContainer<xpu,2> wmat_;
  /*! \brief bias */
  mshadow::TensorContainer<xpu,1> bias_;
  /*! \brief accumulates the gradient of weight matrix */
  mshadow::TensorContainer<xpu,2> gwmat_;
  /*! \brief accumulates the gradient of bias */  
  mshadow::TensorContainer<xpu,1> gbias_;
 
  /*! \brief batched BPTT */
  size_t parallel_size, seq_length;

  /*! \brief var in LSTM layer */
  mshadow::TensorContainer<xpu, 4> it, ft, ot, gt, ct, c_tanht, ht;
  mshadow::TensorContainer<xpu, 2> flush, t;
  mshadow::TensorContainer<xpu, 2> xhprev;
  mshadow::TensorContainer<xpu, 2> lifog;
  mshadow::TensorContainer<xpu, 2> d_xhprev;
  mshadow::TensorContainer<xpu, 2> d_lifog;
  mshadow::TensorContainer<xpu, 2> d_c;
  mshadow::TensorContainer<xpu, 2> d_cprev;
};
}  // namespace layer
}  // namespace cxxnet
#endif  // LAYER_LSTM_LAYER_INL_HPP_
