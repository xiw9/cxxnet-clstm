#include <mshadow/tensor.h>
#include <dmlc/io.h>
#include <dmlc/logging.h>
#include "../src/utils/config.h"
#include "../src/nnet/nnet.h"
#include "../src/io/data.h"
#include "../src/nnet/neural_net-inl.hpp"
#include "../src/nnet/nnet_impl-inl.hpp"
#include "../src/layer/convolution_layer-inl.hpp"
#include "../src/layer/cudnn_convolution_layer-inl.hpp"
#include "../src/layer/fullc_layer-inl.hpp"

namespace cxxnet {

void surgery(int argc, char *argv[]){
  if (argc != 3) {
    printf("Usage: <input config> <input model> <output config> <output model>\n");
  }
  int net_type;
  nnet::CXXNetThreadTrainer<cpu> *net0, *net1;

  dmlc::Stream *fi = dmlc::Stream::Create(argv[2], "r");
  CHECK(fi->Read(&net_type, sizeof(int)) != 0) << "invalid model format";
  net0 = (nnet::CXXNetThreadTrainer<cpu>*) nnet::CreateNet<mshadow::cpu>(net_type);
  dmlc::Stream *cfg = dmlc::Stream::Create(argv[1], "r");
  {
    dmlc::istream is(cfg);
    utils::ConfigStreamReader itr(is);
    itr.Init();
    while (itr.Next()) {
      net0->SetParam(itr.name(), itr.val());
    }
  }
  delete cfg;
  net0->LoadModel(*fi);
  delete fi;

  net1 = (nnet::CXXNetThreadTrainer<cpu>*) nnet::CreateNet<mshadow::cpu>(net_type);
  cfg = dmlc::Stream::Create(argv[3], "r");
  {
    dmlc::istream is(cfg);
    utils::ConfigStreamReader itr(is);
    itr.Init();
    while (itr.Next()) {
      net1->SetParam(itr.name(), itr.val());
    }
  }
  net1->InitModel();
  delete cfg;

  std::map<std::string,int>::iterator it;
  for(it = net0->net_cfg.layer_name_map.begin(); it != net0->net_cfg.layer_name_map.end(); ++it){
    mshadow::TensorContainer<mshadow::cpu, 2> wmat;
    mshadow::TensorContainer<mshadow::cpu, 2> bias;
    std::vector<index_t> shape;
    const char *cwmat = "wmat";
    const char *cbias = "bias";
    if (it->first == "conv_1"){
      net0->GetWeight(&wmat, &shape, it->first.c_str(), cwmat);      
      mshadow::TensorContainer<mshadow::cpu, 2> wmat1;
      wmat1.Resize(mshadow::Shape2(wmat.shape_[0], wmat.shape_[1] / 3 * 20));
      int l = wmat.shape_[1] / 3;
      for (index_t i = 0; i < wmat1.shape_[0]; i++)
        for (index_t j = 0; j < wmat1.shape_[1]; j++){
          wmat1[i][j] = wmat[i][j % l] + wmat[i][j % l + l] + wmat[i][j % l + 2 * l];
          wmat1[i][j] = wmat1[i][j] / 20.0;
        }
      net1->SetWeight(wmat1, it->first.c_str(), cwmat);
      net0->GetWeight(&bias, &shape, it->first.c_str(), cbias);
      net1->SetWeight(bias, it->first.c_str(), cbias);
    } else {
      net0->GetWeight(&wmat, &shape, it->first.c_str(), cwmat);
      if (shape.size() != 0)
        net1->SetWeight(wmat, it->first.c_str(), cwmat);
      net0->GetWeight(&bias, &shape, it->first.c_str(), cbias);
      if (shape.size() != 0)
        net1->SetWeight(bias, it->first.c_str(), cbias);
    }
    std::printf("%s\n", it->first.c_str());
  }

  dmlc::Stream *fo = dmlc::Stream::Create(argv[4], "w");
  fo->Write(&net_type, sizeof(int));
  net1->SaveModel(*fo);
  delete fo;
}
}

int main(int argc, char *argv[]){
  cxxnet::surgery(argc, argv);
  return 0;
}
