#include "myproject.hxx"
#include <algorithm>

using TMVA_SOFIE_myproject::Session;

extern "C" {

// Top-level inference function (like hls4ml myproject)
void myproject_float(//insert_inputs, float *output) {

    static Session session("//insert_dat_path");   

    auto rout = session.infer(//insert_ref_inputs);

    std::copy(rout.begin(), rout.end(), output);
}

}
