/**
 * d2q9 model
 * author: yanweian@shu.edu.cn
 * time: 2019-09-22
*/
#include "../inc/lbm.h"

int main(int argc, char **argv)
{
    int group_size = 4;
    int end = 1;
    int interval = 2;
    // read some value from command line
    Options options(argc, argv);
    if (options.has("g"))
    {
        group_size = options.get<int>("g");
    }
    if (options.has("e"))
    {
        end = options.get<unsigned>("e");
    }
    if (options.has("i"))
    {
        interval = options.get<double>("i");
    }

    cout << "hello lbm!" << endl;
    //========== init cell and preprocess ============
    cout << "===========Init cell and preprocess.\n";
    Lattice lattice;
    LbmParameters lbmparameters;
    lbmparameters.groupsize = MakeVec(group_size, group_size);
    lbmparameters.dxy = MakeVec(1.f, 1.f);
    lbmparameters.Re = 1000.f;
    lbmparameters.rho0 = 1.f;
    lbmparameters.u0 = MakeVec(0.1f, 0.f);
    lattice.init(lbmparameters);
    lattice.setTime(end, interval); //sum = end*interval;

    //========== init opencl environment and run ============
    cout << "===========Init opencl environment.\n";
    OpenCLFPGA openclFpga;
    openclFpga.Run(lattice);
    return SUCCESS_EXIT;
}