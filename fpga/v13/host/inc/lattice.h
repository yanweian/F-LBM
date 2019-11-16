#ifndef __LATTICE_H__
#define __LATTICE_H__
// model parameters
struct LbmParameters
{
    Vector2DIndex size;      // grid size (x,y)
    Vector2DIndex groupsize; // grid group size (x,y)
    LongIndex ncells;        // nums of cells
    Vector2DReal dxy;        // grid cell width
    Vector2DReal Lxy;        // grid length dx*nx
    real dt;                 // 时间步 dt=dx
    real Re;                 // 1000.f
    real niu;                // U*Lx/Re
    real c;                  // velocity of cell dx/dt
    Vector2DReal u0;         // 顶盖移动速度
    real rho0;               // 初始密度
    real tau_f;              // (1/tau)=2.0/(6.0*nu*dt/(dx*dx) + 1.0);
};
// recompute some values ncells, dt,Lxy,c,niu
inline LbmParameters recomputeParameters(LbmParameters lbmparameters)
{
    lbmparameters.size.x = lbmparameters.groupsize.x * N_VECTOR;
    lbmparameters.size.y = lbmparameters.groupsize.y * N_VECTOR;
    lbmparameters.ncells = lbmparameters.size.x * lbmparameters.size.y;
    lbmparameters.dt = lbmparameters.dxy.x;
    lbmparameters.Lxy.x = lbmparameters.size.x * lbmparameters.dxy.x;
    lbmparameters.Lxy.y = lbmparameters.size.y * lbmparameters.dxy.y;
    lbmparameters.c = lbmparameters.dxy.x / lbmparameters.dt;
    lbmparameters.niu = lbmparameters.u0.x * lbmparameters.Lxy.x / lbmparameters.Re;
    lbmparameters.tau_f = 2.0 / (6.0 * lbmparameters.niu * lbmparameters.dt / (lbmparameters.dxy.x * lbmparameters.dxy.x) + 1.0);
    return lbmparameters;
}
// lbm base class
class Lattice
{
public:
    scoped_aligned_ptr<HostCell> grid;
    LbmParameters lbmparameters;
    int iterate_end; // Simulation end time
    int interval;    // save file interval

    Lattice(int end = 5e3, int interval = 1e3)
    {
        this->iterate_end = end;
        this->interval = interval;
    }
    void init(LbmParameters parameters);
    void setTime(int end, int interval)
    {
        this->iterate_end = end;
        this->interval = interval;
    }

private:
    void initCell(); // init cell data like f、rho和u
    void setMask();
};
void Lattice::init(LbmParameters parameters)
{
    lbmparameters = parameters;
    // recompute values
    lbmparameters = recomputeParameters(lbmparameters);
    // Allocate memory space
    grid.reset(lbmparameters.ncells * 2);
    // init cell data
    initCell();
    // set mask
    setMask();
}
void Lattice::initCell() // init cell data like f、rho和u
{
    LongIndex offset = lbmparameters.ncells;
    for (Index y = 0; y < lbmparameters.size.y; y++)
    {
        for (Index x = 0; x < lbmparameters.size.x; x++)
        {
            // compute index
            LongIndex idx = getIndex(MakeVec(x, y), lbmparameters.size);
            // set position
            grid[idx].position = MakeVec(x, y);
            grid[idx + offset].position = MakeVec(x, y);
            // init mask
            grid[idx].mask = 0x0;
            grid[idx + offset].mask = 0x0;
            // process u
            real ux = 0.f;
            if (y == (lbmparameters.size.y - 1))
                ux = lbmparameters.u0.x;
            // init f
            for (int i = 0; i < Q_VECTOR; i++)
            {
                grid[idx].f[i] = feq(lbmparameters.rho0, MakeVec(ux, 0.f), i);
            }
        }
    }
}
void Lattice::setMask() // set mask, 00(x)00(y)
{
    LongIndex offset = lbmparameters.ncells;
    for (Index y = 0; y < lbmparameters.size.y; y++)
    {
        for (Index x = 0; x < lbmparameters.size.x; x++)
        {
            // compute index
            LongIndex idx = getIndex(MakeVec(x, y), lbmparameters.size);
            char mask = grid[idx].mask;
            // mask boundary
            if (x == 0)
                SETLEFT(mask);
            else if (x == lbmparameters.size.x - 1)
                SETRIGHT(mask);
            if (y == 0)
                SETDOWN(mask);
            else if (y == lbmparameters.size.y - 1)
                SETUP(mask);
            // write back mask
            grid[idx].mask = mask;
            grid[idx + offset].mask = mask;
        }
    }
}

#endif