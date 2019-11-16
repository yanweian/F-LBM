#include <iostream>
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <string>
#ifdef _WIN32 // Windows
#include <windows.h>
#else         // Linux
#include <stdio.h>
#include <unistd.h> // readlink, chdir
#endif
using namespace std;

// High-resolution timer.
double getCurrentTimestamp() {
#ifdef _WIN32 // Windows
    // Use the high-resolution performance counter.

    static LARGE_INTEGER ticks_per_second = {};
    if(ticks_per_second.QuadPart == 0) {
        // First call - get the frequency.
        QueryPerformanceFrequency(&ticks_per_second);
    }

    LARGE_INTEGER counter;
    QueryPerformanceCounter(&counter);

    double seconds = double(counter.QuadPart) / double(ticks_per_second.QuadPart);
    return seconds;
#else         // Linux
    timespec a;
  clock_gettime(CLOCK_MONOTONIC, &a);
  return (double(a.tv_nsec) * 1.0e-9) + double(a.tv_sec);
#endif
}


// Floating point precision
//typedef float Float;
typedef double Float;

// 2D vector
typedef struct {
    Float x;
    Float y;
} Float2;

//// SIMULATION PARAMETERS

// Number of dimensions
const int n = 2;

// Grid dims
//const unsigned int nx = 3;
//const unsigned int ny = 6;
//const unsigned int nz = 3;
const unsigned int nx = 257;
const unsigned int ny = 257;

// Grid cell width
const Float dx = 1.0;

// Grid length
const Float Lx=dx*nx;
const Float Ly=dx*ny;

// Number of flow vectors in each cell
const int m = 9;

// Time step length
const double dt = dx;

// Simulation end time
const double t_end = 5.0e3;

// save file interval
const double t_file = 1000;

//Reynolds number
const Float Re=1000.0;

const Float U = 0.1;

// Fluid dynamic viscosity
const Float nu = U*Lx/Re;

// Initial cell fluid density (dimensionless)
const Float rho0 = 1.0;

// Inital cell fluid velocity (dimensionless)
const Float2 u0 = {0.0, 0.0};

// Inital cell fluid velocity 顶盖
const Float2 u1 = {U, 0.0};


//// FUNCTION DEFINITIONS

Float2 MAKE_FLOAT2(Float x, Float y)
{
    Float2 v;
    v.x = x; v.y = y;
    return v;
}

// Dot product of two Float2 vectors
Float dot(Float2 a, Float2 b)
{
    return a.x*b.x + a.y*b.y;
}

// Viscosity parameter
Float tau() {
//    return (6.0*nu*dt/(dx*dx) + 1.0)/2.0;
    return 3.0*nu+0.5;
}

// Get i-th value from cell x,y
unsigned int idx(
        const unsigned int x,
        const unsigned int y)
{
    return x + nx*y;
}

// Get i-th value from cell x,y,i
unsigned int idxi(
        const unsigned int x,
        const unsigned int y,
        const unsigned int i)
{
    return x*m+nx*y*m+i;
}

// Get i-th weight
Float w(unsigned int i)
{
    if (n == 2 && m == 9) {
        if (i == 0)
            return 4.0/9.0;
        else if (i > 0 && i < 5)
            return 1.0/9.0;
        else
            return 1.0/36.0;
    } else {
        fprintf(stderr, "Error in w: m = %d != 19", m);
        fprintf(stderr, ", n = %d != 3\n", n);
        exit(EXIT_FAILURE);
    }
}

void set_e_values(Float2 *e)
{
    if (n == 2 && m == 9) {
        e[0]  = MAKE_FLOAT2( 0.0, 0.0); // zero vel.
        e[1]  = MAKE_FLOAT2( 1.0, 0.0); //  +x
        e[2]  = MAKE_FLOAT2( 0.0, 1.0); //  +y
        e[3]  = MAKE_FLOAT2( -1.0, 0.0); //  -x
        e[4]  = MAKE_FLOAT2( 0.0,-1.0); //  -y
        e[5]  = MAKE_FLOAT2( 1.0, 1.0); //  +x,+y
        e[6]  = MAKE_FLOAT2( -1.0, 1.0); //  -x,+y
        e[7]  = MAKE_FLOAT2( -1.0, -1.0); //  -x,-y
        e[8]  = MAKE_FLOAT2(1.0, -1.0); //  +x,-y
    } else {
        fprintf(stderr, "Error in set_e_values: m = %d != 19", m);
        fprintf(stderr, ", n = %d != 3\n", n);
        exit(EXIT_FAILURE);
    }
}

// Equilibrium distribution along flow vector e
Float feq(
        Float rho,
        Float w,
        Float2 e,
        Float2 u)
{
    Float c2 = dx/dt;
    return rho*w * (1.0 + 3.0/c2*dot(e,u)
                    + 9.0/(2.0*c2*c2)*dot(e,u)*dot(e,u)
                    - 3.0/(2.0*c2)*dot(u,u));
}

// Initialize cell densities, velocities, and flow vectors
void init_rho_v(Float* rho, Float2* u)
{
    unsigned int x, y;
    for (x=0; x<nx; x++) {
        for (y=0; y<ny; y++) {
            // Set velocity to u0
            u[idx(x,y)] = MAKE_FLOAT2(u0.x, u0.y);

            // Set density to rho0
            rho[idx(x,y)] = rho0;
        }
        //顶盖速度初始化
        u[idx(x,ny-1)] = MAKE_FLOAT2(u1.x, u1.y);
    }
}

void init_f(Float* f, Float* f_new, Float* rho, Float2* u, Float2* e)
{
    unsigned int x, y, i;
    Float f_val;
    for (y=0; y<ny; y++) {
        for (x=0; x<nx; x++) {
            for (i=0; i<m; i++) {
                // Set fluid flow vectors to v0
                f_val = feq(rho[idx(x,y)], w(i), e[i], u[idx(x,y)]);
                f[idxi(x,y,i)] = f_val;
                f_new[idxi(x,y,i)] = f_val;
            }
        }
    }

}

// Bhatnagar-Gross-Kroop approximation collision operator
Float bgk(
        Float f,
        Float tau,
        Float rho,
        Float w,
        Float2 e,
        Float2 u)
{
    //Without gravitational drag
    return f - (f - feq(rho, w, e, u))/tau;
}

// Bhatnagar-Gross-Kroop approximation collision operator
Float bgk_edge(
        Float fb,
        Float rho,
        Float w,
        Float2 e,
        Float2 uo,
        Float2 ub)
{
    //Without gravitational drag
    return feq(rho, w, e, uo) + (fb - feq(rho, w, e, ub));
}

// Cell fluid density
Float find_rho(
        const Float* f,
        const unsigned int x,
        const unsigned int y)
{
    int i;
    Float rho = 0.0;
    for (i=0; i<m; i++)
        rho += f[idxi(x,y,i)];
    return rho;
}

// Cell fluid velocity
Float2 find_u(
        const Float* f,
        const Float rho,
        const Float2* e,
        const unsigned int x,
        const unsigned int y)
{
    Float2 u = {0.0, 0.0};
    Float f_i;
    unsigned int i;
    for (i=0; i<m; i++) {
        f_i = f[idxi(x,y,i)];
        u.x += f_i*e[i].x;
        u.y += f_i*e[i].y;
    }
    u.x/=rho;
    u.y/=rho;
    return u;
}


void collide(
        Float* f,
        Float* f_new,
        Float* rho,
        Float2* u,
        const Float2* e){

    unsigned int x, y, i;

    for (y=0; y<ny; y++) {
        for (x=0; x<nx; x++) {
            int idx_ = idx(x,y);
            for (i=0; i<m; i++) {
                int idxi_ = idxi(x,y,i);
                f_new[idxi_] = bgk(f[idxi_], tau(), rho[idx_],
                               w(i), e[i], u[idx_]);
            }
        }
    }
}
void stream(
        Float* f,
        const Float* f_new,
        Float* rho,
        Float2* u,
        const Float2* e){
    unsigned int x, y,i,xb,yb;

    for (y=1; y<ny-1; y++) {
        for (x=1; x<nx-1; x++) {
            Float rho_new=0;
            Float2 u_new=MAKE_FLOAT2(0,0);
            for(i=0;i<m;i++){
                xb=x-(unsigned int)e[i].x;
                yb=y-(unsigned int)e[i].y;
                f[idxi(x,y,i)]=f_new[idxi(xb,yb,i)];
                rho_new+=f[idxi(x,y,i)];
                u_new.x+=e[i].x*f[idxi(x,y,i)];
                u_new.y+=e[i].y*f[idxi(x,y,i)];
            }
//            return;
            int idx_=idx(x,y);
            rho[idx_]=rho_new;
            u[idx_]=MAKE_FLOAT2(u_new.x/rho_new,u_new.y/rho_new);
        }
    }
    for(y=1;y<ny-1;y++){
        x=0;xb=x+1;yb=y;
        rho[idx(x,y)]=rho[idx(xb,yb)];
        u[idx(x,y)]=MAKE_FLOAT2(0,0);

        for(i=0;i<m;i++){
            f[idxi(x,y,i)]=bgk_edge(f[idxi(xb,yb,i)],rho[idx(xb,yb)],w(i),e[i],u[idx(x,y)],u[idx(xb,yb)]);
        }

        x=nx-1;xb=x-1;
        rho[idx(x,y)]=rho[idx(xb,yb)];
        u[idx(x,y)]=MAKE_FLOAT2(0,0);
        for(i=0;i<m;i++) {
            f[idxi(x,y,i)]=bgk_edge(f[idxi(xb,yb,i)],rho[idx(xb,yb)],w(i),e[i],u[idx(x,y)],u[idx(xb,yb)]);
        }
    }
    for(x=0;x<nx;x++){
        
        y=0;yb=y+1;xb=x;
        rho[idx(x,y)]=rho[idx(xb,yb)];
        u[idx(x,y)]=MAKE_FLOAT2(0,0);

        for(i=0;i<m;i++){
            f[idxi(x,y,i)]=bgk_edge(f[idxi(xb,yb,i)],rho[idx(xb,yb)],w(i),e[i],u[idx(x,y)],u[idx(xb,yb)]);
        }

        y=ny-1;yb=y-1;
        rho[idx(x,y)]=rho[idx(xb,yb)];
        u[idx(x,y)]=MAKE_FLOAT2(u1.x,u1.y);

        for(i=0;i<m;i++) {
            f[idxi(x,y,i)]=bgk_edge(f[idxi(xb,yb,i)],rho[idx(xb,yb)],w(i),e[i],u[idx(x,y)],u[idx(xb,yb)]);
        }
    }

}

void output(double m, Float2* u) {
    ostringstream name;
    name << "cavity_" << m << ".dat";
    ofstream out(name.str().c_str());
    out << "Title= \"LBM Lid Driven Flow\"\n" << "VARIABLES= \"X\",\"Y\",\"U\",\"V\"\n" << "ZONE T= \"BOX\",I= " << nx
        << ",J= " << ny << ",F= POINT" << endl;
    for (int y = 0; y < ny; y++) {
        for (int x = 0; x < nx; x++) {
            out << double(x) / Lx << " " << double(y) / Lx << " " << u[idx(x,y)].x << " " << u[idx(x,y)].y << endl;
        }
    }
}
int main(){
    int debug=0;
    printf("### Lattice-Boltzman D%dQ%d test ###\n", n, m);
    Float tau_f=tau();
    printf("tau=%f\n",tau_f);
    // Set cell flow vector values
    Float2 e[m]; set_e_values(e);

    // Particle distributions
    unsigned int ncells = nx*ny;
    Float* f = (Float*)malloc(ncells*m*sizeof(Float));
    Float* f_new = (Float*)malloc(ncells*m*sizeof(Float));

    // Cell densities
    Float* rho = (Float*)malloc(ncells*sizeof(Float));

    // Cell flow velocities
    Float2* u = (Float2*)malloc(ncells*sizeof(Float2));

    // Set densities, velocities and flow vectors
    init_rho_v(rho, u);
    init_f(f, f_new, rho, u, e);

    // Temporal loop
    double t;
    double t_file_elapsed = 0.0;

    const double start_time=getCurrentTimestamp();
    // Temporal loop
    for (t = 0.0; t < t_end; t += dt, t_file_elapsed += dt) {

        // Report time to stdout
        printf("\rt = %.1fs./%.1fs., %.1f%% done", t, t_end, t/t_end*100.0);

        // LBM collision and streaming
        collide(f,f_new, rho, u, e);
        stream(f,f_new,rho,u,e);
//        return 0;
        // Print x-z plane to file
        if (t_file_elapsed >= t_file) {
            output(t, u);
            t_file_elapsed = 0.0;
        }
    }
    const double end_time=getCurrentTimestamp();
    const double total_time = end_time - start_time;

    // Wall-clock time taken.
    printf("\nCPU Total Time: %0.3f ms\n", total_time * 1e3);
    printf("\n");

    // Report values to stdout
    // fprintf(stdout, "rho\n");
    // print_rho(stdout, rho);
    // fprintf(stdout, "u\n");
    // print_u(stdout, u);

    // Clear memory
    free(f);
    free(f_new);
    free(rho);
    free(u);

    return EXIT_SUCCESS;
}