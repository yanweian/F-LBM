//
// Created by yan on 2019/7/8.
//
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

// 3D vector
typedef struct {
    Float x;
    Float y;
    Float z;
} Float3;

Float getV(Float3 u){
    return sqrt(u.x*u.x+u.y*u.y+u.z*u.z);
}
//// SIMULATION PARAMETERS

// Number of dimensions
const int n = 3;

// Grid dims
const unsigned int nx = 37;
const unsigned int ny = 37;
const unsigned int nz = 37;

// Grid cell width
const Float dx = 1.0;

const Float Lx=nx*dx;
// Number of flow vectors in each cell
const int m = 19;

// Time step length
//const double dt = 1.0;
const double dt = 1.0;

// Simulation end time
const double t_end = 5.0e3;

const double t_file = 1000;

//Reynolds number
const Float Re=1000.0;

const Float U = 0.1;


// Initial cell fluid density (dimensionless)
const Float rho0 = 1.0;

// Inital cell fluid velocity (dimensionless)
const Float3 u0 = {0.0, 0.0, 0.0};

// Inital cell fluid velocity 顶盖
const Float3 u1 = {U, 0, 0};
// Fluid dynamic viscosity
const Float nu = getV(u1)*Lx/Re;

//// FUNCTION DEFINITIONS



Float3 MAKE_FLOAT3(Float x, Float y, Float z)
{
    Float3 v;
    v.x = x; v.y = y; v.z = z;
    return v;
}
Float3 MAKE_FLOAT3(Float3 point)
{
    Float3 v;
    v.x = point.x; v.y = point.y; v.z = point.z;
    return v;
}
// Dot product of two Float3 vectors
Float dot(Float3 a, Float3 b)
{
    return a.x*b.x + a.y*b.y + a.z*b.z;
}

// Viscosity parameter
Float tau() {
    return (6.0*nu*dt/(dx*dx) + 1.0)/2.0;
}

// Get i-th value from cell x,y,z
unsigned int idx(
        const unsigned int x,
        const unsigned int y,
        const unsigned int z)
{
    return x + nx*y + nx*ny*z;
}

// Get i-th value from cell x,y,z
unsigned int idxi(
        const unsigned int x,
        const unsigned int y,
        const unsigned int z,
        const unsigned int i)
{
    return idx(x,y,z)*m+i;
}

// Get i-th weight
Float w(unsigned int i)
{
    if (n == 3 && m == 19) {
        if (i == 0)
            return 1.0/3.0;
        else if (i > 0 && i < 7)
            return 1.0/18.0;
        else
            return 1.0/36.0;
    } else {
        fprintf(stderr, "Error in w: m = %d != 19", m);
        fprintf(stderr, ", n = %d != 3\n", n);
        exit(EXIT_FAILURE);
    }
}

void set_e_values(Float3 *e)
{
    if (n == 3 && m == 19) {
        e[0]  = MAKE_FLOAT3( 0.0, 0.0, 0.0); // zero vel.
        e[1]  = MAKE_FLOAT3( 1.0, 0.0, 0.0); // face +x
        e[2]  = MAKE_FLOAT3(-1.0, 0.0, 0.0); // face -x
        e[3]  = MAKE_FLOAT3( 0.0, 1.0, 0.0); // face +y
        e[4]  = MAKE_FLOAT3( 0.0,-1.0, 0.0); // face -y
        e[5]  = MAKE_FLOAT3( 0.0, 0.0, 1.0); // face +z
        e[6]  = MAKE_FLOAT3( 0.0, 0.0,-1.0); // face -z
        e[7]  = MAKE_FLOAT3( 1.0, 1.0, 0.0); // edge +x,+y
        e[8]  = MAKE_FLOAT3(-1.0,-1.0, 0.0); // edge -x,-y
        e[9]  = MAKE_FLOAT3(1.0, -1.0, 0.0); // edge -x,+y
        e[10] = MAKE_FLOAT3( -1.0,1.0, 0.0); // edge +x,-y
        e[11] = MAKE_FLOAT3( 1.0, 0.0, 1.0); // edge +x,+z
        e[12] = MAKE_FLOAT3(-1.0, 0.0,-1.0); // edge -x,-z
        e[13] = MAKE_FLOAT3( -1.0, 0.0, 1.0); // edge +y,+z
        e[14] = MAKE_FLOAT3(1.0, 0.0,-1.0); // edge -y,-z
        e[15] = MAKE_FLOAT3(0.0, 1.0, 1.0); // edge -x,+z
        e[16] = MAKE_FLOAT3( 0.0, -1.0,-1.0); // edge +x,-z
        e[17] = MAKE_FLOAT3( 0.0,1.0, -1.0); // edge -y,+z
        e[18] = MAKE_FLOAT3( 0.0, -1.0,1.0); // edge +y,-z
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
        Float3 e,
        Float3 u)
{
    Float c2 = dx/dt;
    Float eu=dot(e,u);
    Float uu=dot(u,u);
    return rho*w * (1.0 + 3.0/c2*eu
                    + 9.0/(2.0*c2*c2)*eu*eu
                    - 3.0/(2.0*c2)*uu);
}

// Initialize cell densities, velocities, and flow vectors
void init_rho_v(Float* rho, Float3* u)
{
    unsigned int x, y, z;

    for (y=0; y<ny; y++) {
        for (x=0; x<nx; x++) {
            for (z=0; z<nz; z++) {
                // Set velocity to u0
                u[idx(x,y,z)] = MAKE_FLOAT3(u0);

                // Set density to rho0
                rho[idx(x,y,z)] = rho0;
            }
            u[idx(x,y,nz-1)] = MAKE_FLOAT3(u1);
        }
    }
}

void init_f(Float* f, Float* f_new, Float* rho, Float3* u, Float3* e)
{
    unsigned int x, y, z, i;
    Float f_val;

    for (z=0; z<nz; z++) {
        for (y=0; y<ny; y++) {
            for (x=0; x<nx; x++) {
                for (i=0; i<m; i++) {

                    int idx_=idx(x,y,z);
                    int idxi_=idxi(x,y,z,i);
                    // Set fluid flow vectors to v0
                    f_val = feq(rho[idx_], w(i), e[i], u[idx_]);
                    f[idxi_] = f_val;
                    f_new[idxi_] = f_val;
                }
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
        Float3 e,
        Float3 u)
{
    //Without gravitational drag
    return f - (f - feq(rho, w, e, u))/tau;
}
// Bhatnagar-Gross-Kroop approximation collision operator
Float bgk_edge(
        Float fb,
        Float rho,
        Float w,
        Float3 e,
        Float3 uo,
        Float3 ub)
{
    //Without gravitational drag
    return feq(rho, w, e, uo) + (fb - feq(rho, w, e, ub));
}

void collide(
        Float* f,
        Float* f_new,
        Float* rho,
        Float3* u,
        const Float3* e){

    unsigned int x, y,z, i;

    for (y=0; y<ny; y++) {
        for (x=0; x<nx; x++) {
            for(z=0;z<nz;z++){
                int idx_ = idx(x,y,z);
                for (i=0; i<m; i++) {
                    int idxi_ = idxi(x,y,z,i);
                    f_new[idxi_] = bgk(f[idxi_], tau(), rho[idx_],
                                       w(i), e[i], u[idx_]);
                }
            }
        }
    }
}
void stream(
        Float* f,
        const Float* f_new,
        Float* rho,
        Float3* u,
        const Float3* e){
    unsigned int x, y,z,i,xb,yb,zb;

    for (y=1; y<ny-1; y++) {
        for (x=1; x<nx-1; x++) {
            for(z=1;z<nz-1;z++){
                Float rho_new=0;
                Float3 u_new=MAKE_FLOAT3(0,0,0);
                for(i=0;i<m;i++){
                    xb=x-(unsigned int)e[i].x;
                    yb=y-(unsigned int)e[i].y;
                    zb=z-(unsigned int)e[i].z;

                    f[idxi(x,y,z,i)]=f_new[idxi(xb,yb,zb,i)];

                    rho_new+=f[idxi(x,y,z,i)];

                    u_new.x+=e[i].x*f[idxi(x,y,z,i)];
                    u_new.y+=e[i].y*f[idxi(x,y,z,i)];
                    u_new.z+=e[i].z*f[idxi(x,y,z,i)];
                }
//            return;
                int idx_=idx(x,y,z);
                rho[idx_]=rho_new;
                u[idx_]=MAKE_FLOAT3(u_new.x/rho_new,u_new.y/rho_new,u_new.z/rho_new);
            }

        }
    }

    for(y=1;y<ny-1;y++){
        for(z=0;z<nz;z++){
            x=0;xb=x+1;yb=y;zb=z;
            rho[idx(x,y,z)]=rho[idx(xb,yb,zb)];

            for(i=0;i<m;i++){
                f[idxi(x,y,z,i)]=bgk_edge(f[idxi(xb,yb,zb,i)],rho[idx(xb,yb,zb)],w(i),e[i],u[idx(x,y,z)],u[idx(xb,yb,zb)]);
            }

            x=nx-1;xb=x-1;
            rho[idx(x,y,z)]=rho[idx(xb,yb,zb)];

            for(i=0;i<m;i++){
                f[idxi(x,y,z,i)]=bgk_edge(f[idxi(xb,yb,zb,i)],rho[idx(xb,yb,zb)],w(i),e[i],u[idx(x,y,z)],u[idx(xb,yb,zb)]);
            }
        }
    }
    for(x=1;x<nx-1;x++){
        for(z=1;z<nz-1;z++){
            y=0;xb=x;yb=y+1;zb=z;
            rho[idx(x,y,z)]=rho[idx(xb,yb,zb)];

            for(i=0;i<m;i++){
                f[idxi(x,y,z,i)]=bgk_edge(f[idxi(xb,yb,zb,i)],rho[idx(xb,yb,zb)],w(i),e[i],u[idx(x,y,z)],u[idx(xb,yb,zb)]);
            }

            y=ny-1;yb=y-1;
            rho[idx(x,y,z)]=rho[idx(xb,yb,zb)];

            for(i=0;i<m;i++){
                f[idxi(x,y,z,i)]=bgk_edge(f[idxi(xb,yb,zb,i)],rho[idx(xb,yb,zb)],w(i),e[i],u[idx(x,y,z)],u[idx(xb,yb,zb)]);
            }
        }
    }
    for(x=0;x<nx;x++){
        for(y=0;y<ny;y++){
            z=0;xb=x;yb=y;zb=z+1;
            rho[idx(x,y,z)]=rho[idx(xb,yb,zb)];

            for(i=0;i<m;i++){
                f[idxi(x,y,z,i)]=bgk_edge(f[idxi(xb,yb,zb,i)],rho[idx(xb,yb,zb)],w(i),e[i],u[idx(x,y,z)],u[idx(xb,yb,zb)]);
            }

            z=nz-1;zb=z-1;
            rho[idx(x,y,z)]=rho[idx(xb,yb,zb)];
            u[idx(x,y,z)]=MAKE_FLOAT3(u1);
            for(i=0;i<m;i++){
                f[idxi(x,y,z,i)]=bgk_edge(f[idxi(xb,yb,zb,i)],rho[idx(xb,yb,zb)],w(i),e[i],u[idx(x,y,z)],u[idx(xb,yb,zb)]);
            }
        }
    }

}

// Swap Float pointers
void swapFloats(Float* a, Float* b)
{
    Float* tmp = a;
    a = b;
    b = tmp;
}

void output(double m, Float3* u) {
    ostringstream name;
    name << "cavity_" << m << ".dat";
    ofstream out(name.str().c_str());
    out << "Title= \"LBM Lid Driven Flow\"\n" << "VARIABLES= \"X\",\"Y\",\"Z\",\"U\",\"V\",\"UZ\"\n" << "ZONE T= \"BOX\",I= " << nx
        << ",J= " << ny<< ",K= " << nz << ",F= POINT" << endl;
    for (int y = 0; y < ny; y++) {
        for (int x = 0; x < nx; x++) {
            for (int z = 0; z < nz; z++) {
                out << double(x) / (nx-1) << " " << double(y) / (ny-1)<< " " << double(z) / (nz-1) << " " << u[idx(x,y,z)].x << " " << u[idx(x,y,z)].y<< " " << u[idx(x,y,z)].z << endl;
            }
        }
    }
}

int main(int argc, char** argv)
{
    int debug=0;
    printf("### Lattice-Boltzman D%dQ%d test ###\n", n, m);

    // Set cell flow vector values
    Float3 e[m]; set_e_values(e);

    // Particle distributions
    unsigned int ncells = nx*ny*nz;
    Float* f = (Float*)malloc(ncells*m*sizeof(Float));
    Float* f_new = (Float*)malloc(ncells*m*sizeof(Float));

    // Cell densities
    Float* rho = (Float*)malloc(ncells*sizeof(Float));

    // Cell flow velocities
    Float3* u = (Float3*)malloc(ncells*sizeof(Float3));

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
        stream(f, f_new,rho,u,e);

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
    //fprintf(stdout, "rho\n");
    //print_rho(stdout, rho);
    //fprintf(stdout, "u\n");
    //print_u(stdout, u);

    // Clear memory
    free(f);
    free(f_new);
    free(rho);
    free(u);

    return EXIT_SUCCESS;
}