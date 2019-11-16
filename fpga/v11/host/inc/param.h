#ifndef PARAM_H
#define PARAM_H
#include "lbm.h"
// Grid dims
unsigned int nx = 257;
unsigned int ny = 257;
unsigned int ncells = nx*ny;

// Grid cell width
const Float dx = 1.0;

// Grid length
Float Lx=dx*nx;
Float Ly=dx*ny;

// Time step length
const double dt = dx;

// Simulation end time
double t_end = 5.0e3;

// save file interval
double t_file = 1000;

//Reynolds number
const Float Re=1000.0;

const Float U = 0.1;

// Fluid dynamic viscosity
Float nu = U*Lx/Re;

const Float rho0 = 1.0;

const Float2 u0 = {0.0, 0.0};

// Inital cell fluid velocity 顶盖
const Float2 u1 = {U, 0.0};

Float2 e[Q];
#endif