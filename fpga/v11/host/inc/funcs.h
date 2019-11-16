#ifndef FUNCS_H
#define FUNCS_H

#include "param.h"
#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <iostream>
//// FUNCTION DEFINITIONS

Float2 MAKE_FLOAT2(Float x, Float y)
{
    Float2 v;
    v.x = x; v.y = y;
    return v;
}

Float2 MAKE_FLOAT2(Float2 point)
{
    Float2 v;
    v.x = point.x; v.y = point.y;
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

Float tau_f=1/tau();

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
    return x*Q+nx*y*Q+i;
}

// Get i-th weight
Float w(unsigned int i)
{
    if (D == 2 && Q == 9) {
        if (i == 0)
            return 4.0/9.0;
        else if (i > 0 && i < 5)
            return 1.0/9.0;
        else
            return 1.0/36.0;
    } else {
        fprintf(stderr, "Error in w: Q = %d != 19", Q);
        fprintf(stderr, ", D = %d != 3\n", D);
        exit(EXIT_FAILURE);
    }
}

void set_e_values(Float2 *e)
{
    if (D == 2 && Q == 9) {
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
        fprintf(stderr, "Error in set_e_values: Q = %d != 9", Q);
        fprintf(stderr, ", D = %d != 2\n", D);
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

void init_f(Float* f, Float* rho, Float2* u, Float2* e)
{
    unsigned int x, y, i;
    Float f_val;
    for (y=0; y<ny; y++) {
        for (x=0; x<nx; x++) {
            for (i=0; i<Q; i++) {
                // Set fluid flow vectors to v0
                f_val = feq(rho[idx(x,y)], w(i), e[i], u[idx(x,y)]);
                f[idxi(x,y,i)] = f_val;
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
    for (i=0; i<Q; i++)
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
    for (i=0; i<Q; i++) {
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
        const Float2* e){

    unsigned int x, y, i;

    for (y=0; y<ny; y++) {
        for (x=0; x<nx; x++) {
            int idx_ = idx(x,y);
            Float rhon=find_rho(f,x,y);
            Float2 un=find_u(f,rhon,e,x,y);
            for(int i=0;i<Q;i++){
                int idxi_=idxi(x,y,i);
                int idxi_new=idxi(x+e[i].x,y+e[i].y,i);
                if(idxi_new>=0&&idxi_new<ncells*Q){
                    f_new[idxi_new]=bgk(f[idxi_],tau_f,rhon,w(i),e[i],un);
                }
            }
        }
    }
}

void computeU(Float* f,Float2* u){
    unsigned int x, y, i;

    for (y=1; y<ny-1; y++) {
        for (x=1; x<nx-1; x++) {
            int idx_ = idx(x,y);
            Float rhon=find_rho(f,x,y);
            u[idx_]=find_u(f,rhon,e,x,y);
        }
    }
}
#endif
