#ifndef __DEFINES_H__
#define __DEFINES_H__
/**
 * define some fix value
*/

#define SUCCESS_EXIT 0
#define D_VECTOR 2
#define Q_VECTOR 9
#define REAL0 0.f
#define DEBUG false
#define N_VECTOR 2
// mask
// 00(0) 01(1) 10(-1) (0,1,-1)
#define UP 0x2    // (0,-1)0010
#define DOWN 0x1  // (0,1)0001
#define LEFT 0x4  // (1,0)0100
#define RIGHT 0x8 // (-1,0)1000

#define SETUP(x) x = (x | UP)
#define SETDOWN(x) x = (x | DOWN)
#define SETLEFT(x) x = (x | LEFT)
#define SETRIGHT(x) x = (x | RIGHT)

#define ISUP(x) ((x & UP) == UP)
#define ISLEFT(x) ((x & LEFT) == LEFT)
#define ISDOWN(x) ((x & DOWN) == DOWN)
#define ISRIGHT(x) ((x & RIGHT) == RIGHT)

#define SHIFT(x) (x << 4)

typedef float real;
typedef double dreal;
typedef int Index;
typedef int LongIndex;

// 2 dimension model
typedef struct
{
    real x;
    real y;
} Vector2DReal;

typedef struct
{
    Index x;
    Index y;
} Vector2DIndex;

// w and e
#if defined(__OPENCL__)
constant real w[Q_VECTOR] = {4.f / 9.f, 1.f / 9.f, 1.f / 9.f,
                             1.f / 9.f, 1.f / 9.f, 1.f / 36.f,
                             1.f / 36.f, 1.f / 36.f, 1.f / 36.f};
constant Vector2DIndex e[Q_VECTOR] = {{0, 0},
                                      {1, 0},
                                      {0, 1},
                                      {-1, 0},
                                      {0, -1},
                                      {1, 1},
                                      {-1, 1},
                                      {-1, -1},
                                      {1, -1}};
inline Vector2DIndex MakeVecIndex(Index x, Index y)
{
    Vector2DIndex vec = {x, y};
    return vec;
}

inline Vector2DReal MakeVecReal(real x, real y)
{
    Vector2DReal vec = {x, y};
    return vec;
}
inline real dotVecReal(Vector2DReal m, Vector2DReal n)
{
    return m.x * n.x + m.y * n.y;
}
inline real dotVecIndex_Real(Vector2DIndex m, Vector2DReal n)
{
    return m.x * n.x + m.y * n.y;
}
#else
const static real w[Q_VECTOR] = {4.f / 9.f, 1.f / 9.f, 1.f / 9.f,
                                 1.f / 9.f, 1.f / 9.f, 1.f / 36.f,
                                 1.f / 36.f, 1.f / 36.f, 1.f / 36.f};
const static Vector2DIndex e[Q_VECTOR] = {{0, 0},
                                          {1, 0},
                                          {0, 1},
                                          {-1, 0},
                                          {0, -1},
                                          {1, 1},
                                          {-1, 1},
                                          {-1, -1},
                                          {1, -1}};

inline Vector2DIndex MakeVec(Index x, Index y)
{
    Vector2DIndex vec = {x, y};
    return vec;
}

inline Vector2DReal MakeVec(real x, real y)
{
    Vector2DReal vec = {x, y};
    return vec;
}
inline real dotVec(Vector2DReal m, Vector2DReal n)
{
    return m.x * n.x + m.y * n.y;
}
inline real dotVec(Vector2DIndex m, Vector2DReal n)
{
    return m.x * n.x + m.y * n.y;
}

#endif

inline real computeVec(Vector2DReal v)
{
#if defined(__OPENCL__)
    return sqrt(dotVecReal(v, v));
#else
    return sqrt(dotVec(v, v));
#endif
}

inline LongIndex getIndex(Vector2DIndex position, Vector2DIndex size)
{
    return position.y * size.x + position.x;
}

inline real feq(real rho, Vector2DReal u, int i)
{
#if defined(__OPENCL__)
    real eu = dotVecIndex_Real(e[i], u);
    return rho * w[i] * (1.0 + 3.0 * eu + 4.5 * eu * eu - 1.5 * dotVecReal(u, u));
#else
    real eu = dotVec(e[i], u);
    return rho * w[i] * (1.0 + 3.0 * eu + 4.5 * eu * eu - 1.5 * dotVec(u, u));
#endif
}
// Bhatnagar-Gross-Kroop approximation collision operator
inline real bgk(real f, real rho, Vector2DReal u, int i, real tau_f)
{
    //Without gravitational drag
    return f - (f - feq(rho, u, i)) * tau_f;
}
inline real nfeq_boundary(real f, real rho, Vector2DReal u_interior, Vector2DReal u, int i)
{
    //Without gravitational drag
    return feq(rho, u, i) + f - feq(rho, u_interior, i);
}
#endif