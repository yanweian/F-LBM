// some opencl for fpga defines
#pragma OPENCL EXTENSION cl_intel_channels : enable
#define __OPENCL__
// some datas and funcs defines
#include "../host/inc/defines.h"
typedef struct
{
    Vector2DIndex position;
    real rho;
    Vector2DReal u;
    real f[Q_VECTOR];
} Cell;

typedef struct
{
    Vector2DIndex position; // edge position
    real rho;               // rho==rho_edge
    Vector2DReal u;         // iterior
    Vector2DReal u_edge;    // 边界
    real f[Q_VECTOR];       // Internal f
} CellEdge;

// channel define
channel Cell DATAR2CU_CELL_CHANNEL;
channel CellEdge DATAR2BU_CELLEDGE_CHANNEL;
channel Cell CU2VW_CELL_CHANNEL;
channel Cell DR2VW_CELLVECTOR_CHANNEL;

// top level
#include "data_reader.cl"
#include "bu.cl"
#include "cu.cl"
#include "velocity_writer.cl"