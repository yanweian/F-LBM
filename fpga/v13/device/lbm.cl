// some opencl for fpga defines
#define USE_AUTORUN_KERNELS
#define EMULATOR
#if defined(USE_AUTORUN_KERNELS) && !defined(EMULATOR)
#define AUTORUN __attribute__((autorun))
#else
#define AUTORUN
#endif
#define TASK __attribute__((max_global_work_dim(0)))

#pragma OPENCL EXTENSION cl_intel_channels : enable
#define __OPENCL__
// some datas and funcs defines
#include "../host/inc/defines.h"
typedef struct
{
    Cell cells[N_VECTOR][N_VECTOR];
    char isboundary;
    LongIndex offset;
} CellVector;

// channel define
channel CellVector DATAR2BU_CELLVECTOR_CHANNEL;
channel CellVector BU2CU_CELLVECTOR_CHANNEL;
channel CellVector CU2VW_CELLVECTOR_CHANNEL;
channel CellVector DR2VW_CELLVECTOR_CHANNEL;

// top level
#include "bu.cl"
#include "cu.cl"
#include "data_reader.cl"
#include "velocity_writer.cl"