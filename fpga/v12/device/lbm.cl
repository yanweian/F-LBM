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
} CellVector;

typedef struct
{
    Cell cell;
    Vector2DIndex offset;
} CellWithBoundary;

typedef struct
{
    CellWithBoundary cellwbs[N_VECTOR][N_VECTOR];
} CellWithBoundaryVector;

typedef struct
{
    real f;
    Vector2DIndex position;
    int i;
} FWithPosition;

// channel define
channel char SINGALG2DATAR_CHAR_CHANNEL __attribute__((depth(1)));
channel CellVector DATAR2BD_CELLVECTOR_CHANNEL __attribute__((depth(8)));
channel char ISTREAMER2DATAR_CHAR_CHANNEL __attribute__((depth(1)));
channel char BSTREAMER2DATAR_CHAR_CHANNEL __attribute__((depth(1)));
channel CellVector BD2CU_CELLVECTOR_CHANNEL __attribute__((depth(8)));
channel CellVector CU2VW_CELLVECTOR_CHANNEL __attribute__((depth(8)));
channel real DR2CU_REAL_CHANNEL __attribute__((depth(1)));
channel CellWithBoundaryVector BD2BU_CWBV_CHANNEL __attribute__((depth(8)));
channel FWithPosition CU2ISTREAMER_FWP_CHANNEL __attribute__((depth(8)));
channel Cell BU2BSTREAMER_CELL_CHANNEL __attribute__((depth(8)));
channel char VW2SINGALG_CHAR_CHANNEL __attribute__((depth(1)));

// top level
#include "boundary_detector.cl"
#include "boundary_streamer.cl"
#include "bu.cl"
#include "cu.cl"
#include "data_reader.cl"
#include "interior_streamer.cl"
#include "signal_generator.cl"
#include "velocity_writer.cl"