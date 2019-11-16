#include "../host/inc/lbm.h"

#pragma OPENCL EXTENSION cl_intel_channels : enable

// constant define
constant Float2 e[Q] = {{0, 0},
                        {1, 0},
                        {0, 1},
                        {-1, 0},
                        {0, -1},
                        {1, 1},
                        {-1, 1},
                        {-1, -1},
                        {1, -1}};

constant Float w[Q] = {4.0 / 9, 1.0 / 9, 1.0 / 9,
                       1.0 / 9, 1.0 / 9, 1.0 / 36,
                       1.0 / 36, 1.0 / 36, 1.0 / 36};

// channel define
typedef struct
{
    int x;
    int y;
    float rho;
    Float2 u;
    Float f[Q];
} Cell;

typedef struct
{
    int x;
    int y;
    float rho;
    Float2 u;
    Float2 u_edge;
    Float f[Q];
} Cell_edge;

typedef struct
{
    int x;
    int y;
    Float2 u;
} U;

channel Cell_edge CELL_EDGE_ENTRY_CHANNEL __attribute__((depth(1024)));
channel Cell CELL_ENTRY_CHANNEL __attribute__((depth(1024)));
channel U CELL_U_CHANNEL __attribute__((depth(1024)));

// functions
Float dot(Float2 a, Float2 b)
{
    return a.x * b.x + a.y * b.y;
}

Float feq(
    Float rho,
    Float w,
    Float2 e,
    Float2 u)
{
    return rho * w * (1.0f + 3.0f * dot(e, u) + 4.5f * dot(e, u) * dot(e, u) - 1.5f * dot(u, u));
}

// Bhatnagar-Gross-Kroop approximation collision operator
Float bgk(
    Float f,
    Float rho,
    Float2 u,
    Float w,
    Float2 e,
    Float tau)
{
    //Without gravitational drag
    return f - (f - feq(rho, w, e, u)) * tau;
}

// Bhatnagar-Gross-Kroop approximation collision operator
Float bgk_edge(
    Float f,
    Float rho,
    Float2 u,
    Float2 u_edge,
    Float w,
    Float2 e)
{
    //Without gravitational drag
    return feq(rho, w, e, u_edge) + (f - feq(rho, w, e, u));
}

int getIndex(x, y, nx)
{
    return x + y * nx;
}
int getIndexi(x, y, i, nx)
{
    return (x + y * nx) * Q + i;
}

__kernel void lbm(__global float *restrict f,
                  int nx, int ny, float vx, float vy)
{
    for (int x = 1; x < nx - 1; x++)
    {
        for (int y = 1; y < ny - 1; y++)
        {
            // compute index
            int idxi = getIndexi(x, y, 0, nx);
            // get f from global_mem and compute u and rho
            Cell cell;
            cell.x = x;
            cell.y = y;
            cell.u.x = 0.f;
            cell.u.y = 0.f;
            cell.rho = 0.f;
#pragma unroll
            for (int i = 0; i < Q; i++)
            {
                cell.f[i] = f[idxi + i];
                cell.rho += cell.f[i];
                cell.u.x += e[i].x * cell.f[i];
                cell.u.y += e[i].y * cell.f[i];
            }
            cell.u.x /= cell.rho;
            cell.u.y /= cell.rho;
            // write data to channel
            write_channel_intel(CELL_ENTRY_CHANNEL, cell);

            // process edge
            if (x == 1 || x == nx - 2)
            {
                int x_edge = x - 1;
                int y_edge = y;
                Float2 u_edge = {0.f, 0.f};
                if (x == nx - 2)
                    x_edge = x + 1;

                Cell edge_cell;
                edge_cell.x = x_edge;
                edge_cell.y = y_edge;
                edge_cell.u = u_edge;
                edge_cell.rho = cell.rho;
                int idxi_edge = getIndexi(x_edge, y_edge, 0, nx);
//get edge_f from global_mem
#pragma unroll
                for (int i = 0; i < Q; i++)
                {
                    edge_cell.f[i] = f[idxi_edge + i];
                }
                //write data to channel,compute collision next f
                write_channel_intel(CELL_ENTRY_CHANNEL, edge_cell);

                Cell_edge cell_edge;
                cell_edge.x = x_edge;
                cell_edge.y = y_edge;
                cell_edge.u_edge = u_edge;
                cell_edge.rho = cell.rho;
                cell_edge.u = cell.u;
#pragma unroll
                for (int i = 0; i < Q; i++)
                {
                    cell_edge.f[i] = cell.f[i];
                }
                //write data to channel,compute collision next f
                write_channel_intel(CELL_EDGE_ENTRY_CHANNEL, cell_edge);
            }
            // process edge
            if (y == 1 || y == ny - 2)
            {
                int x_edge = x;
                int y_edge = y - 1;
                Float2 u_edge = {0.f, 0.f};
                if (y == ny - 2)
                {
                    y_edge = y + 1;
                    u_edge.x = vx;
                    u_edge.y = vy;
                }

                Cell edge_cell;
                edge_cell.x = x_edge;
                edge_cell.y = y_edge;
                edge_cell.u = u_edge;
                edge_cell.rho = cell.rho;
                int idxi_edge = getIndexi(x_edge, y_edge, 0, nx);
//get edge_f from global_mem
#pragma unroll
                for (int i = 0; i < Q; i++)
                {
                    edge_cell.f[i] = f[idxi_edge + i];
                }
                //write data to channel,compute collision next f
                write_channel_intel(CELL_ENTRY_CHANNEL, edge_cell);

                Cell_edge cell_edge;
                cell_edge.x = x_edge;
                cell_edge.y = y_edge;
                cell_edge.u_edge = u_edge;
                cell_edge.rho = cell.rho;
                cell_edge.u = cell.u;
#pragma unroll
                for (int i = 0; i < Q; i++)
                {
                    cell_edge.f[i] = cell.f[i];
                }
                //write data to channel,compute collision next f
                write_channel_intel(CELL_EDGE_ENTRY_CHANNEL, cell_edge);
            }
            // process edge
            if (x == 1 && y == ny - 2)
            {
                int x_edge = x - 1;
                int y_edge = y + 1;
                Float2 u_edge = {vx, vy};

                Cell edge_cell;
                edge_cell.x = x_edge;
                edge_cell.y = y_edge;
                edge_cell.u = u_edge;
                edge_cell.rho = cell.rho;
                int idxi_edge = getIndexi(x_edge, y_edge, 0, nx);
//get edge_f from global_mem
#pragma unroll
                for (int i = 0; i < Q; i++)
                {
                    edge_cell.f[i] = f[idxi_edge + i];
                }
                //write data to channel,compute collision next f
                write_channel_intel(CELL_ENTRY_CHANNEL, edge_cell);

                Cell_edge cell_edge;
                cell_edge.x = x_edge;
                cell_edge.y = y_edge;
                cell_edge.u_edge = u_edge;
                cell_edge.rho = cell.rho;
                cell_edge.u = cell.u;
#pragma unroll
                for (int i = 0; i < Q; i++)
                {
                    cell_edge.f[i] = cell.f[i];
                }
                //write data to channel,compute collision next f
                write_channel_intel(CELL_EDGE_ENTRY_CHANNEL, cell_edge);
            }
            // process edge
            if (x == nx - 2 && y == ny - 2)
            {
                int x_edge = x + 1;
                int y_edge = y + 1;
                Float2 u_edge = {vx, vy};
                Cell edge_cell;
                edge_cell.x = x_edge;
                edge_cell.y = y_edge;
                edge_cell.u = u_edge;
                edge_cell.rho = cell.rho;
                int idxi_edge = getIndexi(x_edge, y_edge, 0, nx);
//get edge_f from global_mem
#pragma unroll
                for (int i = 0; i < Q; i++)
                {
                    edge_cell.f[i] = f[idxi_edge + i];
                }
                //write data to channel,compute collision next f
                write_channel_intel(CELL_ENTRY_CHANNEL, edge_cell);

                Cell_edge cell_edge;
                cell_edge.x = x_edge;
                cell_edge.y = y_edge;
                cell_edge.u_edge = u_edge;
                cell_edge.rho = cell.rho;
                cell_edge.u = cell.u;
#pragma unroll
                for (int i = 0; i < Q; i++)
                {
                    cell_edge.f[i] = cell.f[i];
                }
                //write data to channel,compute collision next f
                write_channel_intel(CELL_EDGE_ENTRY_CHANNEL, cell_edge);
            }
            // process edge
            if (x == 1 && y == 1)
            {
                int x_edge = x - 1;
                int y_edge = y - 1;
                Float2 u_edge = {0.f, 0.f};
                Cell edge_cell;
                edge_cell.x = x_edge;
                edge_cell.y = y_edge;
                edge_cell.u = u_edge;
                edge_cell.rho = cell.rho;
                int idxi_edge = getIndexi(x_edge, y_edge, 0, nx);
//get edge_f from global_mem
#pragma unroll
                for (int i = 0; i < Q; i++)
                {
                    edge_cell.f[i] = f[idxi_edge + i];
                }
                //write data to channel,compute collision next f
                write_channel_intel(CELL_ENTRY_CHANNEL, edge_cell);

                Cell_edge cell_edge;
                cell_edge.x = x_edge;
                cell_edge.y = y_edge;
                cell_edge.u_edge = u_edge;
                cell_edge.rho = cell.rho;
                cell_edge.u = cell.u;
#pragma unroll
                for (int i = 0; i < Q; i++)
                {
                    cell_edge.f[i] = cell.f[i];
                }
                //write data to channel,compute collision next f
                write_channel_intel(CELL_EDGE_ENTRY_CHANNEL, cell_edge);
            }
            // process edge
            if (x == nx - 2 && y == 1)
            {
                int x_edge = x + 1;
                int y_edge = y - 1;
                Float2 u_edge = {0.f, 0.f};
                Cell edge_cell;
                edge_cell.x = x_edge;
                edge_cell.y = y_edge;
                edge_cell.u = u_edge;
                edge_cell.rho = cell.rho;
                int idxi_edge = getIndexi(x_edge, y_edge, 0, nx);
//get edge_f from global_mem
#pragma unroll
                for (int i = 0; i < Q; i++)
                {
                    edge_cell.f[i] = f[idxi_edge + i];
                }
                //write data to channel,compute collision next f
                write_channel_intel(CELL_ENTRY_CHANNEL, edge_cell);

                Cell_edge cell_edge;
                cell_edge.x = x_edge;
                cell_edge.y = y_edge;
                cell_edge.u_edge = u_edge;
                cell_edge.rho = cell.rho;
                cell_edge.u = cell.u;
#pragma unroll
                for (int i = 0; i < Q; i++)
                {
                    cell_edge.f[i] = cell.f[i];
                }
                //write data to channel,compute collision next f
                write_channel_intel(CELL_EDGE_ENTRY_CHANNEL, cell_edge);
            }
        }
    }
}

__attribute__((reqd_work_group_size(Q, 1, 1)))
__kernel void
collision(__global float *restrict f_next,
          float tau, int nx, int ny, char flag)
{
    int local_id = get_local_id(0);
    __local Cell local_cell[1];
    if (local_id == 0)
    {
        // get data from channel
        local_cell[0] = read_channel_intel(CELL_ENTRY_CHANNEL);
        // write to next channel
        if (flag == 1)
        {
            U ru = {local_cell[0].x, local_cell[0].y, local_cell[0].u};
            write_channel_intel(CELL_U_CHANNEL, ru);
        }
        // write u to global_mem
        // u[getIndex(local_cell[0].x,local_cell[0].y,nx)]=local_cell[0].u;
    }
    // wait the first work-item to load data fully
    barrier(CLK_LOCAL_MEM_FENCE);
    // propagation
    int new_x = local_cell[0].x + e[local_id].x;
    int new_y = local_cell[0].y + e[local_id].y;
    if (new_x > 0 && new_x < nx - 1 && new_y > 0 && new_y < ny - 1)
    {
        int new_id = getIndexi(new_x, new_y, local_id, nx);
        // compute new value;
        f_next[new_id] = bgk(local_cell[0].f[local_id], local_cell[0].rho, local_cell[0].u, w[local_id], e[local_id], tau);
    }
}

__attribute__((reqd_work_group_size(Q, 1, 1)))
__kernel void
edge(__global float *restrict f_next, int nx)
{
    int local_id = get_local_id(0);
    __local Cell_edge local_cell_edge[1];
    if (local_id == 0)
    {
        local_cell_edge[0] = read_channel_intel(CELL_EDGE_ENTRY_CHANNEL);
    }
    // wait the first work-item to load data fully
    barrier(CLK_LOCAL_MEM_FENCE);
    // compute edge new value;
    f_next[getIndexi(local_cell_edge[0].x, local_cell_edge[0].y, local_id, nx)] = bgk_edge(local_cell_edge[0].f[local_id], local_cell_edge[0].rho, local_cell_edge[0].u, local_cell_edge[0].u_edge, w[local_id], e[local_id]);
}
__kernel void writeU(__global Float2 *restrict u, int nx, int cell_num)
{
    while (cell_num > 0)
    {
        U ru = read_channel_intel(CELL_U_CHANNEL);
        u[getIndex(ru.x, ru.y, nx)] = ru.u;
        cell_num--;
    }
}