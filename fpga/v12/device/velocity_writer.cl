TASK kernel void velocity_writer(
    global Vector2DReal *restrict u,
    // Vector2DIndex groupsize, // groupsize*{N_VECTOR,N_VECTOR}
    Index groupsize_x,
    Index groupsize_y,
    int num_interval // 迭代多少次
)
{
    Vector2DIndex groupsize = MakeVecIndex(groupsize_x, groupsize_y);
    Index num_group = groupsize.x * groupsize.y;
    Index size_x = groupsize.x * N_VECTOR;
    int interval = 0;
    int m = 0;
#pragma ivdep
    while (true)
    {
        CellVector cell_vector = read_channel_intel(CU2VW_CELLVECTOR_CHANNEL);
        if (interval == num_interval - 1)
        {
            // compute group start index
            LongIndex groupstart = cell_vector.cells[0][0].position.y * size_x + cell_vector.cells[0][0].position.x;
            // write velocity to ddr
            int i = 0;
            int j = 0;
            LongIndex jstart = groupstart + j * size_x;
#pragma unroll
            for (int n = 0; n < N_VECTOR * N_VECTOR; n++)
            {
                // write u to global_mem
                u[jstart + i] = cell_vector.cells[j][i].u;
                if (DEBUG)
                    printf("velocity_writer u: cells(%d,%d)(%d).u.x=%f\n", cell_vector.cells[j][i].position.x, cell_vector.cells[j][i].position.y, jstart + i, u[jstart + i].x);

                if (++i == N_VECTOR)
                {
                    i = 0;
                    j++;
                    jstart = groupstart + j * size_x;
                }
            }
            // #pragma unroll
            //             for (int j = 0; j < N_VECTOR; j++)
            //             {
            //                 LongIndex jstart = groupstart + j * size_x;
            // #pragma unroll
            //                 for (int i = 0; i < N_VECTOR; i++)
            //                 {
            //                     u[jstart + i] = cell_vector.cells[j][i].u;
            //                     if (DEBUG)
            //                         printf("velocity_writer u: cells(%d,%d)(%d).u.x=%f\n", cell_vector.cells[j][i].position.x, cell_vector.cells[j][i].position.y, jstart + i, u[jstart + i].x);
            //                 }
            //             }
        }
        if (++m == num_group)
        {
            m = 0;
            if (++interval == num_interval)
            {
                // 发送信号给主机
                write_channel_intel(VW2SINGALG_CHAR_CHANNEL, '1');
                interval = 0;
            }
        }
        if (DEBUG)
            printf("velocity_writer: read from cu group %d,interval=%d\n", m, interval);
    }
}