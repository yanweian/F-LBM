/**
read cell data from global buffer */
TASK kernel void data_reader(
    global Cell *restrict cells,
    Index groupsize_x,
    Index groupsize_y,
    // Vector2DIndex groupsize, //groupsize*{N_VECTOR,N_VECTOR}
    LongIndex ncells,
    real u0x, real u0y,
    real tau_f, int interval)
{
    // 将tau_f 参数传递给cus
    write_channel_intel(DR2CU_REAL_CHANNEL, tau_f);
    Vector2DIndex groupsize = MakeVecIndex(groupsize_x, groupsize_y);
    Index size_x = groupsize.x * N_VECTOR;
    // Vector2DIndex groupsize = MakeVecIndex(1, 1);
    bool buf_flag = false;
    int num_interval = interval;
    Index y = 0;
    Index x = 0;
    Index offset = 0;
    while (true)
    {
        // N_VECTOR * N_VECTOR vectorization : y * N_VECTOR * size_x + x * N_VECTOR
        LongIndex groupstart = (y * size_x + x) * N_VECTOR + offset;

        // ================start reader data==================
        CellVector buffer;
        int i = 0;
        int j = 0;
        LongIndex jstart = groupstart + j * size_x;
#pragma unroll
        for (int n = 0; n < N_VECTOR * N_VECTOR; n++)
        {
            // read data from global_mem
            buffer.cells[j][i] = cells[jstart + i];
            // compute rho and u
            if (buffer.cells[j][i].mask == 0x0) //normal point
            {
                real rho = 0.f;
                Vector2DReal u = {0.f, 0.f};
#pragma unroll
                for (int m = 0; m < Q_VECTOR; m++)
                {
                    rho += buffer.cells[j][i].f[m];
                    u.x += e[m].x * buffer.cells[j][i].f[m];
                    u.y += e[m].y * buffer.cells[j][i].f[m];
                }
                u.x /= rho;
                u.y /= rho;
                buffer.cells[j][i].rho = rho;
                buffer.cells[j][i].u = u;
            }
            else if (ISUP(buffer.cells[j][i].mask)) // drive point
            {
                buffer.cells[j][i].u = MakeVecReal(u0x, u0y);
                if (DEBUG)
                    printf("data_reader u: cells(%d,%d).u=%f\n", cells[jstart + i].position.x, cells[jstart + i].position.y, buffer.cells[j][i].u.x);
            }
            if (DEBUG)
                printf("data_reader: cells(%d,%d)=%d\n", cells[jstart + i].position.x, cells[jstart + i].position.y, jstart + i);
            // compute index(j,i)
            if (++i == N_VECTOR)
            {
                i = 0;
                j++;
                jstart = groupstart + j * size_x;
            }
        }
        // #pragma unroll
        //         for (int j = 0; j < N_VECTOR; j++)
        //         {
        //             LongIndex jstart = groupstart + j * size_x;
        // #pragma unroll
        //             for (int i = 0; i < N_VECTOR; i++)
        //             {
        //                 buffer.cells[j][i] = cells[jstart + i];
        //                 // compute rho and u
        //                 if (buffer.cells[j][i].mask == 0x0)
        //                 {
        //                     real rho = 0.f;
        //                     Vector2DReal u = {0.f, 0.f};
        // #pragma unroll
        //                     for (int m = 0; m < Q_VECTOR; m++)
        //                     {
        //                         rho += buffer.cells[j][i].f[m];
        //                         u.x += e[m].x * buffer.cells[j][i].f[m];
        //                         u.y += e[m].y * buffer.cells[j][i].f[m];
        //                     }
        //                     u.x /= rho;
        //                     u.y /= rho;
        //                     buffer.cells[j][i].rho = rho;
        //                     buffer.cells[j][i].u = u;
        //                 }
        //                 else if (ISUP(buffer.cells[j][i].mask))
        //                 {
        //                     buffer.cells[j][i].u = MakeVecReal(u0x, u0y);
        //                     if (DEBUG)
        //                         printf("data_reader u: cells(%d,%d).u=%f\n", cells[jstart + i].position.x, cells[jstart + i].position.y, buffer.cells[j][i].u.x);
        //                 }
        //                 if (DEBUG)
        //                     printf("data_reader: cells(%d,%d)=%d\n", cells[jstart + i].position.x, cells[jstart + i].position.y, jstart + i);
        //             }
        //         }
        // ================end reader data==================

        // transfer data to boundary_detector
        write_channel_intel(DATAR2BD_CELLVECTOR_CHANNEL, buffer);
        if (DEBUG)
            printf("data_reader:transfer group(%d,%d) to bd\n", x, y);

        // compute index
        if (++x == groupsize.x)
        {
            x = 0;
            if (++y == groupsize.y)
            {
                y = 0;
                // receive next iterate signal from streamer
                read_channel_intel(ISTREAMER2DATAR_CHAR_CHANNEL);
                if (DEBUG)
                    printf("data_reader: get signal from Is\n");
                read_channel_intel(BSTREAMER2DATAR_CHAR_CHANNEL);
                if (DEBUG)
                    printf("data_reader: get signal from Bs\n");
                // receive interval from signal
                if (--num_interval == 0)
                {
                    num_interval = interval;
                    read_channel_intel(SINGALG2DATAR_CHAR_CHANNEL);
                }
                // ping-pong buffer, using offset
                offset = ncells;
                if (buf_flag)
                    offset = 0;
                buf_flag = !buf_flag;
                if (DEBUG)
                {
                    printf("data_reader: interval= %d\n", num_interval);
                    printf("data_reader: offset= %d\n", offset);
                }
            }
        }
    }
}