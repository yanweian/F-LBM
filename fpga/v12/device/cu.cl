/**
this code is for colliding parallel */
// this function is the prototype code for each cu kernels in N_VECTOR cu arrays
AUTORUN TASK kernel void cu()
{
    real tau_f = read_channel_intel(DR2CU_REAL_CHANNEL);
    while (true)
    {
        if (DEBUG)
            printf("Cu getting data...\n");
        CellVector cellvector = read_channel_intel(BD2CU_CELLVECTOR_CHANNEL);
        write_channel_intel(CU2VW_CELLVECTOR_CHANNEL, cellvector);
        FWithPosition fwithposition[N_VECTOR * N_VECTOR * Q_VECTOR];
        int findex = 0;
        int i = 0;
        int j = 0;
#pragma unroll
        for (int n = 0; n < N_VECTOR * N_VECTOR; n++)
        {
            Cell cell = cellvector.cells[j][i];
#pragma unroll
            for (int m = 0; m < Q_VECTOR; m++)
            {
                int new_x = cell.position.x + e[m].x;
                int new_y = cell.position.y + e[m].y;
                FWithPosition ftmp = {bgk(cell.f[m], cell.rho, cell.u, m, tau_f), MakeVecIndex(new_x, new_y), m};
                fwithposition[findex++] = ftmp;
            }
            if (++i == N_VECTOR)
            {
                i = 0;
                j++;
            }
        }
        // #pragma unroll
        //         for (int j = 0; j < N_VECTOR; j++)
        //         {
        // #pragma unroll
        //             for (int i = 0; i < N_VECTOR; i++)
        //             {
        //                 Cell cell = cellvector.cells[j][i];
        // #pragma unroll
        //                 for (int m = 0; m < Q_VECTOR; m++)
        //                 {
        //                     int new_x = cell.position.x + e[m].x;
        //                     int new_y = cell.position.y + e[m].y;
        //                     FWithPosition ftmp = {bgk(cell.f[m], cell.rho, cell.u, m, tau_f), MakeVecIndex(new_x, new_y), m};
        //                     fwithposition[findex++] = ftmp;
        //                 }
        //             }
        //         }
        for (int i = 0; i < N_VECTOR * N_VECTOR * Q_VECTOR; i++)
        {
            write_channel_intel(CU2ISTREAMER_FWP_CHANNEL, fwithposition[i]);
        }
    }
}