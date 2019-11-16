/**
detecte boundary condition
*/
TASK AUTORUN kernel void boundary_detector()
{
    while (true)
    {
        // get cellvector.cells[N_Vector][N_Vector] from dr
        CellVector cellvector = read_channel_intel(DATAR2BD_CELLVECTOR_CHANNEL);
        if (DEBUG)
            printf("boundary_detector getting Group data...\n");

        // define boundary datas
        CellWithBoundaryVector cellwbvector;
        Vector2DIndex offsets[N_VECTOR][N_VECTOR];

        // flag for mask the boundary cell
        bool flag = false;
        int i = 0;
        int j = 0;
#pragma unroll
        for (int n = 0; n < N_VECTOR * N_VECTOR; n++)
        {
            Vector2DIndex offset = MakeVecIndex(0, 0);
            Cell cell = cellvector.cells[j][i];
            // get offset(x,y)
            char mask = cellvector.cells[j][i].mask;
            if (ISUP(mask))
                offset.y = -1;
            else if (ISDOWN(mask))
                offset.y = 1;
            if (ISRIGHT(mask))
                offset.x = -1;
            else if (ISLEFT(mask))
                offset.x = 1;
            if (DEBUG)
                printf("boundary_detector: cells(%d,%d).offset=(%d,%d)\n", cell.position.x, cell.position.y, offset.x, offset.y);
            // if has offset mask
            if (mask != 0x0)
            {
                flag = true;
                // set boundary.rho
                cellvector.cells[j][i].rho = cellvector.cells[j + offset.y][i + offset.x].rho;
            }
            offsets[j][i] = offset;
            // compute index(j,i)
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
        //                 Vector2DIndex offset = MakeVecIndex(0, 0);
        //                 Cell cell = cellvector.cells[j][i];
        //                 // get offset(x,y)
        //                 char mask = cellvector.cells[j][i].mask;
        //                 if (ISUP(mask))
        //                     offset.y = -1;
        //                 else if (ISDOWN(mask))
        //                     offset.y = 1;
        //                 if (ISRIGHT(mask))
        //                     offset.x = -1;
        //                 else if (ISLEFT(mask))
        //                     offset.x = 1;
        //                 if (DEBUG)
        //                     printf("boundary_detector: cells(%d,%d).offset=(%d,%d)\n", cell.position.x, cell.position.y, offset.x, offset.y);
        //                 // if has offset mask
        //                 if (mask != 0x0)
        //                 {
        //                     flag = true;
        //                     cellvector.cells[j][i].rho = cellvector.cells[j + offset.y][i + offset.x].rho;
        //                 }
        //                 offsets[j][i] = offset;
        //             }
        //         }

        // transfer cellvector.cells[N_Vector][N_Vector] to cu
        write_channel_intel(BD2CU_CELLVECTOR_CHANNEL, cellvector);

        if (flag)
        {
            int j = 0;
            int i = 0;
#pragma unroll
            for (int n = 0; n < N_VECTOR * N_VECTOR; n++)
            {
                cellwbvector.cellwbs[j][i].offset = offsets[j][i];
                cellwbvector.cellwbs[j][i].cell = cellvector.cells[j][i];
                if (++i == N_VECTOR)
                {
                    i = 0;
                    j++;
                }
            }
            // #pragma unroll
            //             for (int j = 0; j < N_VECTOR; j++)
            //             {
            // #pragma unroll
            //                 for (int i = 0; i < N_VECTOR; i++)
            //                 {
            //                     cellwbvector.cellwbs[j][i].offset = offsets[j][i];
            //                     cellwbvector.cellwbs[j][i].cell = cellvector.cells[j][i];
            //                 }
            //             }
            // transfer data to bu
            write_channel_intel(BD2BU_CWBV_CHANNEL, cellwbvector);
        }
    }
}