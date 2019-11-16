/**
处理边界
数据 struct BoundaryWithInterior
{
    Vector2D position;
    Vector2D u;
    Cell cell_interior;
};*/
TASK AUTORUN kernel void bu()
{
    while (true)
    {
        CellWithBoundaryVector cellwbvector = read_channel_intel(BD2BU_CWBV_CHANNEL);
        int x = 0;
        int y = 0;
        for (int m = 0; m < N_VECTOR * N_VECTOR; m++)
        {
            Cell cell = cellwbvector.cellwbs[y][x].cell;
            if (cell.mask != 0x0)
            {
                Vector2DIndex offset = cellwbvector.cellwbs[y][x].offset;
                Cell cell_interior = cellwbvector.cellwbs[y + offset.y][x + offset.x].cell;
                for (int i = 0; i < Q_VECTOR; i++)
                {
                    cell.f[i] = nfeq_boundary(cell_interior.f[i], cell_interior.rho, cell_interior.u, cell.u, i);
                }
                write_channel_intel(BU2BSTREAMER_CELL_CHANNEL, cell);
            }

            //compute index
            if (++x == N_VECTOR)
            {
                x = 0;
                y++;
            }
        }
    }
}