// __attribute__((reqd_work_group_size(Q_VECTOR, 1, 1)))
// kernel void
// bu(global real *restrict fs, Index size_x)
// {
//     int item_id_0 = get_local_id(0);
//     __local CellEdge local_cell_edge[1];
//     // 每一组第一个工作项读channel
//     if (item_id_0 == 0)
//     {
//         local_cell_edge[0] = read_channel_intel(DATAR2BU_CELLEDGE_CHANNEL);
//         if (DEBUG)
//             printf("bu: get cell = (%d, %d) from dr\n", local_cell_edge[0].position.x, local_cell_edge[0].position.y);
//     }
//     // 等待第一个工作项读取完毕
//     barrier(CLK_LOCAL_MEM_FENCE);
//     CellEdge cell_edge = local_cell_edge[0];
//     // 计算f global index
//     size_t id = (cell_edge.position.y * size_x + cell_edge.position.x) * Q_VECTOR + item_id_0;
//     // 计算 new f
//     fs[id] = nfeq_boundary(cell_edge.f[item_id_0], cell_edge.rho, cell_edge.u, cell_edge.u_edge, item_id_0);
// }

kernel void
bu(global real *restrict fs, Index size_x, Index size_y)
{
    size_t end = 2 * (size_x + size_y - 2);
#pragma ivdep
    while ((end--) > 0)
    {
        CellEdge cell_edge = read_channel_intel(DATAR2BU_CELLEDGE_CHANNEL);
        if (DEBUG)
            printf("bu: get cell = (%d, %d) from dr\n", cell_edge.position.x, cell_edge.position.y);
        // 计算f global index
        size_t id = (cell_edge.position.y * size_x + cell_edge.position.x) * Q_VECTOR;
        for (int item_id_0 = 0; item_id_0 < Q_VECTOR; item_id_0++)
        {
            // 计算 new f
            fs[id + item_id_0] = nfeq_boundary(cell_edge.f[item_id_0], cell_edge.rho, cell_edge.u, cell_edge.u_edge, item_id_0);
        }
    }
}