/**
read cell data from global buffer */
kernel void interior_streamer(
    global Cell *restrict cells,
    LongIndex ncells, Index size_x, Index size_y)
{
    bool buf_flag = false;
    Vector2DIndex size = MakeVecIndex(size_x, size_y);
    LongIndex fullsize = size.x * size.y * Q_VECTOR; // every cell has Q_VECTOR fs
    // LongIndex count_flag = count / 3;
    LongIndex count_flag = 0;
    LongIndex count = fullsize;
    LongIndex offset = ncells;
#pragma ivdep
    while (true)
    {
        // printf("interior_streamer%d\n", count);
        FWithPosition fwp = read_channel_intel(CU2ISTREAMER_FWP_CHANNEL);
        if (fwp.position.x > 0 && fwp.position.x < size.x - 1 && fwp.position.y > 0 && fwp.position.y < size.y - 1)
        {
            LongIndex cellIndex = getIndex(fwp.position, size) + offset;
            cells[cellIndex].f[fwp.i] = fwp.f;
        }
        count--;
        if (count == count_flag)
            write_channel_intel(ISTREAMER2DATAR_CHAR_CHANNEL, '1');
        if (count == 0)
        {
            count = fullsize;
            // ping-pong buffer, using offset
            offset = 0;
            if (buf_flag)
                offset = ncells;
            buf_flag = !buf_flag;
        }
    }
}