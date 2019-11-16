// mask
// 00(0) 01(1) 10(-1) (0,1,-1)
// #define UP 0x2    // (0,-1)0010
// #define DOWN 0x1  // (0,1)0001
// #define LEFT 0x4  // (1,0)0100
// #define RIGHT 0x8 // (-1,0)1000

// #define SETUP(x) x = (x | UP)
// #define SETDOWN(x) x = (x | DOWN)
// #define SETLEFT(x) x = (x | LEFT)
// #define SETRIGHT(x) x = (x | RIGHT)

// #define ISUP(x) ((x & UP) == UP)
// #define ISLEFT(x) ((x & LEFT) == LEFT)
// #define ISDOWN(x) ((x & DOWN) == DOWN)
// #define ISRIGHT(x) ((x & RIGHT) == RIGHT)
// #define SHIFT(x) (x << 4)

// #include <iostream>
// using namespace std;

// int main()
// {
//     unsigned int t_file = 1000;
//     int t = -1;
//     unsigned int a = t_file + t;
//     cout << a << endl;
//     return 0;
// }
