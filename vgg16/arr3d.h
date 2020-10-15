#ifndef _ARR3D_H_
#define _ARR3D_H_
#include<time.h>
#include<stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <vector>

using namespace std;

class arr3d{
public:
    arr3d(int a, int b, int c);

    arr3d &operator=(const arr3d&);

    void rand_seed();

    vector<vector<vector<int>>> block;
    int row, col, chan;
};

int relu(int);
arr3d conv(arr3d&, int, int, int, int);
arr3d maxpooling(arr3d, int, int, int);
arr3d fc(arr3d, int, int);

#endif

