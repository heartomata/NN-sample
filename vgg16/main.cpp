#include <iostream>
#include "arr3d.h"
using namespace std;


int main(){
    arr3d x(224, 224, 3);
    //x.rand_seed();
    int conv_counter = 1;
    int pool_counter = 1;
    int fc_counter = 1;
    //
    x = conv(x, 3, 64, 3, conv_counter++);
    x = conv(x, 64, 64, 3, conv_counter++);
    x = maxpooling(x, 2, 2, pool_counter++);
    //
    x = conv(x, 64, 128, 3, conv_counter++);
    x = conv(x, 128, 128, 3, conv_counter++);
    x = maxpooling(x, 2, 2, pool_counter++);
    //
    x = conv(x, 128, 256, 3, conv_counter++);
    x = conv(x, 256, 256, 3, conv_counter++);
    x = conv(x, 256, 256, 3, conv_counter++);
    x = maxpooling(x, 2, 2, pool_counter++);
    //
    x = conv(x, 256, 512, 3, conv_counter++);
    x = conv(x, 512, 512, 3, conv_counter++);
    x = conv(x, 512, 512, 3, conv_counter++);
    x = maxpooling(x, 2, 2, pool_counter++);
    //
    x = conv(x, 512, 512, 3, conv_counter++);
    x = conv(x, 512, 512, 3, conv_counter++);
    x = conv(x, 512, 512, 3, conv_counter++);
    x = maxpooling(x, 2, 2, pool_counter++);

    x = fc(x, 4096, fc_counter++);
    x = fc(x, 4096, fc_counter++);
    x = fc(x, 1000, fc_counter++);
}
