#include"arr3d.h"

arr3d::arr3d(int a, int b, int c){
    /*block = (int ***)malloc(sizeof(int ***)*c);
    for(int i = 0; i < c; i++){
        block[i] = (int **)malloc(sizeof(int *)*a);
        for(int j = 0; j < a; j++){
            block[i][j] = (int *)malloc(sizeof(int)*b);
        }
    }*/
    block.resize(c,vector<vector<int> >(a,vector<int>(b)));
    row = a;
    col = b;
    chan = c;
    for(int i = 0; i < chan; i++){
        for(int j = 0; j < row; j++){
            for(int k = 0; k < col; k++){
                block[i][j][k] = 0;
            }
        }
    }
}

arr3d& arr3d::operator=(const arr3d &arr1){
    block = arr1.block;
    row = arr1.row;
    col = arr1.col;
    chan = arr1.chan;
	return *this;
}

void arr3d::rand_seed(){
    srand(time(NULL));
    for(int i = 0; i < chan; i++){
        for(int j = 0; j < row; j++){
            for(int k = 0; k < col; k++){
                block[i][j][k] = rand()%255;
            }
        }
    }
        /*for(int i=0;i<chan;i++)
          {
            for(int j=0;j<row;j++)
            {
              for(int k=0;k<col;k++)
              {
                printf("%.2d ",block[i][j][k]);
              }
              printf("\n");
            }
            printf("\n");
          }

    printf("%d %d %d\n", chan, row, col);*/
}


int relu(int x){
    return 0 > x?0 : x;
}

arr3d conv(arr3d& arrin, int chan_in, int chan_out, int filter_size, int layer_num){

    arr3d flt(filter_size, filter_size, chan_in);
    arr3d arrout(arrin.col, arrin.row, chan_out);
    flt.rand_seed();
    int connection = 0;
    //printf("yes");
    for(int i = 0; i < chan_out; i++){
        for(int j = 0; j < arrin.row + 2; j++){
            for(int k = 0; k < arrin.col + 2; k++){
                    //printf("%d %d %d", i, j, k);
                //add bias
                for(int x = 0; x < filter_size; x++){
                    for(int y = 0; y < filter_size; y++){
                        for(int z = 0; z < chan_in; z++){
                            if((j > 0) && (k > 0) && (j <= arrin.row) && (k <= arrin.col)){
                                if((j+x-1 >= 0) && (k+y-1 >= 0) && (j+x-1 < arrin.row) && (k+y-1 < arrin.col)){
                                    arrout.block[i][j-1][k-1] += arrin.block[z][j+x-1][k+y-1] * flt.block[z][x][y];
                                    connection += 1;
                                }else{
                                    connection += 1;
                                }

                                //printf("conn = %d %d %d %d %d %d %d\n", connection, i, j, k, x, y, z);
                            }
                        }
                    }
                }
                if((j > 0) && (k > 0) && (j <= arrin.row) && (k <= arrin.col))arrout.block[i][j-1][k-1] = relu(arrout.block[i][j-1][k-1]);

            }
        }
    }
    for(int i = 0; i < arrin.chan; i++){
        for(int j = 0; j < arrin.row; j++){
            arrin.block[i][j].clear();
        }
    }
    for(int i = 0; i < arrin.chan; i++){
        arrin.block[i].clear();
    }
    printf("Conv %d: memory size: %.2f KB, # of parameter: %d, # of MAC: %d\n", layer_num, (double)(arrout.col*arrout.row*arrout.chan)/250, filter_size*filter_size*chan_in*chan_out, connection);
    return arrout;
}

arr3d maxpooling(arr3d arrin, int stride, int pool_size, int layer_num){
    arr3d arrout((arrin.row - pool_size)/stride + 1, (arrin.col - pool_size)/stride + 1, arrin.chan);
    int max_val, connection = 0;
    for(int i = 0; i < arrout.chan; i++){
        for(int j = 0; j < arrout.col; j++){
            for(int k = 0; k < arrout.row; k++){
                max_val = INT_MIN;
                for(int x = 0; x < pool_size; x++){
                    for(int y = 0; y < pool_size; y++){
                        if(stride*j+x < arrin.row && stride*k+y < arrin.col){
                            if(arrin.block[i][stride*j+x][stride*k+y] > max_val) max_val = arrin.block[i][stride*j+x][stride*k+y];
                        }

                    }
                }
                arrout.block[i][j][k] = max_val;
                connection += 1;// here to count # of MAC?
            }
        }
    }
    printf("Maxpool %d: memory size: %.2f KB, # of parameter: 0, # of MAC: %d\n", layer_num, (double)(arrout.col*arrout.row*arrout.chan)/250, connection);
    return arrout;
}

arr3d fc(arr3d arrin, int output, int layer_num){
    srand(time(NULL));
    arr3d arrout(1, output, 1);
    int connection = 0;
    for(int x = 0; x < output; x++){
        for(int i = 0; i < arrin.chan; i++){
            for(int j = 0; j < arrin.row; j++){
                for(int k = 0; k < arrin.col; k++){
                        arrout.block[0][0][x] += arrin.block[i][j][k] * rand()%255;
                        connection += 1;
                }
            }
        }
    }
    printf("FC%d %d: memory size: %.2f KB, # of parameter: %d, # of MAC: %d\n", output, layer_num, (double)output/250, connection, connection);
    return arrout;
}


