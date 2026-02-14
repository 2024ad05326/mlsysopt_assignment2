#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>

double sigmoid(double z){
    return 1.0/(1.0+exp(-z));
}

int main(){

    int N=1000000;
    int d=10;
    double lr=0.01;
    int epochs=50;

    std::vector<double> X(N*d);
    std::vector<double> y(N);
    std::vector<double> w(d,0.0);

    for(int i=0;i<N*d;i++)
        X[i]=rand()/(double)RAND_MAX;

    for(int i=0;i<N;i++)
        y[i]=rand()%2;

    auto start=std::chrono::high_resolution_clock::now();

    for(int e=0;e<epochs;e++){
        double grad=0.0;

        for(int i=0;i<N;i++){
            double dot=0.0;
            for(int j=0;j<d;j++)
                dot+=X[i*d+j]*w[j];

            double pred=sigmoid(dot);
            grad+=pred-y[i];
        }

        grad/=N;
        for(int j=0;j<d;j++)
            w[j]-=lr*grad;
    }

    auto end=std::chrono::high_resolution_clock::now();
    std::cout<<"Done\n";
}
