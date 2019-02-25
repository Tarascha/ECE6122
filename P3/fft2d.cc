#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <signal.h>
#include <math.h>
#include "mpi.h"
#include <cmath>


#include <time.h>

using namespace std;
// #pragma once

class Complex {
public:
    Complex() : real(0.0f), imag(0.0f) {}

    Complex(float r) : real(r), imag(0.0f) {}

    Complex(float r, float i) : real(r), imag(i) {}

    Complex operator+(const Complex& b) const
    {
      return Complex(real + b.real, imag + b.imag);
    }
    Complex operator-(const Complex& b) const
    {
      return Complex(real - b.real, imag - b.imag);
    }
    Complex operator*(const Complex& b) const{
      return Complex(real*b.real - imag*b.imag,
                 real*b.imag + imag*b.real);
    }

    Complex Mag() const{
      return Complex(sqrt(real*real + imag*imag));
    }
    Complex angle() const{
      return Complex(atan2(imag, real) * 360 / (2 * M_PI));
    }
    Complex conj() const{
      return Complex(real, -imag);
    }

    float real;
    float imag;
};

std::ostream& operator<<(std::ostream& os, const Complex& rhs);

std::ostream& operator<< (std::ostream& os, const Complex& rhs) {
    Complex c(rhs);
    if(fabsf(rhs.imag) < 1e-10) c.imag = 0.0f;
    if(fabsf(rhs.real) < 1e-10) c.real = 0.0f;

    if(c.imag == 0) {
        os << c.real;
    }
    else {
        os << "(" << c.real << "," << c.imag << ")";
    }
    return os;
};

class InputImage {
public:

    InputImage(const char* filename){
       std::ifstream ifs(filename);
    if(!ifs) {
        std::cout << "Can't open image file " << filename << std::endl;
        exit(1);
    }

    ifs >> w >> h;
    data = new Complex[w * h];
    for(int r = 0; r < h; ++r) {
        for(int c = 0; c < w; ++c) {
            float real;
            ifs >> real;
            data[r * w + c] = Complex(real);
        }
    }
    }
    int GetWidth() const{
      return w;
    }
    int GetHeight() const{
      return h;
    }

    //returns a pointer to the image data.  Note the return is a 1D
    //array which represents a 2D image.  The data for row 1 is
    //immediately following the data for row 0 in the 1D array
    Complex* GetImageData() const{
      return data;
    }

    //use this to save output from forward DFT
    void SaveImageData(const char* newFileName, Complex* d,
                               int w, int h)
    {
  ofstream ofs(newFileName);
  if (!ofs)
    {
      cout << "Can't create output image " << newFileName << endl;
      return;
    }
  ofs << w << " " << h << endl;
  for (int r = 0; r < h; ++r)
    { // for each row
      for (int c = 0; c < w; ++c)
        { // for each column
          ofs << d[r * w + c] << " ";
        }
      ofs << endl;
    }
    }
    //use this to save output from reverse DFT
    void save_image_data_real(const char* newFileName, Complex* d, int w, int h){
      ofstream ofs(newFileName);
  if (!ofs)
    {
      cout << "Can't create output image " << newFileName << endl;
      return;
    }
  ofs << w << " " << h << endl;
  for (int r = 0; r < h; ++r)
    { // for each row
      for (int c = 0; c < w; ++c)
        { // for each column
          ofs << d[r * w + c].Mag() << " ";
        }
      ofs << endl;
    }
    }

private:
    int w;
    int h;
    Complex* data;
};



unsigned Sorting(unsigned v,int N)
{
  unsigned n = N;
  unsigned r = 0;
   
  for (--n; n > 0; n >>= 1)
    {
      r <<= 1;
      r |= (v & 0x1);
      v >>= 1;
    }
  return r;
}


void FFT(Complex* h, int N)
{
  Complex* tem = new Complex[N]();
  for( int i = 0; i < N; ++i ) {
      int s= Sorting(i,N);
      tem[s] = h[i];
  }
  for( int i = 0; i < N; ++i ) h[i] = tem[i];
  delete[] tem;

  Complex* W = new Complex[N]();
  for( int i = 0; i < N/2; ++i )
  {
    W[i] = Complex(cos(2*M_PI*i/N),-sin(2*M_PI*i/N));
    W[i+N/2] = Complex(-1) * W[i];
  }
  
   int subLen = 2;
    int subNum = N/2;
    while(subLen <= N)
    {
        Complex* temp = new Complex[N]();
        for( int i = 0; i < subNum; ++i )
            for( int j = 0; j < subLen; ++j )
                temp[i*subLen + j] = temp[i*subLen + j] + h[i*subLen + j%(subLen/2)] + h[i*subLen + j%(subLen/2) + subLen/2] * W[N*j/subLen];
        for( int j = 0; j < N; ++j )
            h[j] = temp[j];
        subLen *= 2;
        subNum /= 2;
        delete[] temp;
    }
    delete[] W;
}

void FFTInverse(Complex* H, int N)
{

  Complex* tem = new Complex[N]();
  for( int i = 0; i < N; ++i ){
      int s= Sorting(i,N);
      tem[s] = H[i];
  }
  for( int i = 0; i < N; ++i ){
      H[i] = tem[i];
  }
  delete[] tem;

  Complex* W = new Complex[N]();
  for( int i = 0; i < N/2; ++i )
  {
    W[i] = Complex(cos(2*M_PI*i/N),sin(2*M_PI*i/N));
    W[i+N/2] = Complex(-1) * W[i];
  }


  int subLen = 2;
    int subNum = N/2;
    while(subLen <= N)
    {
        Complex* temp = new Complex[N]();
        for( int i = 0; i < subNum; ++i )
            for( int j = 0; j < subLen; ++j )
                temp[i*subLen + j] = temp[i*subLen + j] + H[i*subLen + j%(subLen/2)] + H[i*subLen + j%(subLen/2) + subLen/2] * W[N*j/subLen];
        for( int j = 0; j < N; ++j )
            H[j] = temp[j];
        subLen *= 2;
        subNum /= 2;
        delete[] temp;
    }
    for( int i = 0; i < N; ++i )
    {
        H[i] =  H[i]*Complex(1.0/N);
        if( H[i].Mag().real < 1e-10 ) H[i] = Complex(0);
    }

    delete[] W;
}

void Transpose(Complex* h, int width, int height)
{
  for( int row = 0; row < height; ++row )
    for( int col = 0; col < width; ++col )
      if(col > row)
      {
        Complex temp = h[row * width + col];
        h[row * width + col] = h[col * width + row];
        h[col * width + row] = temp;
      }
}

void Transform(const char* inputFN, const char* outpuFN, const int mode)
{
    InputImage image(inputFN);  
    int width = image.GetWidth();
    int height = image.GetHeight();

    int nCPU, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &nCPU);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int rowsPerCPU = height/nCPU;

    Complex* I;
    if( rank == 0) I = image.GetImageData();

    int size = rowsPerCPU * width* sizeof(Complex);
    Complex* h = new Complex[width*rowsPerCPU]();

    // Rank 0 first send each value to other ranks to do fft
    if( rank == 0 )
    {
        for( int cpu = 1; cpu < nCPU; ++cpu)
        {
            int startRow = cpu * rowsPerCPU;
            MPI_Send(I + startRow * width, size , MPI_CHAR, cpu, startRow, MPI_COMM_WORLD);
            cout<< "rank 0 sends to rank "<<cpu<<endl;
        }
        for ( int row = 0; row < rowsPerCPU; ++row )
        {
          Complex* currenth = I + row * width;
          FFT(currenth, width);
        }
    } else {
        int startRow = rank * rowsPerCPU;
        MPI_Recv(h, size , MPI_CHAR, 0,startRow, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        cout<< "rank "<< rank <<" receive from rank 0"<<endl;
        for ( int row = 0; row < rowsPerCPU; ++row )
        {
            Complex* currenth = h + row * width;
            FFT(currenth, width);
        }
        MPI_Send(h, size , MPI_CHAR,  0,startRow, MPI_COMM_WORLD);
        cout<< "rank "<< rank <<" Send back to rank 0"<<endl;
    }
    // After applying fft transfer back to rank0
    if( rank == 0 )
    {
        for( int cpu = 1; cpu < nCPU; ++cpu)
        {
            int startRow = cpu * rowsPerCPU;
            MPI_Recv(I + startRow * width, size , MPI_CHAR, cpu, startRow, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            cout<< "rank 0 receive from rank "<<cpu<<endl;
        }
        cout << "rank " << rank << " sented all being received" << endl;
        string fn1("myafter1d.txt");
        image.SaveImageData(fn1.c_str(), I, width, height);
        Transpose(I, width, height);
    }
    delete [] h;

//End of 1d-Transform.txt
// Shift height and width, Make row become column
    
	    int temp = height;
	    height = width;
	    width = temp;
	    rowsPerCPU = height/nCPU;
	    Complex* h2 = new Complex[width*rowsPerCPU]();

	    if( rank == 0 )
	    {
	        for( int cpu = 1; cpu < nCPU; ++cpu)
	        {
	            int startRow = cpu * rowsPerCPU;
	            MPI_Send(I + startRow * width, size , MPI_CHAR, cpu, startRow, MPI_COMM_WORLD);
	            cout<< "rank 0 sends to rank "<<cpu<<endl;
	        }
	        for ( int row = 0; row < rowsPerCPU; ++row )
	        {
	            Complex* currenth = I + row * width;
	            FFT(currenth, width);
	        }
	    } else {
	        int startRow = rank * rowsPerCPU;
	        MPI_Recv(h2, size , MPI_CHAR, 0,startRow, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	        cout<< "rank "<< rank <<" receive from rank 0"<<endl;
	        for ( int row = 0; row < rowsPerCPU; ++row )
	        {
	            Complex* currenth = h2 + row * width;
	            FFT(currenth, width);
	        }
	        MPI_Send(h2, size , MPI_CHAR,  0,startRow, MPI_COMM_WORLD);
	        cout<< "rank "<< rank <<" Send back to rank 0"<<endl;
	    }
	    // After applying fft transfer back to rank0
	    if( rank == 0 )
	    {
	        for( int cpu = 1; cpu < nCPU; ++cpu)
	        {
	            int startRow = cpu * rowsPerCPU;
	            MPI_Recv(I + startRow * width, size , MPI_CHAR, cpu, startRow, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	            cout<< "rank 0 receive from rank "<<cpu<<endl;
	        }
	        cout << "rank " << rank << " sented all being received" << endl;
	        Transpose(I, width, height);
	        string fn2(outpuFN);
	        image.SaveImageData(fn2.c_str(), I, width, height);
      }
	    delete [] h2;

	    temp = height;
	    height = width;
	    width = temp;
	    rowsPerCPU = height/nCPU;

	//---------------------------------------------** Transform Inverse **/--------------------------------------------------------------------------------------
      
	     Complex* H = new Complex[width*rowsPerCPU]();
    if(mode ==0){
	    if( rank == 0 )
	    {
	        for( int cpu = 1; cpu < nCPU; ++cpu)
	        {
	            int startRow = cpu * rowsPerCPU;
	            MPI_Send(I + startRow * width, size , MPI_CHAR, cpu, startRow, MPI_COMM_WORLD);
	            cout<< "rank 0 sends to rank "<<cpu<<endl;
	        }
	        for ( int row = 0; row < rowsPerCPU; ++row )
	        {
	            Complex* currenth = I + row * width;
	            FFTInverse(currenth, width);
	        }
	    } else {
	        int startRow = rank * rowsPerCPU;
	        MPI_Recv(H, size , MPI_CHAR, 0,startRow, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	        cout<< "rank "<< rank <<" receive from rank 0"<<endl;
	        for ( int row = 0; row < rowsPerCPU; ++row )
	        {
	            Complex* currenth = H + row * width;
	            FFTInverse(currenth, width);
	        }
	        MPI_Send(H, size , MPI_CHAR,  0,startRow, MPI_COMM_WORLD);
	        cout<< "rank "<< rank <<" Send back to rank 0"<<endl;
	    }
	    // After applying fft transfer back to rank0
	    if( rank == 0 )
	    {
	        for( int cpu = 1; cpu < nCPU; ++cpu)
	        {
	            int startRow = cpu * rowsPerCPU;
	            MPI_Recv(I + startRow * width, size , MPI_CHAR, cpu, startRow, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	            cout<< "rank 0 receive from rank "<<cpu<<endl;
	        }
	        cout << "rank " << rank << " sented all being received" << endl;
	        Transpose(I, width, height);
	    }
	    delete [] H;

	//End of 1d-Transform.txt
	// Shift height and width, Make row become column
	    temp = height;
	    height = width;
	    width = temp;
	    rowsPerCPU = height/nCPU;
	    Complex* H2 = new Complex[width*rowsPerCPU]();

	    if( rank == 0 )
	    {
	        for( int cpu = 1; cpu < nCPU; ++cpu)
	        {
	            int startRow = cpu * rowsPerCPU;
	            MPI_Send(I + startRow * width, size , MPI_CHAR, cpu, startRow, MPI_COMM_WORLD);
	            cout<< "rank 0 sends to rank "<<cpu<<endl;
	        }
	        for ( int row = 0; row < rowsPerCPU; ++row )
	        {
	            Complex* currenth = I + row * width;
	            FFTInverse(currenth, width);
	        }
	    } else {
	        int startRow = rank * rowsPerCPU;
	        MPI_Recv(H2, size , MPI_CHAR, 0,startRow, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	        cout<< "rank "<< rank <<" receive from rank 0"<<endl;
	        for ( int row = 0; row < rowsPerCPU; ++row )
	        {
	            Complex* currenth = H2 + row * width;
	            FFTInverse(currenth, width);
	        }
	        MPI_Send(H2, size , MPI_CHAR,  0,startRow, MPI_COMM_WORLD);
	        cout<< "rank "<< rank <<" Send back to rank 0"<<endl;
	    }
	    // After applying fft transfer back to rank0
	    if( rank == 0 )
	    {
	        for( int cpu = 1; cpu < nCPU; ++cpu)
	        {
	            int startRow = cpu * rowsPerCPU;
	            MPI_Recv(I + startRow * width, size , MPI_CHAR, cpu, startRow, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	            cout<< "rank 0 receive from rank "<<cpu<<endl;
	        }
	        cout << "rank " << rank << " sented all being received" << endl;
	        Transpose(I, width, height);
	        temp = height;
	        height = width;
	        width = temp;
	        string fn3(outpuFN);
	        image.save_image_data_real(fn3.c_str(), I, width, height);

	    }
	    delete [] H2;
	}
}

int main(int argc, char** argv)
{
  clock_t startTime, endTime;
  startTime=clock();
  string fn("Tower256.txt"); // default file name
  string mode= "foward";
  string fn9;
  if (argc > 1) {
  	mode = string(argv[1]);
  	fn = string(argv[2]);
  	fn9 = string(argv[3]);
  }  
  MPI_Init(&argc, &argv);
  if (mode == "forward"){
    printf("foward\n");
  	Transform(fn.c_str(), fn9.c_str(), 1);
  } else {
    printf("reverse\n");
  	Transform(fn.c_str(), fn9.c_str(), 0);
  }
  
  MPI_Finalize();
  endTime=clock();
  cout<<"Total time = "<<(double)(endTime - startTime)/CLOCKS_PER_SEC<<"s"<<endl;
  
  return 0;
}  
  

  
