#include <iostream>
#include <string.h>
#include <fstream>
#include <sstream>
#include <stdio.h>
#include <vector>
#include <time.h>
#include <cuda.h>
#include <math.h>
#include <chrono>
#include <ctime>
using namespace std;
//qsub -I -q coc-ice -l nodes=1:ppn=8:gpus=1,walltime=04:30:00,pmem=2gb
//qsub -I -q coc-ice -l nodes=1,walltime=02:30:00,pmem=2gb
const float PI = 3.14159265358979f;
//class for Complex number -------------------------------------------------------
class Complex {
		public:
			__device__ __host__ Complex() : real(0.0f), imag(0.0f){
			}

			__device__ __host__ Complex(float r, float i) : real(r), imag(i){

			}

			__device__ __host__ Complex(float r) : real(r), imag(0.0f){

			}

			__device__ __host__ Complex operator+ (const Complex& b) const{
				return Complex(real + b.real, imag + b.imag);
			}

			__device__ __host__ Complex operator- (const Complex& b) const{
				return Complex(real - b.real, imag - b.imag);
			}

			__device__ __host__ Complex operator* (const Complex& b) const{
				return Complex(real * b.real - imag * b.imag, real * b.imag + imag * b.real);
			}

			__device__ __host__ Complex Mag() const{
				return Complex(sqrt(real * real + imag * imag));
			}

			__device__ __host__ Complex Angle() const{
				return Complex(atan2(imag, real) * 360 / (2 * PI));
			}

			__device__ __host__ Complex Conj() const{
				return Complex(real, -imag);
			}

			void Print() const{
				if(imag == 0){
					cout << real;
				}else{
					cout << '(' << real << ',' << imag << ')' << endl;
				}
			}

		
			float real;
			float imag;
};

ostream& operator<< (ostream& os, const Complex& rhs) {
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
}

//class for input and output image---------------------------------------------------
class InputImage {
public:

    InputImage(const char* filename);
    int get_width() const;
    int get_height() const;

    //returns a pointer to the image data.  Note the return is a 1D
    //array which represents a 2D image.  The data for row 1 is
    //immediately following the data for row 0 in the 1D array
    Complex* get_image_data() const;

    //use this to save output from forward DFT
    void save_image_data(const char* filename, Complex* d, int w, int h);
    //use this to save output from reverse DFT
    void save_image_data_real(const char* filename, Complex* d, int w, int h);
    //use this to check mag
    void save_image_data_mag(const char* filename, Complex* d, int w, int h);

private:
    int w;
    int h;
    Complex* data;
};

InputImage::InputImage(const char* filename) {
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

int InputImage::get_width() const {
    return w;
}

int InputImage::get_height() const {
    return h;
}

Complex* InputImage::get_image_data() const {
    return data;
}

void InputImage::save_image_data(const char *filename, Complex *d, int w, int h) {
    std::ofstream ofs(filename);
    if(!ofs) {
        std::cout << "Can't create output image " << filename << std::endl;
        return;
    }

    ofs << w << " " << h << std::endl;

    for(int r = 0; r < h; ++r) {
        for(int c = 0; c < w; ++c) {
            ofs << d[r * w + c] << " ";
        }
        ofs << std::endl;
    }
}

void InputImage::save_image_data_real(const char* filename, Complex* d, int w, int h) {
    std::ofstream ofs(filename);
    if(!ofs) {
        std::cout << "Can't create output image " << filename << std::endl;
        return;
    }

    ofs << w << " " << h << std::endl;

    for (int r = 0; r < h; ++r) {
        for (int c = 0; c < w; ++c) {
            ofs << d[r * w + c].real << " ";
        }
        ofs << std::endl;
    }
}

void InputImage::save_image_data_mag(const char* newFileName, Complex* d,int w, int h){


  std::ofstream ofs(newFileName);
  if (!ofs)
    {
      std::cout << "Can't create output image " << newFileName << std::endl;
      return;
    }
  ofs << w << " " << h << endl;
  for (int r = 0; r < h; ++r)
    { // for each row
      for (int c = 0; c < w; ++c)
        { // for each column
          ofs << d[r * w + c].Mag() << " ";
        }
      ofs << std::endl;
    }
}


//test,test


__global__ void reorder(Complex* a, Complex* b, int N){
    long idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < N * N){
        int x = idx / N;
        int y = idx % N;
        unsigned r = 0; // reversed index;
        unsigned n = N;
        unsigned num = y;

        for(--n; n > 0; n >>= 1){
            r <<= 1;
            r |= (num & 0x1);
            num >>= 1;
        }

        y = r;

        b[x * N + y] = a[idx];
    }
    
}

__global__ void CountW(Complex* W, int N){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < N/2){
        W[idx] = Complex(cos(2 * PI * idx / N), -sin(2 * PI * idx / N));
        W[idx + N/2] = Complex(-1) * W[idx];
    }
    //__syncthreads();
}

__global__ void CountWInverse(Complex* W, int N){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < N/2){
        W[idx] = Complex(cos(2 * PI * idx / N), sin(2 * PI * idx / N));
        W[idx + N/2] = Complex(-1) * W[idx];
    }
    //__syncthreads();
}

__global__ void Tmatrix(Complex* a, int width, int height){
    long idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < width * height){
        int x = idx / width;
        int y = idx % width;
        if(y > x){
            Complex tmp = a[x * width + y];
            a[x * width + y] = a[y * width + x];
            a[y * width + x] = tmp;
        }
    }
     //__syncthreads();
}

// __global__ void TransformAnArrray(Complex* a, Complex*b, int N, Complex* W){//b is all 0 initially
//     long idx = threadIdx.x + blockIdx.x * blockDim.x;
//     int x = idx / N;// x th row//temp[]->b; H[] -> a[x * width + 0] ~ a[x * width + width - 1]
//     int y = idx % N;// y th element in x th row

//     if(idx < N * N ){//

//         int groupLen = 2;
//         int groupNum = N / 2;
        
//         while(groupLen <= N){//a or b should include x* width first or just use idx as index
//             int i = y / groupLen;
//             int j = y % groupLen;
//             b[idx] = a[x * N + i * groupLen + j % (groupLen/2)] + a[x * N + i * groupLen + j%(groupLen/2) + groupLen/2] * W[N*j/groupLen];
//             __syncthreads();
//             a[idx] = b[idx];// should be ok?
//             groupLen *= 2;
//             groupNum /= 2;
//             __syncthreads();
//         }

//     }//
// }

 __global__ void TransformAnArrray(Complex* a, Complex*b, int N, Complex* W, int groupLen, int groupNum){
    long idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < N * N){
        int x = idx / N;// x th row//temp[]->b; H[] -> a[x * width + 0] ~ a[x * width + width - 1]
        int y = idx % N;// y th element in x th row

        int i = y / groupLen;
        int j = y % groupLen;

        b[idx] = a[x * N + i * groupLen + j % (groupLen/2)] + a[x * N + i * groupLen + j%(groupLen/2) + groupLen/2] * W[N*j/groupLen];

    }
    
 }

__global__ void ConvertAB(Complex* a, Complex*b, int N){
     long idx = threadIdx.x + blockIdx.x * blockDim.x;
     if(idx < N * N){
        a[idx] = b[idx];
     }

}

__global__ void CmpleteInverseT(Complex*a, int N){
    long idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < N * N){
        a[idx] = Complex(1.0 / N) * a[idx];
        if(a[idx].Mag().real < 1e-10){
            a[idx] = Complex(0);
        }
    }
}




int main(int argc, const char * argv[]) {

        clock_t startTime, endTime;
        const char* type = argv[1];
		const char* filename = argv[2];
        const char* outputfile = argv[3];

        startTime = clock();
        
		InputImage Tower(filename);

		int Tower_height = Tower.get_height();
		int Tower_width = Tower.get_width();
        Complex* TowerData = Tower.get_image_data();// data itself// one diamention array
//test-------------------------------------
		// cout << "height = " << Tower_height << "\n" << "width = " << Tower_width << endl;
        // for(long i = 0; i < Tower_width * Tower_height; i++){
        //     printf("thread:%ld, content:%f\n", i, TowerData[i].real);
        // }
//test-------------------------------------

        Complex* d_a;
        Complex* d_b;
        Complex* d_w;

        chrono::steady_clock::time_point tStart;
        tStart = chrono::steady_clock::now();

        cudaMalloc(&d_a, sizeof(Complex) * Tower_width * Tower_height);
        cudaMalloc(&d_b, sizeof(Complex) * Tower_width * Tower_height);
        cudaMalloc(&d_w, sizeof(Complex) * Tower_width);
        
        cudaMemcpy(d_a, TowerData, sizeof(Complex)* Tower_width * Tower_height, cudaMemcpyHostToDevice);

        reorder<<<(Tower_width * Tower_height + 511) / 512, 512>>>(d_a, d_b, Tower_width);
        ConvertAB<<<(Tower_width * Tower_height + 511) / 512, 512>>>(d_a, d_b, Tower_width);

        CountW<<<(Tower_width + 511) / 512, 512>>>(d_w, Tower_width);

        int groupLen = 2;
        int groupNum = Tower_width / 2;

        while(groupLen <= Tower_width){

            TransformAnArrray<<<(Tower_width * Tower_height + 511) / 512, 512>>>(d_a, d_b, Tower_width, d_w, groupLen, groupNum);
            ConvertAB<<<(Tower_width * Tower_height + 511) / 512, 512>>>(d_a, d_b, Tower_width);
            groupLen *= 2;
            groupNum /= 2;

        }
//1D test
        // cudaMemcpy(TowerData, d_a, sizeof(Complex) * Tower_width * Tower_height, cudaMemcpyDeviceToHost);
        // Tower.save_image_data_mag(outputfile, TowerData, Tower_width, Tower_height);
//1D test correct
//---------------------------------------------------------------------------------------------------------
        Tmatrix<<<(Tower_width * Tower_height + 511) / 512, 512>>>(d_a, Tower_width, Tower_height);
        reorder<<<(Tower_width * Tower_height + 511) / 512, 512>>>(d_a, d_b, Tower_width);
        ConvertAB<<<(Tower_width * Tower_height + 511) / 512, 512>>>(d_a, d_b, Tower_width);
//---------------------------------------------------------------------------------------------------------
        groupLen = 2;
        groupNum = Tower_width / 2;

        while(groupLen <= Tower_width){

            TransformAnArrray<<<(Tower_width * Tower_height + 511) / 512, 512>>>(d_a, d_b, Tower_width, d_w, groupLen, groupNum);
            ConvertAB<<<(Tower_width * Tower_height + 511) / 512, 512>>>(d_a, d_b, Tower_width);
            groupLen *= 2;
            groupNum /= 2;

        }

        

        Tmatrix<<<(Tower_width * Tower_height + 511) / 512, 512>>>(d_a, Tower_width, Tower_height);

        if(type[0] == 'f'){
            cudaMemcpy(TowerData, d_a, sizeof(Complex) * Tower_width * Tower_height, cudaMemcpyDeviceToHost);
            chrono::steady_clock::time_point tEnd = chrono::steady_clock::now();
            chrono::duration<double> time_span = chrono::duration_cast<chrono::duration<double> >(tEnd - tStart);
            cout << "Time ellipsed: " << time_span.count() << " seconds... \n";
            Tower.save_image_data(outputfile, TowerData, Tower_width, Tower_height);
        }else{
            reorder<<<(Tower_width * Tower_height + 511) / 512, 512>>>(d_a, d_b, Tower_width);
            ConvertAB<<<(Tower_width * Tower_height + 511) / 512, 512>>>(d_a, d_b, Tower_width);

            CountWInverse<<<(Tower_width + 511) / 512, 512>>>(d_w, Tower_width);// different W

            groupLen = 2;
            groupNum = Tower_width / 2;

            while(groupLen <= Tower_width){

                TransformAnArrray<<<(Tower_width * Tower_height + 511) / 512, 512>>>(d_a, d_b, Tower_width, d_w, groupLen, groupNum);
                ConvertAB<<<(Tower_width * Tower_height + 511) / 512, 512>>>(d_a, d_b, Tower_width);
                groupLen *= 2;
                groupNum /= 2;

            }

            CmpleteInverseT<<<(Tower_width * Tower_height + 511) / 512, 512>>>(d_a, Tower_width);

            Tmatrix<<<(Tower_width * Tower_height + 511) / 512, 512>>>(d_a, Tower_width, Tower_height);
            reorder<<<(Tower_width * Tower_height + 511) / 512, 512>>>(d_a, d_b, Tower_width);
            ConvertAB<<<(Tower_width * Tower_height + 511) / 512, 512>>>(d_a, d_b, Tower_width);

            groupLen = 2;
            groupNum = Tower_width / 2;

            while(groupLen <= Tower_width){

                TransformAnArrray<<<(Tower_width * Tower_height + 511) / 512, 512>>>(d_a, d_b, Tower_width, d_w, groupLen, groupNum);
                ConvertAB<<<(Tower_width * Tower_height + 511) / 512, 512>>>(d_a, d_b, Tower_width);
                groupLen *= 2;
                groupNum /= 2;

            }

            CmpleteInverseT<<<(Tower_width * Tower_height + 511) / 512, 512>>>(d_a, Tower_width);

            Tmatrix<<<(Tower_width * Tower_height + 511) / 512, 512>>>(d_a, Tower_width, Tower_height);
            cudaMemcpy(TowerData, d_a, sizeof(Complex) * Tower_width * Tower_height, cudaMemcpyDeviceToHost);
            chrono::steady_clock::time_point tEnd = chrono::steady_clock::now();
            chrono::duration<double> time_span = chrono::duration_cast<chrono::duration<double> >(tEnd - tStart);
            cout << "Time ellipsed: " << time_span.count() << " seconds... \n";
            Tower.save_image_data(outputfile, TowerData, Tower_width, Tower_height);
        }

        

        //z.Print();

        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_w);
        delete[] TowerData;
        endTime = clock();
        cout<<"Total time = "<<(double)(endTime - startTime)/CLOCKS_PER_SEC<<"s"<<endl;
        //delete TowerData?
        return 0;
}
