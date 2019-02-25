#include <string>
#include <iostream>
#include <thread>
#include <fstream>
#include <sstream>
#include <vector>
#include <ctime>
#include <math.h>
#include <stdlib.h>
#include <cstdlib>
#include <stdio.h>

#include "complex.h"
#include "input_image.h"

using namespace std;
const float PI = 3.14159265358979f;

void Transform1D(Complex* h, int w, int L, Complex* H, Complex* c)
{
	for (int k = 0; k < L; k++) {
		for (int i = 0; i < w; ++i) {
			for (int j = 0; j < w; ++j) {
				H[i + k * w] = H[i + k * w] + c[j + w * i] * h[j + k * w];
			}
		}
	}
}

void Transform1DInverse(Complex* H, int w, int L, Complex* h, Complex* c)
{
	for (int k = 0; k < L; k++) {
		for (int i = 0; i < w; ++i) {
			for (int j = 0; j < w; ++j) {
				h[i + k * w] = h[i + k * w] + c[j + w * i] * H[j + k * w];
			}
		}
	}
}

void Transpose(Complex* h, int width, int height)
{
	for (int row = 0; row < height; ++row)
		for (int col = 0; col < width; ++col)
			if (col > row)
			{
				Complex temp = h[row * width + col];
				h[row * width + col] = h[col * width + row];
				h[col * width + row] = temp;
			}
}

int main(int argc, char *argv[]){
	string a = argv[1];
	string b = "forward";
	string c = "reverse";
	
	clock_t start = clock();
	//Read image data
	InputImage image(argv[2]);
	
	//Initialize variables.
	int width = image.get_width();
	int height = image.get_height();
	Complex* I;
	I = image.get_image_data();
	int length = width / 8;
	Complex* h1 = new Complex[width*length];
	Complex* h2 = new Complex[width*length];
	Complex* h3 = new Complex[width*length];
	Complex* h4 = new Complex[width*length];
	Complex* h5 = new Complex[width*length];
	Complex* h6 = new Complex[width*length];
	Complex* h7 = new Complex[width*length];
	Complex* h8 = new Complex[width*length];
	
	Complex* H1 = new Complex[width*length];
	Complex* H2 = new Complex[width*length];
	Complex* H3 = new Complex[width*length];
	Complex* H4 = new Complex[width*length];
	Complex* H5 = new Complex[width*length];
	Complex* H6 = new Complex[width*length];
	Complex* H7 = new Complex[width*length];
	Complex* H8 = new Complex[width*length];
	
	Complex* h = new Complex[width*height];
	Complex* H = new Complex[width*height];
	Complex* w1 = new Complex[width*width];
	Complex* w2 = new Complex[width*width];
	
	//Initialize w1 w2
	for (int i = 0; i < width; ++i) {
		for (int j = 0; j < width; ++j) {
			w1[j + width * i] = Complex(cos(2 * PI * i * j / width), -sin(2 * PI * i * j / width));
			w2[j + width * i] = Complex(1.0 / width) * Complex(cos(2 * PI * i * j / width), sin(2 * PI * i * j / width));
		}
	}

	//Start 1D DFT
	if (a.compare(b) == 0) {
		for (int i = 0; i < length; i++) {
			for (int j = 0; j < width; j++) {
				h1[j + i * width] = I[j + i * width];
				h2[j + i * width] = I[j + i * width + width * length];
				h3[j + i * width] = I[j + i * width + 2 * width * length];
				h4[j + i * width] = I[j + i * width + 3 * width * length];
				h5[j + i * width] = I[j + i * width + 4 * width * length];
				h6[j + i * width] = I[j + i * width + 5 * width * length];
				h7[j + i * width] = I[j + i * width + 6 * width * length];
				h8[j + i * width] = I[j + i * width + 7 * width * length];
				
				H1[j + i * width] = 0;
				H2[j + i * width] = 0;
				H3[j + i * width] = 0;
				H4[j + i * width] = 0;
				H5[j + i * width] = 0;
				H6[j + i * width] = 0;
				H7[j + i * width] = 0;
				H8[j + i * width] = 0;
				
			}
		}

		thread t1(Transform1D, h1, width, length, H1, w1);
		thread t2(Transform1D, h2, width, length, H2, w1);
		thread t3(Transform1D, h3, width, length, H3, w1);
		thread t4(Transform1D, h4, width, length, H4, w1);
		thread t5(Transform1D, h5, width, length, H5, w1);
		thread t6(Transform1D, h6, width, length, H6, w1);
		thread t7(Transform1D, h7, width, length, H7, w1);
		thread t8(Transform1D, h8, width, length, H8, w1);
		
		t1.join();
		t2.join();
		t3.join();
		t4.join();
		t5.join();
		t6.join();
		t7.join();
		t8.join();
		

		for (int i = 0; i < length; i++) {
			for (int j = 0; j < width; j++) {
				h[j + i * width] = H1[j + i * width];
				h[j + i * width + width * length] = H2[j + i * width];
				h[j + i * width + 2 * width * length] = H3[j + i * width];
				h[j + i * width + 3 * width * length] = H4[j + i * width];
				h[j + i * width + 4 * width * length] = H5[j + i * width];
				h[j + i * width + 5 * width * length] = H6[j + i * width];
				h[j + i * width + 6 * width * length] = H7[j + i * width];
				h[j + i * width + 7 * width * length] = H8[j + i * width];
			
			}
		}
		Transpose(h, width, height);

		//Start 2D DFT
		for (int i = 0; i < length; i++) {
			for (int j = 0; j < width; j++) {
				h1[j + i * width] = h[j + i * width];
				h2[j + i * width] = h[j + i * width + width * length];
				h3[j + i * width] = h[j + i * width + 2 * width * length];
				h4[j + i * width] = h[j + i * width + 3 * width * length];
				h5[j + i * width] = h[j + i * width + 4 * width * length];
				h6[j + i * width] = h[j + i * width + 5 * width * length];
				h7[j + i * width] = h[j + i * width + 6 * width * length];
				h8[j + i * width] = h[j + i * width + 7 * width * length];
			
				H1[j + i * width] = 0;
				H2[j + i * width] = 0;
				H3[j + i * width] = 0;
				H4[j + i * width] = 0;
				H5[j + i * width] = 0;
				H6[j + i * width] = 0;
				H7[j + i * width] = 0;
				H8[j + i * width] = 0;
				
			}
		}

		thread T1(Transform1D, h1, width, length, H1, w1);
		thread T2(Transform1D, h2, width, length, H2, w1);
		thread T3(Transform1D, h3, width, length, H3, w1);
		thread T4(Transform1D, h4, width, length, H4, w1);
		thread T5(Transform1D, h5, width, length, H5, w1);
		thread T6(Transform1D, h6, width, length, H6, w1);
		thread T7(Transform1D, h7, width, length, H7, w1);
		thread T8(Transform1D, h8, width, length, H8, w1);
		
		T1.join();
		T2.join();
		T3.join();
		T4.join();
		T5.join();
		T6.join();
		T7.join();
		T8.join();
		

		for (int i = 0; i < length; i++) {
			for (int j = 0; j < width; j++) {
				H[j + i * width] = H1[j + i * width];
				H[j + i * width + width * length] = H2[j + i * width];
				H[j + i * width + 2 * width * length] = H3[j + i * width];
				H[j + i * width + 3 * width * length] = H4[j + i * width];
				H[j + i * width + 4 * width * length] = H5[j + i * width];
				H[j + i * width + 5 * width * length] = H6[j + i * width];
				H[j + i * width + 6 * width * length] = H7[j + i * width];
				H[j + i * width + 7 * width * length] = H8[j + i * width];
				

			}
		}
		Transpose(H, width, height);

		//Write 2D DFT data.
		image.save_image_data(argv[3], H, width, height);
		clock_t end = clock();
		cout << (double)(end - start) / CLOCKS_PER_SEC << "second" << endl;
		I = H;
	}

	//Inverse 1D DFT
	if (a.compare(c) == 0) {
		for (int i = 0; i < length; i++) {
			for (int j = 0; j < width; j++) {
				h1[j + i * width] = I[j + i * width];
				h2[j + i * width] = I[j + i * width + width * length];
				h3[j + i * width] = I[j + i * width + 2 * width * length];
				h4[j + i * width] = I[j + i * width + 3 * width * length];
				h5[j + i * width] = I[j + i * width + 4 * width * length];
				h6[j + i * width] = I[j + i * width + 5 * width * length];
				h7[j + i * width] = I[j + i * width + 6 * width * length];
				h8[j + i * width] = I[j + i * width + 7 * width * length];

				H1[j + i * width] = 0;
				H2[j + i * width] = 0;
				H3[j + i * width] = 0;
				H4[j + i * width] = 0;
				H5[j + i * width] = 0;
				H6[j + i * width] = 0;
				H7[j + i * width] = 0;
				H8[j + i * width] = 0;

			}
		}

		thread t1(Transform1D, h1, width, length, H1, w1);
		thread t2(Transform1D, h2, width, length, H2, w1);
		thread t3(Transform1D, h3, width, length, H3, w1);
		thread t4(Transform1D, h4, width, length, H4, w1);
		thread t5(Transform1D, h5, width, length, H5, w1);
		thread t6(Transform1D, h6, width, length, H6, w1);
		thread t7(Transform1D, h7, width, length, H7, w1);
		thread t8(Transform1D, h8, width, length, H8, w1);

		t1.join();
		t2.join();
		t3.join();
		t4.join();
		t5.join();
		t6.join();
		t7.join();
		t8.join();


		for (int i = 0; i < length; i++) {
			for (int j = 0; j < width; j++) {
				h[j + i * width] = H1[j + i * width];
				h[j + i * width + width * length] = H2[j + i * width];
				h[j + i * width + 2 * width * length] = H3[j + i * width];
				h[j + i * width + 3 * width * length] = H4[j + i * width];
				h[j + i * width + 4 * width * length] = H5[j + i * width];
				h[j + i * width + 5 * width * length] = H6[j + i * width];
				h[j + i * width + 6 * width * length] = H7[j + i * width];
				h[j + i * width + 7 * width * length] = H8[j + i * width];

			}
		}
		Transpose(h, width, height);

		
		for (int i = 0; i < length; i++) {
			for (int j = 0; j < width; j++) {
				h1[j + i * width] = h[j + i * width];
				h2[j + i * width] = h[j + i * width + width * length];
				h3[j + i * width] = h[j + i * width + 2 * width * length];
				h4[j + i * width] = h[j + i * width + 3 * width * length];
				h5[j + i * width] = h[j + i * width + 4 * width * length];
				h6[j + i * width] = h[j + i * width + 5 * width * length];
				h7[j + i * width] = h[j + i * width + 6 * width * length];
				h8[j + i * width] = h[j + i * width + 7 * width * length];

				H1[j + i * width] = 0;
				H2[j + i * width] = 0;
				H3[j + i * width] = 0;
				H4[j + i * width] = 0;
				H5[j + i * width] = 0;
				H6[j + i * width] = 0;
				H7[j + i * width] = 0;
				H8[j + i * width] = 0;

			}
		}

		thread T1(Transform1D, h1, width, length, H1, w1);
		thread T2(Transform1D, h2, width, length, H2, w1);
		thread T3(Transform1D, h3, width, length, H3, w1);
		thread T4(Transform1D, h4, width, length, H4, w1);
		thread T5(Transform1D, h5, width, length, H5, w1);
		thread T6(Transform1D, h6, width, length, H6, w1);
		thread T7(Transform1D, h7, width, length, H7, w1);
		thread T8(Transform1D, h8, width, length, H8, w1);

		T1.join();
		T2.join();
		T3.join();
		T4.join();
		T5.join();
		T6.join();
		T7.join();
		T8.join();


		for (int i = 0; i < length; i++) {
			for (int j = 0; j < width; j++) {
				H[j + i * width] = H1[j + i * width];
				H[j + i * width + width * length] = H2[j + i * width];
				H[j + i * width + 2 * width * length] = H3[j + i * width];
				H[j + i * width + 3 * width * length] = H4[j + i * width];
				H[j + i * width + 4 * width * length] = H5[j + i * width];
				H[j + i * width + 5 * width * length] = H6[j + i * width];
				H[j + i * width + 6 * width * length] = H7[j + i * width];
				H[j + i * width + 7 * width * length] = H8[j + i * width];


			}
		}
		Transpose(H, width, height);
        I = H;

		for (int i = 0; i < length; i++) {
			for (int j = 0; j < width; j++) {
				h1[j + i * width] = I[j + i * width];
				h2[j + i * width] = I[j + i * width + width * length];
				h3[j + i * width] = I[j + i * width + 2 * width * length];
				h4[j + i * width] = I[j + i * width + 3 * width * length];
				h5[j + i * width] = I[j + i * width + 4 * width * length];
				h6[j + i * width] = I[j + i * width + 5 * width * length];
				h7[j + i * width] = I[j + i * width + 6 * width * length];
				h8[j + i * width] = I[j + i * width + 7 * width * length];
				
				H1[j + i * width] = 0;
				H2[j + i * width] = 0;
				H3[j + i * width] = 0;
				H4[j + i * width] = 0;
				H5[j + i * width] = 0;
				H6[j + i * width] = 0;
				H7[j + i * width] = 0;
				H8[j + i * width] = 0;
				
			}
		}

		thread t9(Transform1DInverse, h1, width, length, H1, w2);
		thread t10(Transform1DInverse, h2, width, length, H2, w2);
		thread t11(Transform1DInverse, h3, width, length, H3, w2);
		thread t12(Transform1DInverse, h4, width, length, H4, w2);
		thread t13(Transform1DInverse, h5, width, length, H5, w2);
		thread t14(Transform1DInverse, h6, width, length, H6, w2);
		thread t15(Transform1DInverse, h7, width, length, H7, w2);
		thread t16(Transform1DInverse, h8, width, length, H8, w2);
		
		t9.join();
		t10.join();
		t11.join();
		t12.join();
		t13.join();
		t14.join();
		t15.join();
		t16.join();
		

		for (int i = 0; i < length; i++) {
			for (int j = 0; j < width; j++) {
				h[j + i * width] = H1[j + i * width];
				h[j + i * width + width * length] = H2[j + i * width];
				h[j + i * width + 2 * width * length] = H3[j + i * width];
				h[j + i * width + 3 * width * length] = H4[j + i * width];
				h[j + i * width + 4 * width * length] = H5[j + i * width];
				h[j + i * width + 5 * width * length] = H6[j + i * width];
				h[j + i * width + 6 * width * length] = H7[j + i * width];
				h[j + i * width + 7 * width * length] = H8[j + i * width];
				

			}
		}
		Transpose(h, width, height);

		//Start 2D DFT
		for (int i = 0; i < length; i++) {
			for (int j = 0; j < width; j++) {
				h1[j + i * width] = h[j + i * width];
				h2[j + i * width] = h[j + i * width + width * length];
				h3[j + i * width] = h[j + i * width + 2 * width * length];
				h4[j + i * width] = h[j + i * width + 3 * width * length];
				h5[j + i * width] = h[j + i * width + 4 * width * length];
				h6[j + i * width] = h[j + i * width + 5 * width * length];
				h7[j + i * width] = h[j + i * width + 6 * width * length];
				h8[j + i * width] = h[j + i * width + 7 * width * length];
				
				H1[j + i * width] = 0;
				H2[j + i * width] = 0;
				H3[j + i * width] = 0;
				H4[j + i * width] = 0;
				H5[j + i * width] = 0;
				H6[j + i * width] = 0;
				H7[j + i * width] = 0;
				H8[j + i * width] = 0;
				
			}
		}

		thread T9(Transform1DInverse, h1, width, length, H1, w2);
		thread T10(Transform1DInverse, h2, width, length, H2, w2);
		thread T11(Transform1DInverse, h3, width, length, H3, w2);
		thread T12(Transform1DInverse, h4, width, length, H4, w2);
		thread T13(Transform1DInverse, h5, width, length, H5, w2);
		thread T14(Transform1DInverse, h6, width, length, H6, w2);
		thread T15(Transform1DInverse, h7, width, length, H7, w2);
		thread T16(Transform1DInverse, h8, width, length, H8, w2);
		
		T9.join();
		T10.join();
		T11.join();
		T12.join();
		T13.join();
		T14.join();
		T15.join();
		T16.join();
		

		for (int i = 0; i < length; i++) {
			for (int j = 0; j < width; j++) {
				H[j + i * width] = H1[j + i * width];
				H[j + i * width + width * length] = H2[j + i * width];
				H[j + i * width + 2 * width * length] = H3[j + i * width];
				H[j + i * width + 3 * width * length] = H4[j + i * width];
				H[j + i * width + 4 * width * length] = H5[j + i * width];
				H[j + i * width + 5 * width * length] = H6[j + i * width];
				H[j + i * width + 6 * width * length] = H7[j + i * width];
				H[j + i * width + 7 * width * length] = H8[j + i * width];
				
			}
		}
		Transpose(H, width, height);

		//Write 2D DFT data.
		image.save_image_data(argv[3], H, width, height);
		clock_t end = clock();
		cout << (double)(end - start) / CLOCKS_PER_SEC << "second" << endl;
		I = H;
	}

	delete[] H;
	delete[] h;
	delete[] h1;
	delete[] h2;
	delete[] h3;
	delete[] h4;
	delete[] h5;
	delete[] h6;
	delete[] h7;
	delete[] h8;
	delete[] H1;
	delete[] H2;
	delete[] H3;
	delete[] H4;
	delete[] H5;
	delete[] H6;
	delete[] H7;
	delete[] H8;
	delete[] w1;
	delete[] w2;
	
	return 0;
}

