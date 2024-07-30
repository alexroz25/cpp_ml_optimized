#ifndef TEST_H
#define TEST_H
#include "Matrix.h"
#include <numeric>
#include <random>
#include <algorithm>
#include <chrono>

// #define BENCHMARK

void eval(bool a, const char* func) {
	std::cout << func << ": " << (a ? "PASS\n" : "FAIL\n");
}

void TEST1() { // Matrix()
	Matrix<double> m;
	eval(m.values().size() == 0 && m.rows() == 0 && m.cols() == 0, __FUNCTION__);
}

void TEST2() { // Matrix(r, c)
	Matrix<float> m(10, 5);
	eval(m.values().size() == 50 && m.rows() == 10 && m.cols() == 5, __FUNCTION__);
}

void TEST3() { // Matrix(r, c, v)
	Matrix<char> m(4, 2, { 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H' });
	eval(m.values().size() == 8 && m.rows() == 4 && m.cols() == 2 && m.at(0, 0) == 'A' && m.at(3, 0) == 'G' && m.at(3, 1) == 'H', __FUNCTION__);
}

void TEST4() { // at()
	Matrix<int> m(3, 4, { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 });
	bool a = true;
	int i = 1;
	for (int r = 0; r < 3; ++r)
		for (int c = 0; c < 4; ++c)
			a &= m.at(r, c) == i++;
	eval(a, __FUNCTION__);
}

void TEST5() { // multiply(other)
	Matrix<int> a(3, 2, { 2, 5, -2, 0, 10, -11 });
	Matrix<int> b(2, 3, { 1, 2, -3, -1, -1, 5 });
	Matrix<int> out = a.multiply(b);
	eval(out.at(0, 0) == -3 && out.at(0, 1) == -1 && out.at(0, 2) == 19 && 
		 out.at(1, 0) == -2 && out.at(1, 1) == -4 && out.at(1, 2) == 6  && 
		 out.at(2, 0) == 21 && out.at(2, 1) == 31 && out.at(2, 2) == -85, __FUNCTION__);
}

void TEST6() { // multiply_transpose(otherT)
	Matrix<int> a(3, 2, { 2, 5, -2, 0, 10, -11 });
	Matrix<int> b(4, 2, { 1, 2, -3, -1, -1, 5, 0, 9 });
	Matrix<int> out = a.multiply_transpose(b);
	eval(out.at(0, 0) == 12  && out.at(0, 1) == -11 && out.at(0, 2) == 23  && out.at(0, 3) == 45 &&
		 out.at(1, 0) == -2  && out.at(1, 1) == 6   && out.at(1, 2) == 2   && out.at(1, 3) == 0  &&
		 out.at(2, 0) == -12 && out.at(2, 1) == -19 && out.at(2, 2) == -65 && out.at(2, 3) == -99, __FUNCTION__);
}

void TEST7() { // transpose_multiply(other)
	Matrix<int> a(3, 2, { 2, 5, -2, 0, 10, -11 });
	Matrix<int> b(3, 3, { 1, 2, -3, -1, -1, 5, -3, 0, -9 });
	Matrix<int> out = a.transpose_multiply(b);
	eval(out.at(0, 0) == -26 && out.at(0, 1) == 6  && out.at(0, 2) == -106 &&
		 out.at(1, 0) == 38  && out.at(1, 1) == 10 && out.at(1, 2) == 84, __FUNCTION__);
}

void BENCHMARK1(int n) {
	std::vector<int> v1(n * n);
	std::iota(v1.begin(), v1.end(), 1);
	std::vector<int> v2 = v1;
	std::shuffle(v1.begin(), v1.end(), std::default_random_engine(time(0)));
	std::shuffle(v2.begin(), v2.end(), std::default_random_engine(time(0)));

	Matrix<int> m1(n, n, v1);
	Matrix<int> m2(n, n, v2);

	std::cout << n << " x " << n << " BENCHMARK: ";
	auto t1 = std::chrono::high_resolution_clock::now();
	Matrix<int> ans = m1.transpose_multiply(m2);
	auto t2 = std::chrono::high_resolution_clock::now();

	std::chrono::duration<double> ms_double = t2 - t1;
	std::cout << ms_double.count() << "s\n";
}

void TEST() {
	TEST1();
	TEST2();
	TEST3();
	TEST4();
	TEST5();
	TEST6();
	TEST7();

#ifdef BENCHMARK
	BENCHMARK1(4096);
#endif

}



#endif