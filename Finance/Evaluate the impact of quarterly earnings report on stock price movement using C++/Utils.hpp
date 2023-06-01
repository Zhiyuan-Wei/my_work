#pragma once
#include <vector>
#include <map>
#include <cmath>
#include <iostream>

using namespace std;

typedef vector<double> Vector;
typedef vector<Vector> Matrix;

// struct to store information
struct CalculationResult {
    Vector AAR;
    Vector CAAR;
    CalculationResult(Vector AAR_, Vector CAAR_): AAR(AAR_), CAAR(CAAR_) {}
};

void plotResults(vector<vector<CalculationResult>>& calculated_results,int n);

// Overload vector power
Vector vec_power(const Vector& v, double n);

// Overload << operator for Vector, Matrix
ostream& operator << (ostream& out, const Vector& V);
ostream& operator << (ostream& out, const Matrix& M);

// Overload Vector x Vector
Vector operator+(const Vector& V1, const Vector& V2);
Vector operator-(const Vector& V1, const Vector& V2);
Vector operator*(const Vector& V1, const Vector& V2);
Vector operator/(const Vector& V1, const Vector& V2);
double operator^(const Vector& V1, const Vector& V2);

// Overload Matrix x Matrix
Matrix transpose(const Matrix& M);
Matrix operator^(const Matrix& V1, const Matrix& V2);
Matrix operator*(const Matrix& V1, const Matrix& V2);
Matrix operator/(const Matrix& V1, const Matrix& V2);
Matrix operator+(const Matrix& V1, const Matrix& V2);
Matrix operator-(const Matrix& V1, const Matrix& V2);

// Overload double x Vector
Vector operator+(double d, const Vector& V2);
Vector operator-(double d, const Vector& V2);
Vector operator*(double d, const Vector& V2);
Vector operator/(double d, const Vector& V2);

// Overload Vector x double
Vector operator+(const Vector& V2, double d);
Vector operator-(const Vector& V2, double d);
Vector operator*(const Vector& V2, double d);
Vector operator/(const Vector& V2, double d);

// Overload double x Matrix
Matrix operator+(double d, const Matrix& V2);
Matrix operator-(double d, const Matrix& V2);
Matrix operator*(double d, const Matrix& V2);
Matrix operator/(double d, const Matrix& V2);

// Overload Matrix x double
Matrix operator+(const Matrix& V2, double d);
Matrix operator-(const Matrix& V2, double d);
Matrix operator*(const Matrix& V2, double d);
Matrix operator/(const Matrix& V2, double d);
