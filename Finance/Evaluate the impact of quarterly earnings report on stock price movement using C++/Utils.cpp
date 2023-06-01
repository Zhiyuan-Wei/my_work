#include "Utils.hpp"


void plotResults(vector<vector<CalculationResult>>& calculated_results,int n) {
    FILE *gnuplotPipe=popen("gnuplot -persist", "w");;

    const char *title = "CAAR Outcome plot";
    const char *yLabel = "CAAR";
    fprintf(gnuplotPipe, "set grid\n");
    fprintf(gnuplotPipe, "set title '%s'\n", title);
    // fprintf(gnuplotPipe, "set arrow from 0,graph(0,0) to 0,graph(1,1) nohead lc rgb 'red'\n");
    fprintf(gnuplotPipe, "set xlabel 'Date'\nset ylabel '%s'\n", yLabel);
    fprintf(gnuplotPipe, "set xrange [-%d:%d]\n", n, n);

    int size=(int)calculated_results[0][0].CAAR.size();
    if (gnuplotPipe){
        fprintf(gnuplotPipe,"plot '-' with lines title 'Beat', '-' with lines title 'Meet',  '-' with lines title 'Miss'\n");
        fflush(gnuplotPipe);
        // Beat
        for (int i = 0; i < size; i++) {
            // fprintf(gnuplotPipe, "%f %f\n", -5.0+i, calculated_results[0][0].CAAR[i]);  // Write coordinates for line 1
            fprintf(gnuplotPipe,"%lf %lf\n",static_cast<double>(-n+i), calculated_results[0][0].CAAR[i]);
        }
        fprintf(gnuplotPipe, "e\n");
        // Meet
        for (int i = 0; i < size; i++) {
            fprintf(gnuplotPipe,"%lf %lf\n",static_cast<double>(-n+i), calculated_results[1][0].CAAR[i]);
        }
        fprintf(gnuplotPipe, "e\n");
        // Miss
        for (int i = 0; i < size; i++) {
            fprintf(gnuplotPipe,"%lf %lf\n",static_cast<double>(-n+i), calculated_results[2][0].CAAR[i]);
        }
        fprintf(gnuplotPipe, "e\n");

        fclose(gnuplotPipe);
        pclose(gnuplotPipe);


        printf("press enter to continue...");
//        getchar();


    }else {
        printf("gnuplot not found...");
    }
}

// Overload << operator for Vector, Matrix, map
ostream& operator << (ostream& out, const Vector& V) {
    for (auto x: V) out << x << " ";
    out << endl;
    return out;
}

ostream& operator << (ostream& out, const Matrix& M) {
    size_t m = M.size();
    for (size_t i = 0; i < m; ++i) {
        out << M[i];
    }
    return out;
}

// Overload vector power
Vector vec_power(const Vector& v, double n) {
    int d = v.size();
    Vector res;
    res.resize(d);
    for (size_t i = 0; i < d; ++i) res[i] = pow(v[i], n);
    return res;
}

// Overload Vector x Vector
Vector operator+(const Vector& V1, const Vector& V2) {
    size_t n = V1.size();
    Vector res;
    res.resize(n);
    for (size_t i = 0; i < n; ++i) res[i] = V1[i] + V2[i];
    return res;
}

Vector operator-(const Vector& V1, const Vector& V2) {
    size_t n = V1.size();
    Vector res;
    res.resize(n);
    for (size_t i = 0; i < n; ++i) res[i] = V1[i] - V2[i];
    return res;
}

Vector operator*(const Vector& V1, const Vector& V2) {
    size_t n = V1.size();
    Vector res;
    res.resize(n);
    for (size_t i = 0; i < n; ++i) res[i] = V1[i] * V2[i];
    return res;
}

Vector operator/(const Vector& V1, const Vector& V2) {
    size_t n = V1.size();
    Vector res;
    res.resize(n);
    for (size_t i = 0; i < n; ++i) res[i] = V1[i] / V2[i];
    return res;
}

double operator^(const Vector& V1, const Vector& V2) {
    size_t n = V1.size();
    double res = 0.0;
    for (size_t i = 0; i < n; ++i) res += V1[i] * V2[i];
    return res;
}

// Overload Matrix x Matrix
Matrix transpose(const Matrix& M) {
    size_t m = M.size(), n = M[0].size();
    Matrix t;
    t.resize(n);
    for (size_t i = 0; i < n; ++i) t[i].resize(m);
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            t[j][i] = M[i][j];
        }
    }
    return t;
}

Matrix operator^(const Matrix& V1, const Matrix& V2) {
    Matrix V2_t = transpose(V2);
    size_t m = V1.size(), n = V2_t.size();
    Matrix res;
    res.resize(m);
    for (size_t i = 0; i < m; ++i) res[i].resize(n);
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            res[i][j] = V1[i] ^ V2_t[j];
        }
    }
    return res;
}

Matrix operator*(const Matrix& V1, const Matrix& V2) {
    size_t m = V1.size(), n = V1[0].size();
    Matrix res;
    res.resize(m);
    for (size_t i = 0; i < m; ++i) res[i].resize(n);
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            res[i][j] = V1[i][j] * V2[i][j];
        }
    }
    return res;
}

Matrix operator/(const Matrix& V1, const Matrix& V2) {
    size_t m = V1.size(), n = V1[0].size();
    Matrix res;
    res.resize(m);
    for (size_t i = 0; i < m; ++i) res[i].resize(n);
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            res[i][j] = V1[i][j] / V2[i][j];
        }
    }
    return res;
}

Matrix operator+(const Matrix& V1, const Matrix& V2) {
    size_t m = V1.size(), n = V1[0].size();
    Matrix res;
    res.resize(m);
    for (size_t i = 0; i < m; ++i) res[i].resize(n);
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            res[i][j] = V1[i][j] + V2[i][j];
        }
    }
    return res;
}

Matrix operator-(const Matrix& V1, const Matrix& V2) {
    size_t m = V1.size(), n = V1[0].size();
    Matrix res;
    res.resize(m);
    for (size_t i = 0; i < m; ++i) res[i].resize(n);
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            res[i][j] = V1[i][j] - V2[i][j];
        }
    }
    return res;
}

// Overload double x Vector
Vector operator+(double d, const Vector& V2) {
    Vector temp;
    temp.assign(V2.size(), d);
    return temp + V2;
}

Vector operator-(double d, const Vector& V2) {
    Vector temp;
    temp.assign(V2.size(), d);
    return temp - V2;
}

Vector operator*(double d, const Vector& V2) {
    Vector temp;
    temp.assign(V2.size(), d);
    return temp * V2;
}

Vector operator/(double d, const Vector& V2) {
    Vector temp;
    temp.assign(V2.size(), d);
    return temp / V2;
}

// Overload Vector x double
Vector operator+(const Vector& V2, double d) {
    Vector temp;
    temp.assign(V2.size(), d);
    return V2 + temp;
}

Vector operator-(const Vector& V2, double d) {
    Vector temp;
    temp.assign(V2.size(), d);
    return V2 - temp;
}

Vector operator*(const Vector& V2, double d){
    Vector temp;
    temp.assign(V2.size(), d);
    return V2 * temp;
}

Vector operator/(const Vector& V2, double d) {
    Vector temp;
    temp.assign(V2.size(), d);
    return V2 / temp;
}

// Overload double x Matrix
Matrix operator+(double d, const Matrix& V2) {
    size_t m = V2.size(), n = V2[0].size();
    Matrix temp;
    temp.resize(m);
    for (size_t i = 0; i < m; ++i) temp[i].assign(n, d);
    return temp + V2;
}

Matrix operator-(double d, const Matrix& V2) {
    size_t m = V2.size(), n = V2[0].size();
    Matrix temp;
    temp.resize(m);
    for (size_t i = 0; i < m; ++i) temp[i].assign(n, d);
    return temp - V2;
}

Matrix operator*(double d, const Matrix& V2) {
    size_t m = V2.size(), n = V2[0].size();
    Matrix temp;
    temp.resize(m);
    for (size_t i = 0; i < m; ++i) temp[i].assign(n, d);
    return temp * V2;
}

Matrix operator/(double d, const Matrix& V2) {
    size_t m = V2.size(), n = V2[0].size();
    Matrix temp;
    temp.resize(m);
    for (size_t i = 0; i < m; ++i) temp[i].assign(n, d);
    return temp / V2;
}

// Overload Matrix x double
Matrix operator+(const Matrix& V2, double d) {
    size_t m = V2.size(), n = V2[0].size();
    Matrix temp;
    temp.resize(m);
    for (size_t i = 0; i < m; ++i) temp[i].assign(n, d);
    return V2 + temp;
}

Matrix operator-(const Matrix& V2, double d) {
    size_t m = V2.size(), n = V2[0].size();
    Matrix temp;
    temp.resize(m);
    for (size_t i = 0; i < m; ++i) temp[i].assign(n, d);
    return V2 - temp;
}

Matrix operator*(const Matrix& V2, double d) {
    size_t m = V2.size(), n = V2[0].size();
    Matrix temp;
    temp.resize(m);
    for (size_t i = 0; i < m; ++i) temp[i].assign(n, d);
    return V2 * temp;
}

Matrix operator/(const Matrix& V2, double d) {
    size_t m = V2.size(), n = V2[0].size();
    Matrix temp;
    temp.resize(m);
    for (size_t i = 0; i < m; ++i) temp[i].assign(n, d);
    return V2 / temp;
}
