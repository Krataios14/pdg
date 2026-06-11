#include "pdg/linalg.hpp"
#include "test_framework.hpp"

#include <cstdint>
#include <random>

using namespace pdg;

TEST(csc_build_and_multiply) {
    // A = [1 0 2; 0 3 0] with a duplicate entry summed at (0,2)
    Triplets T;
    T.add(0, 0, 1.0);
    T.add(1, 1, 3.0);
    T.add(0, 2, 1.5);
    T.add(0, 2, 0.5);
    std::vector<int> map;
    SparseCSC A = buildCSC(2, 3, T, &map);
    CHECK(A.nnz() == 3);
    CHECK(map[2] == map[3]);  // duplicates share a slot

    double x[3] = {1.0, 2.0, 3.0};
    double y[2] = {0.0, 0.0};
    cscMulAdd(A, x, y);
    CHECK_NEAR(y[0], 1.0 + 6.0, 1e-14);
    CHECK_NEAR(y[1], 6.0, 1e-14);

    double yt[3] = {0.0, 0.0, 0.0};
    double v[2] = {1.0, 1.0};
    cscTMulAdd(A, v, yt);
    CHECK_NEAR(yt[0], 1.0, 1e-14);
    CHECK_NEAR(yt[1], 3.0, 1e-14);
    CHECK_NEAR(yt[2], 2.0, 1e-14);

    // scatterValues refresh
    T.tv[0] = 10.0;
    scatterValues(A, T, map);
    double y2[2] = {0.0, 0.0};
    cscMulAdd(A, x, y2);
    CHECK_NEAR(y2[0], 10.0 + 6.0, 1e-14);
}

TEST(dense_lu) {
    Mat A(3, 3);
    A(0, 0) = 4; A(0, 1) = 1; A(0, 2) = 0;
    A(1, 0) = 1; A(1, 1) = 3; A(1, 2) = 1;
    A(2, 0) = 0; A(2, 1) = 1; A(2, 2) = 2;
    double b[3] = {1.0, 2.0, 3.0};
    Mat Acopy = A;
    CHECK(luSolve(Acopy, b));
    // verify A x = rhs
    double r[3];
    matVec(A, b, r);
    CHECK_NEAR(r[0], 1.0, 1e-12);
    CHECK_NEAR(r[1], 2.0, 1e-12);
    CHECK_NEAR(r[2], 3.0, 1e-12);
}

namespace {
// build a random symmetric quasidefinite KKT-like matrix:
// [ H  B' ; B  -E ] with H, E diagonally dominant SPD
void makeQuasidefinite(int nx, int ny, uint32_t seed, Triplets& T, std::vector<int>& signs) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> U(-1.0, 1.0);
    const int n = nx + ny;
    signs.assign(n, 1);
    for (int i = nx; i < n; ++i) signs[i] = -1;
    for (int i = 0; i < nx; ++i) T.add(i, i, 3.0 + U(rng));
    for (int i = nx; i < n; ++i) T.add(i, i, -(3.0 + U(rng)));
    for (int i = 0; i < nx; ++i)
        for (int j = nx; j < n; ++j)
            if (U(rng) > 0.4) T.add(i, j, U(rng));
}
}  // namespace

TEST(sparse_ldl_quasidefinite) {
    const int nx = 25, ny = 18, n = nx + ny;
    Triplets T;
    std::vector<int> signs;
    makeQuasidefinite(nx, ny, 42, T, signs);
    SparseCSC K = buildCSC(n, n, T);

    LDLSolver ldl;
    ldl.analyze(K, signs);
    CHECK(ldl.factor(K, 1e-10, 1e-14));

    std::mt19937 rng(7);
    std::uniform_real_distribution<double> U(-1.0, 1.0);
    Vec b(n), x(n);
    for (auto& v : b) v = U(rng);
    const double res = ldl.solveRefine(K, b.data(), x.data(), 2);
    CHECK(res < 1e-10);

    // verify K x = b explicitly
    Vec r = b;
    symMulAdd(K, x.data(), r.data(), -1.0);
    CHECK(normInf(r.data(), n) < 1e-9);
}

TEST(sparse_ldl_with_rcm_and_dense_column) {
    // banded matrix + one dense column (mimics a free-final-time variable)
    const int n = 60;
    Triplets T;
    std::vector<int> signs(n, 1);
    for (int i = 0; i < n; ++i) T.add(i, i, 10.0);
    for (int i = 0; i + 1 < n; ++i) T.add(i, i + 1, 1.0);
    for (int i = 0; i < n - 1; ++i) T.add(i, n - 1, 0.5);  // dense last column
    SparseCSC K = buildCSC(n, n, T);

    std::vector<int> perm = rcmOrder(K, -1);
    // perm must be a permutation
    std::vector<char> seen(n, 0);
    for (int v : perm) { CHECK(v >= 0 && v < n && !seen[v]); seen[v] = 1; }

    LDLSolver ldl;
    ldl.analyze(K, signs);
    CHECK(ldl.factor(K, 0.0, 1e-14));
    Vec b(n, 1.0), x(n);
    const double res = ldl.solveRefine(K, b.data(), x.data(), 2);
    CHECK(res < 1e-10);
}

TEST(sym_mul_add) {
    // K = [2 1; 1 -3] stored as upper
    Triplets T;
    T.add(0, 0, 2.0);
    T.add(0, 1, 1.0);
    T.add(1, 1, -3.0);
    SparseCSC K = buildCSC(2, 2, T);
    double x[2] = {1.0, 2.0}, y[2] = {0.0, 0.0};
    symMulAdd(K, x, y);
    CHECK_NEAR(y[0], 4.0, 1e-14);
    CHECK_NEAR(y[1], -5.0, 1e-14);
}

TEST_MAIN()
