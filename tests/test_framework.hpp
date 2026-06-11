// Minimal single-header test harness.
#pragma once

#include <cmath>
#include <cstdio>
#include <functional>
#include <string>
#include <vector>

namespace pdgtest {

struct Registry {
    struct Entry { std::string name; std::function<void()> fn; };
    static std::vector<Entry>& tests() { static std::vector<Entry> t; return t; }
    static int& failures() { static int f = 0; return f; }
    static const char*& current() { static const char* c = ""; return c; }
};

struct Registrar {
    Registrar(const char* name, std::function<void()> fn) {
        Registry::tests().push_back({name, std::move(fn)});
    }
};

inline void reportFailure(const char* file, int line, const std::string& msg) {
    std::printf("  FAIL [%s] %s:%d  %s\n", Registry::current(), file, line, msg.c_str());
    ++Registry::failures();
}

inline int runAll() {
    int n = 0;
    for (auto& t : Registry::tests()) {
        Registry::current() = t.name.c_str();
        std::printf("[ RUN  ] %s\n", t.name.c_str());
        const int before = Registry::failures();
        t.fn();
        std::printf("[ %s ] %s\n", Registry::failures() == before ? " OK " : "FAIL", t.name.c_str());
        ++n;
    }
    std::printf("%d test(s), %d failure(s)\n", n, Registry::failures());
    return Registry::failures() == 0 ? 0 : 1;
}

}  // namespace pdgtest

#define TEST(name)                                                        \
    static void pdgtest_##name();                                         \
    static pdgtest::Registrar pdgtest_reg_##name(#name, pdgtest_##name);  \
    static void pdgtest_##name()

#define CHECK(cond)                                                       \
    do {                                                                  \
        if (!(cond)) pdgtest::reportFailure(__FILE__, __LINE__, #cond);   \
    } while (0)

#define CHECK_NEAR(a, b, tol)                                                          \
    do {                                                                               \
        const double va = (a), vb = (b), vt = (tol);                                   \
        if (!(std::fabs(va - vb) <= vt)) {                                             \
            char buf[256];                                                             \
            std::snprintf(buf, sizeof(buf), "%s ~= %s  (%.10g vs %.10g, tol %.3g)",    \
                          #a, #b, va, vb, vt);                                         \
            pdgtest::reportFailure(__FILE__, __LINE__, buf);                           \
        }                                                                              \
    } while (0)

#define TEST_MAIN() \
    int main() { return pdgtest::runAll(); }
