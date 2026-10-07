#ifndef SYMMETRIC_EIGEN_HPP
#define SYMMETRIC_EIGEN_HPP

#include <algorithm>
#include <cmath>
#include <vector>

/// Eigenvalues w (ascending) and orthonormal eigenvectors V (columns, row major) of the real symmetric matrix A
/// (n x n, row major), by the cyclic Jacobi method. Real eigenvectors also in degenerate subspaces, which a complex
/// solver does not guarantee; the sign of each one is fixed by making its largest component positive
inline void symmetric_eigen(std::vector<double> A__, const int n__, std::vector<double>& w__, std::vector<double>& V__)
{
    V__.assign(n__ * n__, 0.);
    for( int i = 0; i < n__; ++i ) {
        V__[i * n__ + i] = 1.;
    }
    double norm2 = 0.;
    for( auto a : A__ ) {
        norm2 += a * a;
    }
    for( int sweep = 0; sweep < 100; ++sweep ) {
        double off = 0.;
        for( int p = 0; p < n__; ++p ) {
            for( int q = p + 1; q < n__; ++q ) {
                off += A__[p * n__ + q] * A__[p * n__ + q];
            }
        }
        if( off <= 1.e-32 * norm2 ) {
            break;
        }
        for( int p = 0; p < n__; ++p ) {
            for( int q = p + 1; q < n__; ++q ) {
                double apq = A__[p * n__ + q];
                if( apq == 0. ) {
                    continue;
                }
                /* rotation in the (p, q) plane that zeroes A(p, q): A <- J^T A J, V <- V J */
                double theta = (A__[q * n__ + q] - A__[p * n__ + p]) / (2. * apq);
                double t = (theta >= 0. ? 1. : -1.) / (std::abs(theta) + std::sqrt(theta * theta + 1.));
                double c = 1. / std::sqrt(t * t + 1.);
                double s = t * c;
                for( int k = 0; k < n__; ++k ) {
                    double akp = A__[k * n__ + p], akq = A__[k * n__ + q];
                    A__[k * n__ + p] = c * akp - s * akq;
                    A__[k * n__ + q] = s * akp + c * akq;
                }
                for( int k = 0; k < n__; ++k ) {
                    double apk = A__[p * n__ + k], aqk = A__[q * n__ + k];
                    A__[p * n__ + k] = c * apk - s * aqk;
                    A__[q * n__ + k] = s * apk + c * aqk;
                }
                for( int k = 0; k < n__; ++k ) {
                    double vkp = V__[k * n__ + p], vkq = V__[k * n__ + q];
                    V__[k * n__ + p] = c * vkp - s * vkq;
                    V__[k * n__ + q] = s * vkp + c * vkq;
                }
            }
        }
    }
    /* ascending order, and the sign of each eigenvector */
    std::vector<int> order(n__);
    for( int i = 0; i < n__; ++i ) {
        order[i] = i;
    }
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return A__[a * n__ + a] < A__[b * n__ + b]; });
    std::vector<double> V(n__ * n__);
    w__.resize(n__);
    for( int j = 0; j < n__; ++j ) {
        int col = order[j];
        w__[j] = A__[col * n__ + col];
        int largest = 0;
        for( int i = 0; i < n__; ++i ) {
            if( std::abs(V__[i * n__ + col]) > std::abs(V__[largest * n__ + col]) + 1.e-12 ) {
                largest = i;
            }
        }
        double sign = V__[largest * n__ + col] < 0. ? -1. : 1.;
        for( int i = 0; i < n__; ++i ) {
            V[i * n__ + j] = sign * V__[i * n__ + col];
        }
    }
    V__ = V;
}

#endif
