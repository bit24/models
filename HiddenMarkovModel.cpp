// Code that implements solutions to the Likelihood, Decoding, and Learning problems
// based off https://web.stanford.edu/~jurafsky/slp3/A.pdf

#include <bits/stdc++.h>

using namespace std;

typedef long long ll;
typedef long double ld;
typedef pair<int, int> pi;
typedef pair<ll, ll> pl;

typedef vector<int> vi;
typedef vector<ld> vd;
typedef vector<ll> vl;

#define pb push_back
#define f first
#define s second

namespace debug {
    const int DEBUG = true;

    template<class T1, class T2>
    void pr(const pair<T1, T2> &x);

    template<class T, size_t SZ>
    void pr(const array<T, SZ> &x);

    template<class T>
    void pr(const vector<T> &x);

    template<class T>
    void pr(const set<T> &x);

    template<class T1, class T2>
    void pr(const map<T1, T2> &x);

    template<class T>
    void pr(const T &x) { if (DEBUG) cout << x; }

    template<class T, class... Ts>
    void pr(const T &first, const Ts &... rest) { pr(first), pr(rest...); }

    template<class T1, class T2>
    void pr(const pair<T1, T2> &x) { pr("{", x.f, ", ", x.s, "}"); }

    template<class T>
    void prIn(const T &x) {
        pr("{");
        bool fst = 1;
        for (auto &a : x) {
            pr(fst ? "" : ", ", a), fst = 0;
        }
        pr("}");
    }

    template<class T, size_t SZ>
    void pr(const array<T, SZ> &x) { prIn(x); }

    template<class T>
    void pr(const vector<T> &x) { prIn(x); }

    template<class T>
    void pr(const set<T> &x) { prIn(x); }

    template<class T1, class T2>
    void pr(const map<T1, T2> &x) { prIn(x); }

    void ps() { pr("\n"), cout << flush; }

    template<class Arg, class... Args>
    void ps(const Arg &first, const Args &... rest) {
        pr(first, " ");
        ps(rest...);
    }
}
using namespace debug;

const int MAXN = 1000;
const ld EPS = 1e-9;

int N; // number of states
int M; // number of observation symbols
int T;

ld A[MAXN][MAXN]; // transition probability matrix
ld B[MAXN][MAXN]; // observation likelihoods
ld PI[MAXN]; // initial probability distribution

ld alpha[MAXN][MAXN]; // used for storing dp values
ld beta[MAXN][MAXN];

ld g[MAXN][MAXN];
ld x[MAXN][MAXN][MAXN];

bool equalish(ld a, ld b) {
    ld d = max(fabs(a), fabs(b));
    return d == 0.0 || fabs(a - b) / d < EPS;
}

void prParam() {
    ps("printing A");
    for (int i = 1; i <= N; i++) {
        for (int j = 1; j <= N; j++) {
            pr(A[i][j], " ");
        }
        ps();
    }

    ps("printing B");
    for (int i = 1; i <= N; i++) {
        for (int j = 1; j <= M; j++) {
            pr(B[i][j], " ");
        }
        ps();
    }

    ps("printing PI");
    for (int i = 1; i <= N; i++) {
        pr(PI[i], " ");
    }
    ps();
    ps("fin print");
}

// calculates likelihood of obs given parameters
// assume obs padded for 1-indexing
ld likelihood(vi obs) {
    for (int i = 1; i <= N; i++) {
        alpha[1][i] = PI[i] * B[i][obs[1]];
    }

    for (int t = 2; t <= T; t++) {
        for (int i = 1; i <= N; i++) {
            ld sum = 0;
            for (int j = 1; j <= N; j++) {
                sum += alpha[t - 1][j] * A[j][i] * B[i][obs[t]];
            }
            alpha[t][i] = sum;
        }
    }

    ld ans = 0;
    for (int i = 1; i <= N; i++) {
        ans += alpha[T][i];
    }
    return ans;
}

// calculates likelihood, but in reverse from end to beginning
// assume obs padded for 1-indexing
ld reverseLikelihood(vi obs) {
    for (int i = 1; i <= N; i++) {
        beta[T][i] = 1;
    }

    for (int t = T - 1; t >= 1; t--) {
        for (int i = 1; i <= N; i++) {
            ld sum = 0;

            for (int j = 1; j <= N; j++) {
                sum += A[i][j] * B[j][obs[t + 1]] * beta[t + 1][j];
            }
            beta[t][i] = sum;
        }
    }

    ld ans = 0;
    for (int i = 1; i <= N; i++) {
        ans += PI[i] * B[i][obs[1]] * beta[1][i];
    }
    return ans;
}

// calculates most likely hidden state sequence given obs and parameters
// Viterbi algorithm, difference from likelihood is max instead of sum
// assume obs padded for 1-indexing
ld decoding(vi obs) {
    for (int i = 1; i <= N; i++) {
        alpha[1][i] = PI[i] * B[i][obs[1]];
    }

    for (int t = 2; t <= T; t++) {
        for (int i = 1; i <= N; i++) {
            ld cMax = 0;
            for (int j = 1; j <= N; j++) {
                cMax = max(cMax, alpha[t - 1][j] * A[j][i] * B[i][obs[t]]);
            }
            alpha[t][i] = cMax;
        }
    }

    ld ans = 0;
    for (int i = 1; i <= N; i++) {
        ans = max(ans, alpha[T][i]);
    }

    // TODO: implement backtracing

    return ans;
}

default_random_engine re;
uniform_real_distribution<ld> unifLD(0, 1);

// randomizes A and B with weights between 0 and 1
// technically an invalid probability distribution but should converge to a valid one
void randomInit() {
    for (int i = 1; i <= N; i++) {
        for (int j = 1; j <= N; j++) {
            A[i][j] = unifLD(re);
        }
    }

    for (int i = 1; i <= N; i++) {
        for (int j = 1; j <= M; j++) {
            B[i][j] = unifLD(re);
        }
    }
}

int escCnt = 1e4;

// uses expectation maximization algorithm
void learning(vi obs) { // takes O(N^2*T) time
    randomInit();

    bool converge = false;
    while (!converge && escCnt-- > 0) {
        ld pObs = likelihood(obs);
        ld pObs2 = reverseLikelihood(obs);
//        ps(pObs, pObs2);
        assert(equalish(pObs, pObs2)); // sanity check

        for (int t = 1; t <= T; t++) {
            for (int i = 1; i <= N; i++) {
                g[t][i] = alpha[t][i] * beta[t][i] / pObs; // probability of being in state i at time t
            }
        }

        for (int t = 1; t < T; t++) {
            for (int i = 1; i <= N; i++) {
                for (int j = 1; j <= N; j++) {
//                    ps(obs[t + 1]);
                    x[t][i][j] = alpha[t][i] * A[i][j] * B[j][obs[t + 1]] * beta[t + 1][j] /
                                 pObs; // probability of transitioning from state i to state j between time t and t+1
                }
            }
        }

        converge = true;

        for (int i = 1; i <= N; i++) {
            ld den = 0; // counts total number of transitions from i
            for (int t = 1; t <= T; t++) {
                for (int k = 1; k <= N; k++) {
                    den += x[t][i][k];
                }
            }


            for (int j = 1; j <= N; j++) {
                ld num = 0;
                for (int t = 1; t <= T; t++) {
                    num += x[t][i][j];
                }

                ld val = num / den;
                if (converge && !equalish(A[i][j], val)) {
                    converge = false;
                }
                A[i][j] = val; // update A[i][j]
            }
        }

        for (int i = 1; i <= N; i++) {
            for (int sym = 1; sym <= M; sym++) {
                ld num = 0, den = 0;
                for (int t = 1; t <= T; t++) {
                    if (obs[t] == sym) {
                        num += g[t][i];
                    }
                    den += g[t][i];
                }
                ld val = num / den;
                if (converge && !equalish(B[i][sym], val)) {
                    converge = false;
                }
                B[i][sym] = val;
            }
        }

        // debug
//        prParam();
    }
}

int main() {
    freopen("obs.in", "r", stdin);
    N = 2;
    M = 3;
    T = 33;

    uniform_int_distribution<int> unifINT(1, M);

    PI[1] = .5;
    PI[2] = .5;
//    PI[1] = 1;

    vi obs(34);
    obs[0] = -1;

    for (int i = 1; i <= 33; i++) {
        cin >> obs[i];
    }
    ps(obs);

    learning(obs);
    prParam(); // compared and approximately the same as Eisner's ice cream data

    ps(likelihood(obs));
}