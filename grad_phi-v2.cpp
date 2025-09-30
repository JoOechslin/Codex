/* UPDATED COMPILER CALL INCLUDING FFTW3

/opt/homebrew/opt/llvm/bin/clang++ -O3 -std=c++17 -arch arm64 \
  -Xpreprocessor -fopenmp \
  -I/opt/homebrew/opt/libomp/include \
  -I/opt/homebrew/include \
  -L/opt/homebrew/opt/libomp/lib -lomp \
  -L/opt/homebrew/lib -lfftw3 -lfftw3_threads \
  -Wl,-rpath,/opt/homebrew/lib \
  -bundle -undefined dynamic_lookup \
  $(python3 -m pybind11 --includes) \
  grad_phi-v2.cpp -o grad_phi$(python3-config --extension-suffix)

*/

/* Version history
grad_phi-v2.cpp
2025-09-29 Added fof_groups function. Significant performance enhancement over Python version of same function
           Further optimized cooling_heating_step_cpp by paralleling several Ng**3 loops
2025-09-28 Amended PMEngine::step so that it releases the GIL. Changed combined_step to run ::step and cooling_heating_step_cpp in parallel
2025-09-27 Introduced combined_step function

grad_phi.cpp
2025-09    Original version
*/

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <algorithm>
#include <complex>
#include <vector>
#include <memory>
#include <atomic>
#include <array>
#include <limits>
#include <cstring>
#include <thread>
#include <chrono>
#include <tuple>
#include <cstdint>
#include <fftw3.h>
#include <omp.h>

using cplx = std::complex<double>;
namespace py = pybind11;

using Vec3Grid = std::array<py::array_t<double>, 3>;

struct ViscosityResult {
    py::array_t<double> q;
    py::array_t<double> div_v;
    double S_max;
};

namespace {

using ArrD = py::array_t<double, py::array::c_style | py::array::forcecast>;

constexpr double Mpc_to_cm   = 3.085677581e24;
constexpr double Myr_to_s    = 3.15576e13;
constexpr double Msun_to_g   = 1.98847e33;
constexpr double m_p         = 1.6726219e-24;
constexpr double k_B         = 1.380649e-16;
constexpr double gamma_ad    = 5.0 / 3.0;
constexpr double mu_ion      = 0.59;
constexpr double mu_neutral  = 1.22;
constexpr double X_H         = 0.76;
constexpr double G_cgs       = 6.67430e-8;
constexpr double sigma_T     = 6.6524587158e-25;
constexpr double m_e         = 9.10938356e-28;
constexpr double c_light     = 2.99792458e10;
constexpr double a_rad       = 7.5657e-15;
constexpr double Mpc_to_cm_cubed = Mpc_to_cm * Mpc_to_cm * Mpc_to_cm;

inline double safe_log10(double v) {
    return std::log10(std::max(v, 1e-99));
}

struct CellKey {
    int64_t x = 0;
    int64_t y = 0;
    int64_t z = 0;
};

inline bool cell_key_equal(const CellKey &a, const CellKey &b) {
    return a.x == b.x && a.y == b.y && a.z == b.z;
}

inline bool cell_key_less(const CellKey &a, const CellKey &b) {
    if (a.x != b.x) return a.x < b.x;
    if (a.y != b.y) return a.y < b.y;
    return a.z < b.z;
}



struct CosmologyParams {
    double H0_cos = 0.0;      // 1 / Myr
    double Omega_m = 0.0;
    double Omega_r = 0.0;
    double Omega_k = 0.0;
    double Omega_lambda = 0.0;
    bool is_set = false;
};

CosmologyParams g_cosmo;

inline double E_of_a(double a) {
    if (!g_cosmo.is_set) {
        throw std::runtime_error("Cosmology parameters not initialised. Call set_cosmology_params.");
    }
    const double a2 = a * a;
    const double a3 = a2 * a;
    const double a4 = a2 * a2;
    return std::sqrt(g_cosmo.Omega_r / a4 + g_cosmo.Omega_m / a3 + g_cosmo.Omega_k / a2 + g_cosmo.Omega_lambda);
}

void set_cosmology_params_cpp(double H0_cos, double Omega_m, double Omega_r,
                              double Omega_k, double Omega_lambda) {
    g_cosmo.H0_cos = H0_cos;
    g_cosmo.Omega_m = Omega_m;
    g_cosmo.Omega_r = Omega_r;
    g_cosmo.Omega_k = Omega_k;
    g_cosmo.Omega_lambda = Omega_lambda;
    g_cosmo.is_set = true;
}

inline double T_CMB_of_z(double z) {
    return 2.7255 * (1.0 + z);
}

inline double u_CMB(double z, double mu) {
    return 1.5 * k_B * T_CMB_of_z(z) / (mu * m_p);
}

inline std::pair<double, double> lambda_S_Compton(double z, double mu, double x_e = 2e-4,
                                                  double fHe = 0.079, double w = 1.0) {
    const double Tcmb = T_CMB_of_z(z);
    const double A = (8.0 * sigma_T * a_rad * std::pow(Tcmb, 4)) / (3.0 * m_e * c_light);
    const double lam = w * A * (x_e / (1.0 + x_e + fHe));
    const double S = lam * u_CMB(z, mu);
    return {lam, S};
}

inline double Zsolar_of_nH_z(double nH, double z, double n0 = 7e-4,
                             double alpha = 1.1, double Z_cap = 1.0) {
    double Z_floor = 0.05 * std::pow(10.0, -0.15 * z);
    double Z_max   = Z_cap * std::pow(10.0, -0.15 * z);
    double s = std::pow(std::max(nH, 1e-12) / n0, alpha);
    return Z_floor + (Z_max - Z_floor) * (s / (1.0 + s));
}

inline double u_from_T(double T, double mu = mu_ion, double gamma = gamma_ad) {
    return (k_B * T) / ((gamma - 1.0) * mu * m_p);
}

inline double T_from_u(double u, double mu = mu_ion, double gamma = gamma_ad) {
    return (gamma - 1.0) * u * mu * m_p / k_B;
}

struct AxisInfo {
    std::vector<double> grid;
    double minv = 0.0;
    double maxv = 0.0;
    bool log_input = false;
    bool uniform = false;
    double inv_dx = 0.0;
    double origin = 0.0;
};

AxisInfo make_axis(const ArrD &arr, bool log_input) {
    AxisInfo axis;
    axis.log_input = log_input;
    py::buffer_info info = arr.request();
    const double *ptr = static_cast<const double *>(info.ptr);
    axis.grid.resize(static_cast<size_t>(info.size));
    for (size_t i = 0; i < axis.grid.size(); ++i) {
        double val = ptr[i];
        axis.grid[i] = log_input ? safe_log10(val) : val;
    }
    if (axis.grid.empty()) {
        throw std::runtime_error("Axis grid must contain at least one value");
    }
    axis.minv = axis.grid.front();
    axis.maxv = axis.grid.front();
    for (double v : axis.grid) {
        axis.minv = std::min(axis.minv, v);
        axis.maxv = std::max(axis.maxv, v);
    }
    axis.origin = axis.grid.front();
    axis.uniform = false;
    axis.inv_dx = 0.0;
    if (axis.grid.size() >= 2) {
        const size_t nseg = axis.grid.size() - 1;
        double mean = 0.0;
        for (size_t i = 0; i < nseg; ++i) {
            mean += axis.grid[i + 1] - axis.grid[i];
        }
        mean /= static_cast<double>(nseg);
        double tol = std::max(1e-8, std::abs(mean) * 1e-6);
        axis.uniform = std::abs(mean) > 0.0;
        if (axis.uniform) {
            for (size_t i = 0; i < nseg; ++i) {
                double diff = axis.grid[i + 1] - axis.grid[i];
                if (std::abs(diff - mean) > tol) {
                    axis.uniform = false;
                    break;
                }
            }
        }
        if (axis.uniform) {
            axis.inv_dx = 1.0 / mean;
        }
    }
    return axis;
}

inline double transform_value(const AxisInfo &axis, double value) {
    double v = axis.log_input ? safe_log10(value) : value;
    if (v <= axis.minv) return axis.minv;
    if (v >= axis.maxv) return axis.maxv;
    return v;
}

inline void index_and_weight(const AxisInfo &axis, double value, size_t idx[2], double weight[2]) {
    const size_t count = axis.grid.size();
    if (count == 1) {
        idx[0] = 0;
        idx[1] = 0;
        weight[0] = 1.0;
        weight[1] = 0.0;
        return;
    }

    if (value <= axis.grid.front()) {
        idx[0] = 0;
        idx[1] = 1;
        weight[0] = 1.0;
        weight[1] = 0.0;
        return;
    }
    if (value >= axis.grid.back()) {
        idx[0] = count - 2;
        idx[1] = count - 1;
        weight[0] = 0.0;
        weight[1] = 1.0;
        return;
    }

    if (axis.uniform) {
        double pos = (value - axis.origin) * axis.inv_dx;
        const double max_pos = static_cast<double>(count - 1) - 1e-12;
        if (pos < 0.0) pos = 0.0;
        if (pos > max_pos) pos = max_pos;
        size_t i0 = static_cast<size_t>(std::floor(pos));
        double frac = pos - static_cast<double>(i0);
        idx[0] = i0;
        idx[1] = i0 + 1;
        weight[0] = 1.0 - frac;
        weight[1] = frac;
        return;
    }

    auto upper = std::upper_bound(axis.grid.begin(), axis.grid.end(), value);
    size_t i1 = static_cast<size_t>(upper - axis.grid.begin());
    size_t i0 = i1 - 1;
    double x0 = axis.grid[i0];
    double x1 = axis.grid[i1];
    double frac = (value - x0) / (x1 - x0);
    idx[0] = i0;
    idx[1] = i1;
    weight[0] = 1.0 - frac;
    weight[1] = frac;
}

struct LambdaTable4D {
    std::array<AxisInfo, 4> axes;
    std::array<size_t, 4> shape;
    std::array<size_t, 4> stride;
    std::vector<double> values;

    double evaluate(double z, double Z, double nH, double T) const {
        const double query[4] = {z, Z, nH, T};
        size_t idx[4][2];
        double weight[4][2];
        int limits[4];
        for (int axis = 0; axis < 4; ++axis) {
            double v = transform_value(axes[axis], query[axis]);
            index_and_weight(axes[axis], v, idx[axis], weight[axis]);
            limits[axis] = axes[axis].grid.size() > 1 ? 2 : 1;
        }

        double accum = 0.0;
        for (int a = 0; a < limits[0]; ++a) {
            size_t iz = idx[0][a];
            double wz = weight[0][a];
            size_t base_z = iz * stride[0];
            for (int b = 0; b < limits[1]; ++b) {
                size_t iZ = idx[1][b];
                double wZ = weight[1][b];
                size_t base_Z = base_z + iZ * stride[1];
                double w_zZ = wz * wZ;
                for (int c = 0; c < limits[2]; ++c) {
                    size_t inH = idx[2][c];
                    double wnH = weight[2][c];
                    size_t base_nH = base_Z + inH * stride[2];
                    double w_all = w_zZ * wnH;
                    for (int d = 0; d < limits[3]; ++d) {
                        size_t iT = idx[3][d];
                        double wT = weight[3][d];
                        size_t linear = base_nH + iT;
                        accum += w_all * wT * values[linear];
                    }
                }
            }
        }
        return accum;
    }
};

struct LambdaTable1D {
    AxisInfo axis;
    std::vector<double> values;

    double evaluate(double T) const {
        size_t idx[2];
        double w[2];
        double v = transform_value(axis, T);
        index_and_weight(axis, v, idx, w);
        return w[0] * values[idx[0]] + w[1] * values[idx[1]];
    }
};

std::shared_ptr<LambdaTable4D> g_lambda_table;
std::shared_ptr<LambdaTable1D> g_lambda_collis;

inline std::shared_ptr<LambdaTable4D> require_lambda_table() {
    auto table = g_lambda_table;
    if (!table) {
        throw std::runtime_error("Cooling Lambda table not initialised. Call set_lambda_table.");
    }
    return table;
}

inline std::shared_ptr<LambdaTable1D> require_lambda_collis() {
    auto table = g_lambda_collis;
    if (!table) {
        throw std::runtime_error("Collisional Lambda table not initialised. Call set_lambda_collis.");
    }
    return table;
}

struct InputData {
    ArrD array;
    py::buffer_info info;
    const double *ptr = nullptr;
    size_t size = 0;
    bool is_scalar = true;
    std::vector<ssize_t> shape;

    explicit InputData(py::object obj) : array(ArrD::ensure(obj)) {
        if (!array) {
            throw std::runtime_error("Expected array-like input convertible to float64");
        }
        info = array.request();
        ptr = static_cast<const double *>(info.ptr);
        size = static_cast<size_t>(info.size);
        is_scalar = (info.ndim == 0) || (size == 1);
        if (!is_scalar) {
            shape.assign(info.shape.begin(), info.shape.end());
        }
    }

    double value(size_t idx) const {
        return is_scalar ? ptr[0] : ptr[idx];
    }
};

struct BroadcastInfo {
    std::vector<ssize_t> shape;
    size_t size = 1;
    bool has_shape = false;
};

BroadcastInfo resolve_broadcast(const std::vector<const InputData *> &inputs) {
    BroadcastInfo out;
    for (const InputData *inp : inputs) {
        if (!inp || inp->is_scalar) {
            continue;
        }
        if (!out.has_shape) {
            out.shape = inp->shape;
            out.size = inp->size;
            out.has_shape = true;
        } else if (inp->shape != out.shape) {
            throw std::runtime_error("Input arrays must have identical shape or be scalar");
        }
    }
    if (!out.has_shape) {
        out.size = 1;
    }
    return out;
}

int parse_threads(const py::object &threads_obj) {
    if (threads_obj.is_none()) {
        return omp_get_max_threads();
    }
    int t = threads_obj.cast<int>();
    return t > 0 ? t : omp_get_max_threads();
}

void set_lambda_table_cpp(const py::object z_grid_obj,
                          const py::object Z_grid_obj,
                          const py::object nH_grid_obj,
                          const py::object T_grid_obj,
                          const py::object lambda_obj) {
    ArrD z_grid = ArrD::ensure(z_grid_obj);
    ArrD Z_grid = ArrD::ensure(Z_grid_obj);
    ArrD nH_grid = ArrD::ensure(nH_grid_obj);
    ArrD T_grid = ArrD::ensure(T_grid_obj);
    ArrD lambda = ArrD::ensure(lambda_obj);
    if (!z_grid || !Z_grid || !nH_grid || !T_grid || !lambda) {
        throw std::runtime_error("Failed to convert inputs to float64 arrays");
    }

    auto table = std::make_shared<LambdaTable4D>();
    table->axes = {make_axis(z_grid, false),
                   make_axis(Z_grid, false),
                   make_axis(nH_grid, true),
                   make_axis(T_grid, true)};

    py::buffer_info info = lambda.request();
    if (info.ndim != 4) {
        throw std::runtime_error("Lambda table must be 4D (nz, nZ, nNH, nT)");
    }
    table->shape = {static_cast<size_t>(info.shape[0]),
                    static_cast<size_t>(info.shape[1]),
                    static_cast<size_t>(info.shape[2]),
                    static_cast<size_t>(info.shape[3])};

    for (int axis = 0; axis < 4; ++axis) {
        if (table->shape[axis] != table->axes[axis].grid.size()) {
            throw std::runtime_error("Lambda table shape does not match grid length");
        }
    }

    table->stride[3] = 1;
    table->stride[2] = table->shape[3] * table->stride[3];
    table->stride[1] = table->shape[2] * table->stride[2];
    table->stride[0] = table->shape[1] * table->stride[1];

    const double *src = static_cast<const double *>(info.ptr);
    table->values.assign(src, src + info.size);

    std::atomic_store(&g_lambda_table, std::move(table));
}

void set_lambda_collis_cpp(const py::object T_grid_obj, const py::object lambda_obj) {
    ArrD T_grid = ArrD::ensure(T_grid_obj);
    ArrD lambda = ArrD::ensure(lambda_obj);
    if (!T_grid || !lambda) {
        throw std::runtime_error("Failed to convert collisional inputs to float64 arrays");
    }

    auto table = std::make_shared<LambdaTable1D>();
    table->axis = make_axis(T_grid, true);

    py::buffer_info info = lambda.request();
    if (info.ndim != 1) {
        throw std::runtime_error("Collisional Lambda table must be 1D");
    }
    if (static_cast<size_t>(info.shape[0]) != table->axis.grid.size()) {
        throw std::runtime_error("Collisional table shape does not match T grid length");
    }

    const double *src = static_cast<const double *>(info.ptr);
    table->values.assign(src, src + info.size);

    std::atomic_store(&g_lambda_collis, std::move(table));
}

py::object lambda_T_nH_Z_z_cpp(const py::object T_obj,
                               const py::object nH_obj,
                               const py::object Z_obj,
                               const py::object z_obj,
                               const py::object threads_obj = py::none()) {
    auto table = std::atomic_load(&g_lambda_table);
    if (!table) {
        throw std::runtime_error("Lambda cooling table not initialised. Call set_lambda_table first.");
    }

    InputData T_data(T_obj);
    InputData nH_data(nH_obj);
    InputData Z_data(Z_obj);
    InputData z_data(z_obj);

    BroadcastInfo binfo = resolve_broadcast({&T_data, &nH_data, &Z_data, &z_data});
    size_t N = binfo.size;

    std::vector<double> output(N);
    if (N == 1) {
        output[0] = table->evaluate(z_data.value(0),
                                    Z_data.value(0),
                                    nH_data.value(0),
                                    T_data.value(0));
    } else {
        int threads = parse_threads(threads_obj);
        #pragma omp parallel for schedule(static) num_threads(threads)
        for (py::ssize_t i = 0; i < static_cast<py::ssize_t>(N); ++i) {
            output[static_cast<size_t>(i)] = table->evaluate(z_data.value(static_cast<size_t>(i)),
                                                             Z_data.value(static_cast<size_t>(i)),
                                                             nH_data.value(static_cast<size_t>(i)),
                                                             T_data.value(static_cast<size_t>(i)));
        }
    }

    if (!binfo.has_shape) {
        return py::float_(output[0]);
    }

    py::array_t<double> result(py::array::ShapeContainer(binfo.shape));
    std::memcpy(result.mutable_data(), output.data(), N * sizeof(double));
    return result;
}

py::object lambda_collis_cpp(const py::object T_obj, py::object threads_obj = py::none()) {
    auto table = std::atomic_load(&g_lambda_collis);
    if (!table) {
        throw std::runtime_error("Collisional cooling table not initialised. Call set_lambda_collis first.");
    }

    InputData T_data(T_obj);
    BroadcastInfo binfo = resolve_broadcast({&T_data});
    size_t N = binfo.size;

    std::vector<double> output(N);
    if (N == 1) {
        output[0] = table->evaluate(T_data.value(0));
    } else {
        int threads = parse_threads(threads_obj);
        #pragma omp parallel for schedule(static) num_threads(threads)
        for (py::ssize_t i = 0; i < static_cast<py::ssize_t>(N); ++i) {
            output[static_cast<size_t>(i)] = table->evaluate(T_data.value(static_cast<size_t>(i)));
        }
    }

    if (!binfo.has_shape) {
        return py::float_(output[0]);
    }

    py::array_t<double> result(py::array::ShapeContainer(binfo.shape));
    std::memcpy(result.mutable_data(), output.data(), N * sizeof(double));
    return result;
}

} // namespace

py::array_t<double> compute_phi_grad(const py::array_t<double> x_in, double eps, int n_threads=8, double L=0.0) {
    // Get array info
    py::buffer_info buf = x_in.request();
    if (buf.ndim != 2 || buf.shape[1] != 3)
        throw std::runtime_error("Input must be shape (N, 3)");

    const size_t N = buf.shape[0];
    const double *x = static_cast<double *>(buf.ptr);

    // Allocate output array (N, 3)
    py::array_t<double> grad_out({N, (size_t)3});
    py::buffer_info grad_buf = grad_out.request();
    double *grad = static_cast<double *>(grad_buf.ptr);

    const double eps2 = eps * eps;

    //Parallel over particles
    #pragma omp parallel for num_threads(n_threads) schedule(static)
    for (long i = 0; i < (long)N; i++) {
        double gx = 0.0, gy = 0.0, gz = 0.0;

        double xi = x[3*i + 0];
        double yi = x[3*i + 1];
        double zi = x[3*i + 2];

        for (size_t j = 0; j < N; j++) {
            if (j == (size_t)i) continue;
            double dx = x[3*j + 0] - xi;
            double dy = x[3*j + 1] - yi;
            double dz = x[3*j + 2] - zi;

            if (L > 0) {
                dx -= L * std::round(dx / L);
                dy -= L * std::round(dy / L);
                dz -= L * std::round(dz / L);
            }

            double dist2 = dx*dx + dy*dy + dz*dz + eps2;
            double inv_r3 = 1.0 / (dist2 * std::sqrt(dist2));  // r^-3

            gx += dx * inv_r3;
            gy += dy * inv_r3;
            gz += dz * inv_r3;
        }

        grad[3*i + 0] = gx;
        grad[3*i + 1] = gy;
        grad[3*i + 2] = gz;
    }

    return grad_out;
}

std::pair<double, double> compute_energies(
    const py::array_t<double> x_in,          // Positions [N,3]
    const py::array_t<double> u_in,          // Peculiar velocities [N,3]
    const py::array_t<double> center_in,     // Center [3]
    double eps,                        // Softening length
    double H_a,                        // H_a in 1/Myr
    int n_threads=8
) {
    py::buffer_info buf_x = x_in.request();
    py::buffer_info buf_u = u_in.request();
    py::buffer_info buf_c = center_in.request();

    const size_t N = buf_x.shape[0];
    const double* x = static_cast<double*>(buf_x.ptr);
    const double* u = static_cast<double*>(buf_u.ptr);
    const double* center = static_cast<double*>(buf_c.ptr);

    double KE = 0.0;
    double PE = 0.0;

    #pragma omp parallel for num_threads(n_threads) schedule(static) reduction(+:KE,PE)
    for (size_t i = 0; i < N; i++) {
        // ---- Compute KE ----
        double dx = x[3*i+0] - center[0];
        double dy = x[3*i+1] - center[1];
        double dz = x[3*i+2] - center[2];

        double vx = H_a * dx + u[3*i+0];
        double vy = H_a * dy + u[3*i+1];
        double vz = H_a * dz + u[3*i+2];

        KE += 0.5 * (vx*vx + vy*vy + vz*vz);

        // ---- Compute PE ----
        double xi = x[3*i+0];
        double yi = x[3*i+1];
        double zi = x[3*i+2];

        double pot_i = 0.0;
        for (size_t j = 0; j < N; j++) {
            if (i == j) continue;
            double xj = x[3*j+0];
            double yj = x[3*j+1];
            double zj = x[3*j+2];

            double dx = xi - xj;
            double dy = yi - yj;
            double dz = zi - zj;
            double r2 = dx*dx + dy*dy + dz*dz + eps*eps;
            double r = std::sqrt(r2);

            pot_i += 1.0 / r;
        }
        PE -= 0.5 * pot_i;  // factor 0.5 to avoid double counting
    }
    return std::make_pair(KE, PE); //Multiply KE by mass and PE by G * mass
}

double velocity_stat(const py::array_t<double> u_in, int n_threads=8) {
    py::buffer_info buf_u = u_in.request();
    const size_t N = buf_u.shape[0];
    const double* u = static_cast<double*>(buf_u.ptr);
    double v_max = 0.0;

    #pragma omp parallel for num_threads(n_threads) schedule(static) reduction(max:v_max)
    for (size_t i = 0; i < N; i++) {
        double vx = u[3*i+0];
        double vy = u[3*i+1];
        double vz = u[3*i+2];
        v_max = std::max(std::sqrt(vx*vx + vy*vy + vz*vz), v_max);
    }
    return v_max; //highest v_pec
}

namespace {

// Forward declarations for raw kernels defined later in this file
void compute_delta_field_TSC_raw(const double* pos,
                                 ssize_t Np,
                                 int Ngrid,
                                 double L,
                                 int n_threads,
                                 double* delta_out);

void compute_delta_field_CIC_raw(const double* pos,
                                 ssize_t Np,
                                 int Ngrid,
                                 double L,
                                 int n_threads,
                                 double* delta_out);

void gather_scalar_CIC_raw(const double* grid,
                           int Ngrid,
                           double Lbox,
                           double dx,
                           const double* pos,
                           ssize_t N,
                           int n_threads,
                           ptrdiff_t out_stride,
                           double* out);

void tsc_gather_scalar_TSC_raw(const double* grid,
                           int Ngrid,
                           double Lbox,
                           double dx,
                           const double* pos,
                           ssize_t N,
                           int n_threads,
                           ptrdiff_t out_stride,
                           double* out);

void interpolate_forces_CIC_raw(const double* Fx,
                            const double* Fy,
                            const double* Fz,
                            int Ngrid,
                            double Lbox,
                            double dx,
                            const double* pos,
                            ssize_t N,
                            int n_threads,
                            double* out);

void interpolate_forces_TSC_raw(const double* Fx,
                                const double* Fy,
                                const double* Fz,
                                int Ngrid,
                                double Lbox,
                                double dx,
                                const double* pos,
                                ssize_t N,
                                int n_threads,
                                double* out);

void interpolate_potential_CIC_raw(const double* grid,
                               int Ngrid,
                               double Lbox,
                               double dx,
                               const double* pos,
                               ssize_t N,
                               int n_threads,
                               double* out);

} // namespace

py::array_t<double> compute_delta_field_CIC(const py::array_t<double, py::array::c_style | py::array::forcecast>& positions,
                                            int Ngrid, double L, int n_threads) {
    py::buffer_info buf = positions.request();
    if (buf.ndim != 2 || buf.shape[1] != 3)
        throw std::runtime_error("Input must be of shape (N, 3)");

    const ssize_t Np = buf.shape[0];
    py::array_t<double> delta({Ngrid, Ngrid, Ngrid});
    compute_delta_field_CIC_raw(static_cast<const double*>(buf.ptr),
                                Np,
                                Ngrid,
                                L,
                                n_threads,
                                static_cast<double*>(delta.mutable_data()));
    return delta;
}

py::array_t<double> interpolate_forces_CIC(
    const py::array_t<double, py::array::c_style | py::array::forcecast>& Fx,
    const py::array_t<double, py::array::c_style | py::array::forcecast>& Fy,
    const py::array_t<double, py::array::c_style | py::array::forcecast>& Fz,
    const py::array_t<double, py::array::c_style | py::array::forcecast>& pod,
    int grid_n,
    double box_size,
    int n_threads = 8)
{
    auto buf_Fx = Fx.request();
    auto buf_Fy = Fy.request();
    auto buf_Fz = Fz.request();
    auto buf_pos = pod.request();

    const ssize_t N = buf_pos.shape[0];
    const double dx = box_size / grid_n;

    py::array_t<double> result({N, static_cast<ssize_t>(3)});
    interpolate_forces_CIC_raw(static_cast<const double*>(buf_Fx.ptr),
                           static_cast<const double*>(buf_Fy.ptr),
                           static_cast<const double*>(buf_Fz.ptr),
                           grid_n,
                           box_size,
                           dx,
                           static_cast<const double*>(buf_pos.ptr),
                           N,
                           n_threads,
                           static_cast<double*>(result.mutable_data()));
    return result;
}

// 1D TSC kernel
inline double W_tsc(double r) {
    double ar = std::abs(r);
    if (ar < 0.5)         return 0.75 - ar*ar;
    else if (ar < 1.5)    { double t = 1.5 - ar; return 0.5 * t * t; }
    else                  return 0.0;
}

// Build 1D TSC stencil centered on nearest grid index
inline void tsc_stencil_1d(double xg, int Ngrid, int idx[3], double w[3]) {
    // wrap xg to [0, Ngrid)
    double x = xg - std::floor(xg / Ngrid) * Ngrid;
    // nearest grid index
    int jc = static_cast<int>(std::floor(x + 0.5));
    if (jc >= Ngrid) jc -= Ngrid;

    int j0 = jc - 1, j1 = jc, j2 = jc + 1;
    // periodic wrap
    idx[0] = (j0 % Ngrid + Ngrid) % Ngrid;
    idx[1] = (j1 % Ngrid + Ngrid) % Ngrid;
    idx[2] = (j2 % Ngrid + Ngrid) % Ngrid;

    // distances to those indices (in grid units), minimum image
    auto dist = [&](double j)->double {
        double d = x - j;
        // map to [-Ngrid/2, Ngrid/2] to be safe near boundaries
        d -= std::round(d / Ngrid) * Ngrid;
        return d;
    };
    w[0] = W_tsc(dist((double)j0));
    w[1] = W_tsc(dist((double)j1));
    w[2] = W_tsc(dist((double)j2));
}

py::array_t<double> interpolate_forces_TSC(
    const py::array_t<double, py::array::c_style | py::array::forcecast>& Fx,
    const py::array_t<double, py::array::c_style | py::array::forcecast>& Fy,
    const py::array_t<double, py::array::c_style | py::array::forcecast>& Fz,
    const py::array_t<double, py::array::c_style | py::array::forcecast>& pod,
    int grid_n,
    double box_size,
    int n_threads = 8)
{
    auto buf_Fx = Fx.request();
    auto buf_Fy = Fy.request();
    auto buf_Fz = Fz.request();
    auto buf_pos = pod.request();

    const ssize_t N = buf_pos.shape[0];
    const double dx = box_size / grid_n;

    py::array_t<double> result({N, static_cast<ssize_t>(3)});
    interpolate_forces_TSC_raw(static_cast<const double*>(buf_Fx.ptr),
                               static_cast<const double*>(buf_Fy.ptr),
                               static_cast<const double*>(buf_Fz.ptr),
                               grid_n,
                               box_size,
                               dx,
                               static_cast<const double*>(buf_pos.ptr),
                               N,
                               n_threads,
                               static_cast<double*>(result.mutable_data()));
    return result;
}

py::array_t<double> interpolate_potential_CIC(
    const py::array_t<double, py::array::c_style | py::array::forcecast>& phi_in,
    const py::array_t<double, py::array::c_style | py::array::forcecast>& pod,
    int grid_n,
    double box_size,
    int n_threads = 8)
{
    auto buf_phi = phi_in.request();
    auto buf_pos = pod.request();

    const ssize_t N = buf_pos.shape[0];
    const double dx = box_size / grid_n;

    py::array_t<double> result(N);
    interpolate_potential_CIC_raw(static_cast<const double*>(buf_phi.ptr),
                              grid_n,
                              box_size,
                              dx,
                              static_cast<const double*>(buf_pos.ptr),
                              N,
                              n_threads,
                              static_cast<double*>(result.mutable_data()));
    return result;
}


namespace {

void compute_delta_field_TSC_raw(const double* pos,
                                 ssize_t Np,
                                 int Ngrid,
                                 double L,
                                 int n_threads,
                                 double* delta_out)
{
    const double dx = L / Ngrid;
    const ssize_t Ngrid3 = static_cast<ssize_t>(Ngrid) * Ngrid * Ngrid;

    std::vector<double> rho(static_cast<size_t>(Ngrid3), 0.0);

    #pragma omp parallel for num_threads(n_threads) schedule(static)
    for (ssize_t p = 0; p < Np; ++p) {
        const double xg = pos[3*p + 0] / dx;
        const double yg = pos[3*p + 1] / dx;
        const double zg = pos[3*p + 2] / dx;

        int ix[3], iy[3], iz[3];
        double wx[3], wy[3], wz[3];
        tsc_stencil_1d(xg, Ngrid, ix, wx);
        tsc_stencil_1d(yg, Ngrid, iy, wy);
        tsc_stencil_1d(zg, Ngrid, iz, wz);

        for (int a = 0; a < 3; ++a) {
            const double wxa = wx[a];
            if (wxa == 0.0) continue;
            const int i = ix[a];

            for (int b = 0; b < 3; ++b) {
                const double wyb = wy[b];
                if (wyb == 0.0) continue;
                const int j = iy[b];
                const double wxy = wxa * wyb;

                for (int c = 0; c < 3; ++c) {
                    const double wzc = wz[c];
                    if (wzc == 0.0) continue;
                    const int k = iz[c];
                    const ssize_t idx = (static_cast<ssize_t>(i) * Ngrid + j) * Ngrid + k;
                    #pragma omp atomic
                    rho[static_cast<size_t>(idx)] += wxy * wzc;
                }
            }
        }
    }

    double total = 0.0;
    #pragma omp parallel for reduction(+:total) num_threads(n_threads)
    for (ssize_t idx = 0; idx < Ngrid3; ++idx)
        total += rho[static_cast<size_t>(idx)];

    const double mean = total / static_cast<double>(Ngrid3);

    #pragma omp parallel for num_threads(n_threads)
    for (ssize_t idx = 0; idx < Ngrid3; ++idx) {
        const double value = rho[static_cast<size_t>(idx)];
        delta_out[idx] = (value - mean) / mean;
    }
}

void compute_delta_field_CIC_raw(const double* pos,
                                 ssize_t Np,
                                 int Ngrid,
                                 double L,
                                 int n_threads,
                                 double* delta_out)
{
    const double dx = L / Ngrid;
    const ssize_t Ngrid3 = static_cast<ssize_t>(Ngrid) * Ngrid * Ngrid;

    std::fill(delta_out, delta_out + Ngrid3, 0.0);

    #pragma omp parallel for num_threads(n_threads) schedule(static)
    for (ssize_t p = 0; p < Np; ++p) {
        double x = std::fmod(pos[3*p + 0], L) / dx;
        double y = std::fmod(pos[3*p + 1], L) / dx;
        double z = std::fmod(pos[3*p + 2], L) / dx;

        if (x < 0) x += Ngrid;
        if (y < 0) y += Ngrid;
        if (z < 0) z += Ngrid;

        int i0 = static_cast<int>(std::floor(x)) % Ngrid;
        int j0 = static_cast<int>(std::floor(y)) % Ngrid;
        int k0 = static_cast<int>(std::floor(z)) % Ngrid;

        double fx = x - i0;
        double fy = y - j0;
        double fz = z - k0;

        for (int dx_i = 0; dx_i <= 1; ++dx_i) {
            for (int dy_i = 0; dy_i <= 1; ++dy_i) {
                for (int dz_i = 0; dz_i <= 1; ++dz_i) {
                    double wx = dx_i ? fx : 1 - fx;
                    double wy = dy_i ? fy : 1 - fy;
                    double wz = dz_i ? fz : 1 - fz;
                    double w = wx * wy * wz;

                    int ii = (i0 + dx_i) % Ngrid;
                    int jj = (j0 + dy_i) % Ngrid;
                    int kk = (k0 + dz_i) % Ngrid;
                    const ssize_t idx = (static_cast<ssize_t>(ii) * Ngrid + jj) * Ngrid + kk;
                    #pragma omp atomic
                    delta_out[idx] += w;
                }
            }
        }
    }

    double total = 0.0;
    #pragma omp parallel for reduction(+:total) num_threads(n_threads)
    for (ssize_t idx = 0; idx < Ngrid3; ++idx)
        total += delta_out[idx];

    const double mean = total / static_cast<double>(Ngrid3);
    #pragma omp parallel for num_threads(n_threads)
    for (ssize_t idx = 0; idx < Ngrid3; ++idx) {
        const double value = delta_out[idx];
        delta_out[idx] = (value - mean) / mean;
    }
}

inline int wrap_index(int idx, int N) {
    if (idx >= N) idx -= N;
    else if (idx < 0) idx += N;
    return idx;
}

void gather_scalar_CIC_raw(const double* grid,
                           int Ngrid,
                           double Lbox,
                           double dx,
                           const double* pos,
                           ssize_t N,
                           int n_threads,
                           ptrdiff_t out_stride,
                           double* out)
{
    #pragma omp parallel for num_threads(n_threads) schedule(static)
    for (ssize_t p = 0; p < N; ++p) {
        double qx = pos[3*p + 0] / dx;
        double qy = pos[3*p + 1] / dx;
        double qz = pos[3*p + 2] / dx;

        qx -= std::floor(qx / Ngrid) * Ngrid;
        qy -= std::floor(qy / Ngrid) * Ngrid;
        qz -= std::floor(qz / Ngrid) * Ngrid;

        int i0 = static_cast<int>(std::floor(qx));
        int j0 = static_cast<int>(std::floor(qy));
        int k0 = static_cast<int>(std::floor(qz));

        double fx = qx - i0;
        double fy = qy - j0;
        double fz = qz - k0;

        double sum = 0.0;
        for (int dx_i = 0; dx_i <= 1; ++dx_i) {
            int ii = wrap_index(i0 + dx_i, Ngrid);
            double wx = dx_i ? fx : 1.0 - fx;

            for (int dy_i = 0; dy_i <= 1; ++dy_i) {
                int jj = wrap_index(j0 + dy_i, Ngrid);
                double wy = dy_i ? fy : 1.0 - fy;

                for (int dz_i = 0; dz_i <= 1; ++dz_i) {
                    int kk = wrap_index(k0 + dz_i, Ngrid);
                    double wz = dz_i ? fz : 1.0 - fz;
                    double w = wx * wy * wz;
                    sum += w * grid[(static_cast<ssize_t>(ii) * Ngrid + jj) * Ngrid + kk];
                }
            }
        }

        out[p * out_stride] = sum;
    }
}

void tsc_gather_scalar_TSC_raw(const double* grid,
                           int Ngrid,
                           double Lbox,
                           double dx,
                           const double* pos,
                           ssize_t N,
                           int n_threads,
                           ptrdiff_t out_stride,
                           double* out)
{
    #pragma omp parallel for num_threads(n_threads) schedule(static)
    for (ssize_t p = 0; p < N; ++p) {
        double xg = pos[3*p + 0] / dx;
        double yg = pos[3*p + 1] / dx;
        double zg = pos[3*p + 2] / dx;

        int ix[3], iy[3], iz[3];
        double wx[3], wy[3], wz[3];
        tsc_stencil_1d(xg, Ngrid, ix, wx);
        tsc_stencil_1d(yg, Ngrid, iy, wy);
        tsc_stencil_1d(zg, Ngrid, iz, wz);

        double sum = 0.0;
        for (int a = 0; a < 3; ++a) {
            double wxa = wx[a];
            if (wxa == 0.0) continue;
            int i = ix[a];
            for (int b = 0; b < 3; ++b) {
                double wyb = wy[b];
                if (wyb == 0.0) continue;
                int j = iy[b];
                double wxy = wxa * wyb;
                for (int c = 0; c < 3; ++c) {
                    double wzc = wz[c];
                    if (wzc == 0.0) continue;
                    int k = iz[c];
                    sum += wxy * wzc * grid[(static_cast<ssize_t>(i) * Ngrid + j) * Ngrid + k];
                }
            }
        }

        out[p * out_stride] = sum;
    }
}

void interpolate_forces_CIC_raw(const double* Fx,
                            const double* Fy,
                            const double* Fz,
                            int Ngrid,
                            double Lbox,
                            double dx,
                            const double* pos,
                            ssize_t N,
                            int n_threads,
                            double* out)
{
    #pragma omp parallel for num_threads(n_threads) schedule(static)
    for (ssize_t p = 0; p < N; ++p) {
        double qx = pos[3*p + 0] / dx;
        double qy = pos[3*p + 1] / dx;
        double qz = pos[3*p + 2] / dx;

        qx -= std::floor(qx / Ngrid) * Ngrid;
        qy -= std::floor(qy / Ngrid) * Ngrid;
        qz -= std::floor(qz / Ngrid) * Ngrid;

        int i0 = static_cast<int>(std::floor(qx));
        int j0 = static_cast<int>(std::floor(qy));
        int k0 = static_cast<int>(std::floor(qz));

        double fx = qx - i0;
        double fy = qy - j0;
        double fz = qz - k0;

        double sumx = 0.0, sumy = 0.0, sumz = 0.0;
        for (int dx_i = 0; dx_i <= 1; ++dx_i) {
            int ii = wrap_index(i0 + dx_i, Ngrid);
            double wx = dx_i ? fx : 1.0 - fx;

            for (int dy_i = 0; dy_i <= 1; ++dy_i) {
                int jj = wrap_index(j0 + dy_i, Ngrid);
                double wy = dy_i ? fy : 1.0 - fy;

                for (int dz_i = 0; dz_i <= 1; ++dz_i) {
                    int kk = wrap_index(k0 + dz_i, Ngrid);
                    double wz = dz_i ? fz : 1.0 - fz;
                    double w = wx * wy * wz;
                    const ssize_t idx = (static_cast<ssize_t>(ii) * Ngrid + jj) * Ngrid + kk;
                    sumx += w * Fx[idx];
                    sumy += w * Fy[idx];
                    sumz += w * Fz[idx];
                }
            }
        }

        out[3*p + 0] = sumx;
        out[3*p + 1] = sumy;
        out[3*p + 2] = sumz;
    }
}

void interpolate_forces_TSC_raw(const double* Fx,
                                const double* Fy,
                                const double* Fz,
                                int Ngrid,
                                double Lbox,
                                double dx,
                                const double* pos,
                                ssize_t N,
                                int n_threads,
                                double* out)
{
    #pragma omp parallel for num_threads(n_threads) schedule(static)
    for (ssize_t p = 0; p < N; ++p) {
        double xg = pos[3*p + 0] / dx;
        double yg = pos[3*p + 1] / dx;
        double zg = pos[3*p + 2] / dx;

        int ix[3], iy[3], iz[3];
        double wx[3], wy[3], wz[3];
        tsc_stencil_1d(xg, Ngrid, ix, wx);
        tsc_stencil_1d(yg, Ngrid, iy, wy);
        tsc_stencil_1d(zg, Ngrid, iz, wz);

        double sumx = 0.0, sumy = 0.0, sumz = 0.0;
        for (int a = 0; a < 3; ++a) {
            double wxa = wx[a];
            if (wxa == 0.0) continue;
            int i = ix[a];
            for (int b = 0; b < 3; ++b) {
                double wyb = wy[b];
                if (wyb == 0.0) continue;
                int j = iy[b];
                double wxy = wxa * wyb;
                for (int c = 0; c < 3; ++c) {
                    double wzc = wz[c];
                    if (wzc == 0.0) continue;
                    int k = iz[c];
                    const ssize_t idx = (static_cast<ssize_t>(i) * Ngrid + j) * Ngrid + k;
                    double w = wxy * wzc;
                    sumx += w * Fx[idx];
                    sumy += w * Fy[idx];
                    sumz += w * Fz[idx];
                }
            }
        }

        out[3*p + 0] = sumx;
        out[3*p + 1] = sumy;
        out[3*p + 2] = sumz;
    }
}

void interpolate_potential_CIC_raw(const double* grid,
                               int Ngrid,
                               double Lbox,
                               double dx,
                               const double* pos,
                               ssize_t N,
                               int n_threads,
                               double* out)
{
    gather_scalar_CIC_raw(grid, Ngrid, Lbox, dx, pos, N, n_threads, 1, out);
}

} // namespace

py::array_t<double> compute_delta_field_TSC(const py::array_t<double, py::array::c_style | py::array::forcecast>& positions,
                                            int Ngrid, double L, int n_threads)
{
    py::buffer_info buf = positions.request();
    if (buf.ndim != 2 || buf.shape[1] != 3)
        throw std::runtime_error("positions must be of shape (N, 3)");

    const ssize_t Np = buf.shape[0];
    const double* pos_ptr = static_cast<const double*>(buf.ptr);

    py::array_t<double> delta({Ngrid, Ngrid, Ngrid});
    compute_delta_field_TSC_raw(pos_ptr,
                                Np,
                                Ngrid,
                                L,
                                n_threads,
                                static_cast<double*>(delta.mutable_data()));
    return delta;
}

// standard sinc(x) = sin(x)/x
inline double sinc_std(double x) {
    if (std::abs(x) < 1e-15) return 1.0;
    return std::sin(x) / x;
}

struct PMEngine {
    int Ngrid;
    double L, dx, two_pi;
    bool use_TSC;
    int n_threads;

    // persistent FFTW
    std::vector<cplx> buf_spatial_c;   // in-place complex buffer for fwd/back
    std::vector<cplx> work_k;          // scratch k-space
    std::vector<double> kfreq;         // k grid
    // reusable buffers to avoid per-step allocations
    std::vector<double> delta_buf;
    std::vector<cplx> phi_k_buf;
    std::vector<cplx> kx_phi_buf;
    std::vector<cplx> ky_phi_buf;
    std::vector<cplx> kz_phi_buf;
    std::vector<double> Fx_buf;
    std::vector<double> Fy_buf;
    std::vector<double> Fz_buf;
    std::vector<double> phi_grid_buf;
    std::vector<double> accel_buf;
    std::vector<double> phi_part_buf;
    std::vector<double> delta_part_buf;
    fftw_plan plan_fwd, plan_back;

    PMEngine(int Ngrid_, double L_, bool use_TSC_, int n_threads_)
      : Ngrid(Ngrid_), L(L_), dx(L_/Ngrid_), two_pi(2.0*M_PI),
        use_TSC(use_TSC_), n_threads(n_threads_),
        buf_spatial_c(Ngrid_*Ngrid_*Ngrid_),
        work_k(Ngrid_*Ngrid_*Ngrid_),
        kfreq(Ngrid_),
        delta_buf(Ngrid_*Ngrid_*Ngrid_),
        phi_k_buf(Ngrid_*Ngrid_*Ngrid_),
        kx_phi_buf(Ngrid_*Ngrid_*Ngrid_),
        ky_phi_buf(Ngrid_*Ngrid_*Ngrid_),
        kz_phi_buf(Ngrid_*Ngrid_*Ngrid_),
        Fx_buf(Ngrid_*Ngrid_*Ngrid_),
        Fy_buf(Ngrid_*Ngrid_*Ngrid_),
        Fz_buf(Ngrid_*Ngrid_*Ngrid_),
        phi_grid_buf(Ngrid_*Ngrid_*Ngrid_)
    {
        // k grid
        for(int n=0;n<Ngrid;n++){
            int m = (n<=Ngrid/2)? n : n - Ngrid;
            kfreq[n] = two_pi * double(m) / L;
        }
        // FFTW threaded plans (reused)
        fftw_init_threads();
        fftw_plan_with_nthreads(n_threads);
        fftw_complex* in  = reinterpret_cast<fftw_complex*>(buf_spatial_c.data());
        fftw_complex* out = reinterpret_cast<fftw_complex*>(work_k.data());
        plan_fwd  = fftw_plan_dft_3d(Ngrid,Ngrid,Ngrid, in, out, FFTW_FORWARD,  FFTW_MEASURE);
        plan_back = fftw_plan_dft_3d(Ngrid,Ngrid,Ngrid, in, out, FFTW_BACKWARD, FFTW_MEASURE);
    }

    ~PMEngine(){
        fftw_destroy_plan(plan_fwd);
        fftw_destroy_plan(plan_back);
        fftw_cleanup_threads();
    }

    py::tuple step(const py::array_t<double, py::array::c_style | py::array::forcecast>& pod,
                   double mass,
                   double a,
                   double G)
    {
        py::buffer_info pod_buf = pod.request();
        if (pod_buf.ndim != 2 || pod_buf.shape[1] != 3) {
            throw std::runtime_error("pod must have shape (N, 3)");
        }

        const ssize_t Np = pod_buf.shape[0];
        const double* pod_ptr = static_cast<const double*>(pod_buf.ptr);
        const size_t grid_size = static_cast<size_t>(Ngrid) * static_cast<size_t>(Ngrid) * static_cast<size_t>(Ngrid);
        const double dx_box = L / static_cast<double>(Ngrid);
        const double rho_bar = (static_cast<double>(Np) * mass) / (L * L * L);

        if (delta_buf.size() != grid_size) {
            delta_buf.resize(grid_size);
            phi_k_buf.resize(grid_size);
            kx_phi_buf.resize(grid_size);
            ky_phi_buf.resize(grid_size);
            kz_phi_buf.resize(grid_size);
            Fx_buf.resize(grid_size);
            Fy_buf.resize(grid_size);
            Fz_buf.resize(grid_size);
            phi_grid_buf.resize(grid_size);
        }
        accel_buf.resize(static_cast<size_t>(Np) * 3);
        phi_part_buf.resize(static_cast<size_t>(Np));
        delta_part_buf.resize(static_cast<size_t>(Np));
        double potential_sum = 0.0;

        {
            py::gil_scoped_release release;

            if (use_TSC) {
                compute_delta_field_TSC_raw(pod_ptr, Np, Ngrid, L, n_threads, delta_buf.data());
            } else {
                compute_delta_field_CIC_raw(pod_ptr, Np, Ngrid, L, n_threads, delta_buf.data());
            }

            #pragma omp parallel for if(n_threads>1) num_threads(n_threads) schedule(static)
            for (ssize_t idx = 0; idx < static_cast<ssize_t>(grid_size); ++idx) {
                buf_spatial_c[static_cast<size_t>(idx)] = cplx(delta_buf[static_cast<size_t>(idx)], 0.0);
            }

            fftw_execute(plan_fwd);

            const double Wfloor = 1e-6;
            #pragma omp parallel for collapse(2) if(n_threads>1) num_threads(n_threads) schedule(static)
            for (int i = 0; i < Ngrid; ++i) {
                for (int j = 0; j < Ngrid; ++j) {
                    for (int k = 0; k < Ngrid; ++k) {
                        size_t lin = (static_cast<size_t>(i) * Ngrid + static_cast<size_t>(j)) * Ngrid + static_cast<size_t>(k);
                        double kx = kfreq[i];
                        double ky = kfreq[j];
                        double kz = kfreq[k];
                        double Wx = std::pow(sinc_std(0.5 * kx * dx), use_TSC ? 3 : 2);
                        double Wy = std::pow(sinc_std(0.5 * ky * dx), use_TSC ? 3 : 2);
                        double Wz = std::pow(sinc_std(0.5 * kz * dx), use_TSC ? 3 : 2);
                        double W = Wx * Wy * Wz;
                        if (W < Wfloor) W = Wfloor;

                        cplx dk = work_k[lin] / W;
                        double k2 = kx * kx + ky * ky + kz * kz;

                        cplx phik = (k2 > 0.0) ? (-4.0 * M_PI * G * rho_bar / a) * (dk / k2) : cplx(0.0, 0.0);
                        phi_k_buf[lin] = phik;
                        kx_phi_buf[lin] = cplx(0.0, -1.0) * kx * phik;
                        ky_phi_buf[lin] = cplx(0.0, -1.0) * ky * phik;
                        kz_phi_buf[lin] = cplx(0.0, -1.0) * kz * phik;
                    }
                }
            }

            auto ifft3 = [&](const std::vector<cplx>& K, std::vector<double>& real_out) {
                fftw_complex* in = reinterpret_cast<fftw_complex*>(buf_spatial_c.data());
                fftw_complex* out = reinterpret_cast<fftw_complex*>(work_k.data());
                std::memcpy(in, K.data(), sizeof(cplx) * K.size());
                fftw_execute(plan_back);
                const double invN3 = 1.0 / static_cast<double>(grid_size);
                #pragma omp parallel for if(n_threads>1) num_threads(n_threads) schedule(static)
                for (ssize_t idx = 0; idx < static_cast<ssize_t>(grid_size); ++idx) {
                    real_out[static_cast<size_t>(idx)] = out[static_cast<size_t>(idx)][0] * invN3;
                }
            };

            ifft3(kx_phi_buf, Fx_buf);
            ifft3(ky_phi_buf, Fy_buf);
            ifft3(kz_phi_buf, Fz_buf);
            ifft3(phi_k_buf, phi_grid_buf);

            if (use_TSC) {
                interpolate_forces_TSC_raw(Fx_buf.data(),
                                            Fy_buf.data(),
                                            Fz_buf.data(),
                                            Ngrid,
                                            L,
                                            dx_box,
                                            pod_ptr,
                                            Np,
                                            n_threads,
                                            accel_buf.data());
            } else {
                interpolate_forces_CIC_raw(Fx_buf.data(),
                                       Fy_buf.data(),
                                       Fz_buf.data(),
                                       Ngrid,
                                       L,
                                       dx_box,
                                       pod_ptr,
                                       Np,
                                       n_threads,
                                       accel_buf.data());
            }

            interpolate_potential_CIC_raw(phi_grid_buf.data(),
                                       Ngrid,
                                       L,
                                       dx_box,
                                       pod_ptr,
                                       Np,
                                       n_threads,
                                       phi_part_buf.data());

            interpolate_potential_CIC_raw(delta_buf.data(),
                                       Ngrid,
                                       L,
                                       dx_box,
                                       pod_ptr,
                                       Np,
                                       n_threads,
                                       delta_part_buf.data());

            const double inv_a2 = 1.0 / (a * a);
            #pragma omp parallel for if(n_threads>1) num_threads(n_threads) schedule(static)
            for (ssize_t idx = 0; idx < Np; ++idx) {
                size_t base = static_cast<size_t>(idx) * 3;
                accel_buf[base + 0] *= inv_a2;
                accel_buf[base + 1] *= inv_a2;
                accel_buf[base + 2] *= inv_a2;
            }

            #pragma omp parallel for reduction(+:potential_sum) if(n_threads>1) num_threads(n_threads) schedule(static)
            for (ssize_t idx = 0; idx < Np; ++idx) {
                potential_sum += phi_part_buf[static_cast<size_t>(idx)];
            }
        }

        double PE = 0.5 * mass * potential_sum;

        py::ssize_t N = Np;
        py::array_t<double> acc_arr({N, static_cast<py::ssize_t>(3)});
        std::memcpy(acc_arr.mutable_data(), accel_buf.data(), accel_buf.size() * sizeof(double));

        py::array_t<double> delta_part_arr(N);
        std::memcpy(delta_part_arr.mutable_data(), delta_part_buf.data(), delta_part_buf.size() * sizeof(double));

        return py::make_tuple(acc_arr, delta_part_arr, py::float_(PE));
    }
};


using Array3D = py::array_t<double, py::array::c_style | py::array::forcecast>;
using Array1D = py::array_t<double, py::array::c_style | py::array::forcecast>;

// --- 1. CIC density deposition ---
std::pair<Array3D, double> cic_deposit_density(py::array_t<double> pos_Mpc, double Lbox_Mpc, int Ng) {
    auto pos = pos_Mpc.unchecked<2>();  // shape (N, 3)
    int N = pos.shape(0);
    double dx = Lbox_Mpc / Ng;

    //Array3D rho({Ng, Ng, Ng});
    auto rho = py::array_t<double>( {Ng, Ng, Ng} );
    std::memset(rho.mutable_data(), 0, sizeof(double) * Ng * Ng * Ng);
    auto r = rho.mutable_unchecked<3>();

    for (int p = 0; p < N; ++p) {
        double qx = std::fmod(pos(p,0), Lbox_Mpc) / dx;
        double qy = std::fmod(pos(p,1), Lbox_Mpc) / dx;
        double qz = std::fmod(pos(p,2), Lbox_Mpc) / dx;

        int i = int(std::floor(qx)) % Ng;
        int j = int(std::floor(qy)) % Ng;
        int k = int(std::floor(qz)) % Ng;

        double dxp = qx - i;
        double dyp = qy - j;
        double dzp = qz - k;

        for (int ox = 0; ox <= 1; ++ox)
        for (int oy = 0; oy <= 1; ++oy)
        for (int oz = 0; oz <= 1; ++oz) {
            double wx = (ox == 0) ? (1 - dxp) : dxp;
            double wy = (oy == 0) ? (1 - dyp) : dyp;
            double wz = (oz == 0) ? (1 - dzp) : dzp;
            double w = wx * wy * wz;

            int ii = (i + ox) % Ng;
            int jj = (j + oy) % Ng;
            int kk = (k + oz) % Ng;

            r(ii, jj, kk) += w;
        }
    }

    // Normalize by volume
    for (int i = 0; i < Ng; ++i)
    for (int j = 0; j < Ng; ++j)
    for (int k = 0; k < Ng; ++k)
        r(i, j, k) /= (dx * dx * dx);

    return std::make_pair(rho, dx);
}


// --- 2. CIC energy deposition ---
std::pair<Array3D, double> cic_deposit_energy(const py::array_t<double> pos_Mpc,
                                              const py::array_t<double> scalar,
                                              double Lbox_Mpc, int Ng) {
    auto pos = pos_Mpc.unchecked<2>();
    auto u   = scalar.unchecked<1>();
    int N = pos.shape(0);
    double dx = Lbox_Mpc / Ng;

    Array3D num({Ng, Ng, Ng});  // weighted energy sum
    Array3D den({Ng, Ng, Ng});  // weights
    std::memset(num.mutable_data(), 0, sizeof(double) * Ng * Ng * Ng);
    std::memset(den.mutable_data(), 0, sizeof(double) * Ng * Ng * Ng);
    auto n = num.mutable_unchecked<3>();
    auto d = den.mutable_unchecked<3>();

    for (int p = 0; p < N; ++p) {
        double qx = std::fmod(pos(p,0), Lbox_Mpc) / dx;
        double qy = std::fmod(pos(p,1), Lbox_Mpc) / dx;
        double qz = std::fmod(pos(p,2), Lbox_Mpc) / dx;

        int i = int(std::floor(qx)) % Ng;
        int j = int(std::floor(qy)) % Ng;
        int k = int(std::floor(qz)) % Ng;

        double dxp = qx - i;
        double dyp = qy - j;
        double dzp = qz - k;

        for (int ox = 0; ox <= 1; ++ox)
        for (int oy = 0; oy <= 1; ++oy)
        for (int oz = 0; oz <= 1; ++oz) {
            double wx = (ox == 0) ? (1 - dxp) : dxp;
            double wy = (oy == 0) ? (1 - dyp) : dyp;
            double wz = (oz == 0) ? (1 - dzp) : dzp;
            double w = wx * wy * wz;

            int ii = (i + ox) % Ng;
            int jj = (j + oy) % Ng;
            int kk = (k + oz) % Ng;

            n(ii, jj, kk) += w * u(p);
            d(ii, jj, kk) += w;
        }
    }

    // Normalize num / den where den > 0
    Array3D u_grid({Ng, Ng, Ng});
    auto out = u_grid.mutable_unchecked<3>();
    for (int i = 0; i < Ng; ++i)
    for (int j = 0; j < Ng; ++j)
    for (int k = 0; k < Ng; ++k)
        out(i, j, k) = (d(i, j, k) > 0) ? n(i, j, k) / d(i, j, k) : 0.0;

    return std::make_pair(u_grid, dx);
}


// --- 1b. TSC density deposition ---
std::pair<Array3D, double> tsc_deposit_density(py::array_t<double> pos_Mpc, double Lbox_Mpc, int Ng) {
    auto pos = pos_Mpc.unchecked<2>();
    int N = pos.shape(0);
    double dx = Lbox_Mpc / Ng;

    auto rho = py::array_t<double>({Ng, Ng, Ng});
    std::memset(rho.mutable_data(), 0, sizeof(double) * Ng * Ng * Ng);
    auto grid = rho.mutable_unchecked<3>();

    for (int p = 0; p < N; ++p) {
        double xg = pos(p, 0) / dx;
        double yg = pos(p, 1) / dx;
        double zg = pos(p, 2) / dx;

        int ix[3], iy[3], iz[3];
        double wx[3], wy[3], wz[3];
        tsc_stencil_1d(xg, Ng, ix, wx);
        tsc_stencil_1d(yg, Ng, iy, wy);
        tsc_stencil_1d(zg, Ng, iz, wz);

        for (int a = 0; a < 3; ++a) {
            double wxa = wx[a];
            if (wxa == 0.0) continue;
            int i = ix[a];
            for (int b = 0; b < 3; ++b) {
                double wyb = wy[b];
                if (wyb == 0.0) continue;
                int j = iy[b];
                double wxy = wxa * wyb;
                for (int c = 0; c < 3; ++c) {
                    double wzc = wz[c];
                    if (wzc == 0.0) continue;
                    int k = iz[c];
                    grid(i, j, k) += wxy * wzc;
                }
            }
        }
    }

    for (int i = 0; i < Ng; ++i)
    for (int j = 0; j < Ng; ++j)
    for (int k = 0; k < Ng; ++k)
        grid(i, j, k) /= (dx * dx * dx);

    return std::make_pair(rho, dx);
}


// --- 2b. TSC energy deposition ---
std::pair<Array3D, double> tsc_deposit_energy(const py::array_t<double> pos_Mpc,
                                              const py::array_t<double> scalar,
                                              double Lbox_Mpc, int Ng) {
    auto pos = pos_Mpc.unchecked<2>();
    auto u   = scalar.unchecked<1>();
    int N = pos.shape(0);
    double dx = Lbox_Mpc / Ng;

    Array3D num({Ng, Ng, Ng});
    Array3D den({Ng, Ng, Ng});
    std::memset(num.mutable_data(), 0, sizeof(double) * Ng * Ng * Ng);
    std::memset(den.mutable_data(), 0, sizeof(double) * Ng * Ng * Ng);
    auto n = num.mutable_unchecked<3>();
    auto d = den.mutable_unchecked<3>();

    for (int p = 0; p < N; ++p) {
        double xg = pos(p, 0) / dx;
        double yg = pos(p, 1) / dx;
        double zg = pos(p, 2) / dx;

        int ix[3], iy[3], iz[3];
        double wx[3], wy[3], wz[3];
        tsc_stencil_1d(xg, Ng, ix, wx);
        tsc_stencil_1d(yg, Ng, iy, wy);
        tsc_stencil_1d(zg, Ng, iz, wz);

        for (int a = 0; a < 3; ++a) {
            double wxa = wx[a];
            if (wxa == 0.0) continue;
            int i = ix[a];
            for (int b = 0; b < 3; ++b) {
                double wyb = wy[b];
                if (wyb == 0.0) continue;
                int j = iy[b];
                double wxy = wxa * wyb;
                for (int c = 0; c < 3; ++c) {
                    double wzc = wz[c];
                    if (wzc == 0.0) continue;
                    int k = iz[c];
                    double w = wxy * wzc;
                    n(i, j, k) += w * u(p);
                    d(i, j, k) += w;
                }
            }
        }
    }

    Array3D u_grid({Ng, Ng, Ng});
    auto out = u_grid.mutable_unchecked<3>();
    for (int i = 0; i < Ng; ++i)
    for (int j = 0; j < Ng; ++j)
    for (int k = 0; k < Ng; ++k)
        out(i, j, k) = (d(i, j, k) > 0.0) ? n(i, j, k) / d(i, j, k) : 0.0;

    return std::make_pair(u_grid, dx);
}


// --- 1c. Wrapper density deposition ---
std::pair<Array3D, double> deposit_density(py::array_t<double> pos_Mpc,
                                           double Lbox_Mpc,
                                           int Ng,
                                           bool use_TSC) {
    return use_TSC ? tsc_deposit_density(std::move(pos_Mpc), Lbox_Mpc, Ng)
                   : cic_deposit_density(std::move(pos_Mpc), Lbox_Mpc, Ng);
}


// --- 2c. Wrapper energy deposition ---
std::pair<Array3D, double> deposit_energy(const py::array_t<double> pos_Mpc,
                                          const py::array_t<double> scalar,
                                          double Lbox_Mpc,
                                          int Ng,
                                          bool use_TSC) {
    return use_TSC ? tsc_deposit_energy(pos_Mpc, scalar, Lbox_Mpc, Ng)
                   : cic_deposit_energy(pos_Mpc, scalar, Lbox_Mpc, Ng);
}


// --- 3. CIC gather (trilinear interpolation) ---
py::array_t<double> cic_gather(const py::array_t<double, py::array::c_style | py::array::forcecast>& grid,
                               const py::array_t<double, py::array::c_style | py::array::forcecast>& pos_Mpc,
                               double Lbox_Mpc, double dx, int n_threads = 8) {
    auto buf_grid = grid.request();
    auto buf_pos = pos_Mpc.request();
    const int Ng = buf_grid.shape[0];
    const ssize_t N = buf_pos.shape[0];

    py::array_t<double> result(N);
    gather_scalar_CIC_raw(static_cast<const double*>(buf_grid.ptr),
                          Ng,
                          Lbox_Mpc,
                          dx,
                          static_cast<const double*>(buf_pos.ptr),
                          N,
                          n_threads,
                          1,
                          static_cast<double*>(result.mutable_data()));
    return result;
}


py::array_t<double> compute_delta_field(const py::array_t<double, py::array::c_style | py::array::forcecast>& positions,
                                        int Ngrid,
                                        double L,
                                        bool use_TSC,
                                        int n_threads = 8) {
    return use_TSC ? compute_delta_field_TSC(positions, Ngrid, L, n_threads)
                   : compute_delta_field_CIC(positions, Ngrid, L, n_threads);
}


py::array_t<double> gather(const py::array_t<double, py::array::c_style | py::array::forcecast>& grid,
                           const py::array_t<double, py::array::c_style | py::array::forcecast>& pos_Mpc,
                           double Lbox_Mpc,
                           double dx,
                           bool use_TSC,
                           int n_threads = 8) {
    auto buf_grid = grid.request();
    auto buf_pos = pos_Mpc.request();
    const int Ng = buf_grid.shape[0];
    const ssize_t N = buf_pos.shape[0];

    py::array_t<double> result(N);
    if (use_TSC) {
        tsc_gather_scalar_TSC_raw(static_cast<const double*>(buf_grid.ptr),
                                  Ng,
                                  Lbox_Mpc,
                                  dx,
                                  static_cast<const double*>(buf_pos.ptr),
                                  N,
                                  n_threads,
                                  1,
                                  static_cast<double*>(result.mutable_data()));
    } else {
        gather_scalar_CIC_raw(static_cast<const double*>(buf_grid.ptr),
                              Ng,
                              Lbox_Mpc,
                              dx,
                              static_cast<const double*>(buf_pos.ptr),
                              N,
                              n_threads,
                              1,
                              static_cast<double*>(result.mutable_data()));
    }
    return result;
}


namespace {

inline int wrap(int idx, int Ng) {
    if (idx >= Ng) idx -= Ng;
    if (idx < 0) idx += Ng;
    return idx;
}

Vec3Grid cic_deposit_vec_equal_mass_impl(const py::array_t<double> pos_Mpc,
                                         const py::array_t<double> vec,
                                         double Lbox_Mpc, int Ng) {
    auto pos = pos_Mpc.unchecked<2>();
    auto vel = vec.unchecked<2>();
    const int Np = pos.shape(0);
    const double dx = Lbox_Mpc / Ng;

    py::array_t<double> num_x({Ng, Ng, Ng});
    py::array_t<double> num_y({Ng, Ng, Ng});
    py::array_t<double> num_z({Ng, Ng, Ng});
    py::array_t<double> weight({Ng, Ng, Ng});
    std::memset(num_x.mutable_data(), 0, sizeof(double) * Ng * Ng * Ng);
    std::memset(num_y.mutable_data(), 0, sizeof(double) * Ng * Ng * Ng);
    std::memset(num_z.mutable_data(), 0, sizeof(double) * Ng * Ng * Ng);
    std::memset(weight.mutable_data(), 0, sizeof(double) * Ng * Ng * Ng);

    auto nx = num_x.mutable_unchecked<3>();
    auto ny = num_y.mutable_unchecked<3>();
    auto nz = num_z.mutable_unchecked<3>();
    auto w  = weight.mutable_unchecked<3>();

    for (int p = 0; p < Np; ++p) {
        double qx = std::fmod(pos(p, 0), Lbox_Mpc) / dx;
        double qy = std::fmod(pos(p, 1), Lbox_Mpc) / dx;
        double qz = std::fmod(pos(p, 2), Lbox_Mpc) / dx;

        int i = static_cast<int>(std::floor(qx)) % Ng;
        int j = static_cast<int>(std::floor(qy)) % Ng;
        int k = static_cast<int>(std::floor(qz)) % Ng;

        double dxp = qx - i;
        double dyp = qy - j;
        double dzp = qz - k;

        for (int ox = 0; ox <= 1; ++ox)
        for (int oy = 0; oy <= 1; ++oy)
        for (int oz = 0; oz <= 1; ++oz) {
            double wx = (ox == 0) ? (1.0 - dxp) : dxp;
            double wy = (oy == 0) ? (1.0 - dyp) : dyp;
            double wz = (oz == 0) ? (1.0 - dzp) : dzp;
            double weight_val = wx * wy * wz;

            int ii = wrap(i + ox, Ng);
            int jj = wrap(j + oy, Ng);
            int kk = wrap(k + oz, Ng);

            nx(ii, jj, kk) += weight_val * vel(p, 0);
            ny(ii, jj, kk) += weight_val * vel(p, 1);
            nz(ii, jj, kk) += weight_val * vel(p, 2);
            w(ii, jj, kk)  += weight_val;
        }
    }

    auto finalize = [&](py::array_t<double>& num){
        auto arr = num.mutable_unchecked<3>();
        #pragma omp parallel for collapse(3) schedule(static)
        for (int i = 0; i < Ng; ++i)
        for (int j = 0; j < Ng; ++j)
        for (int k = 0; k < Ng; ++k) {
            double denom = w(i, j, k);
            arr(i, j, k) = denom > 0.0 ? arr(i, j, k) / denom : 0.0;
        }
    };

    finalize(num_x);
    finalize(num_y);
    finalize(num_z);

    return {std::move(num_x), std::move(num_y), std::move(num_z)};
}

std::array<py::array_t<double>, 3> gradient_central_impl(const py::array_t<double> grid, double dx) {
    auto g = grid.unchecked<3>();
    const int Ng = g.shape(0);
    const double inv_2dx = 1.0 / (2.0 * dx);

    py::array_t<double> gx({Ng, Ng, Ng});
    py::array_t<double> gy({Ng, Ng, Ng});
    py::array_t<double> gz({Ng, Ng, Ng});
    auto out_x = gx.mutable_unchecked<3>();
    auto out_y = gy.mutable_unchecked<3>();
    auto out_z = gz.mutable_unchecked<3>();

    #pragma omp parallel for collapse(3)
    for (int i = 0; i < Ng; ++i) {
        for (int j = 0; j < Ng; ++j) {
            for (int k = 0; k < Ng; ++k) {
                int ip = wrap(i + 1, Ng);
                int im = wrap(i - 1, Ng);
                int jp = wrap(j + 1, Ng);
                int jm = wrap(j - 1, Ng);
                int kp = wrap(k + 1, Ng);
                int km = wrap(k - 1, Ng);

                out_x(i, j, k) = (g(ip, j, k) - g(im, j, k)) * inv_2dx;
                out_y(i, j, k) = (g(i, jp, k) - g(i, jm, k)) * inv_2dx;
                out_z(i, j, k) = (g(i, j, kp) - g(i, j, km)) * inv_2dx;
            }
        }
    }

    return {std::move(gx), std::move(gy), std::move(gz)};
}

py::array_t<double> divergence_central_impl(const py::array_t<double> vx,
                                            const py::array_t<double> vy,
                                            const py::array_t<double> vz,
                                            double dx) {
    auto vxv = vx.unchecked<3>();
    auto vyv = vy.unchecked<3>();
    auto vzv = vz.unchecked<3>();
    const int Ng = vxv.shape(0);
    const double inv_2dx = 1.0 / (2.0 * dx);

    py::array_t<double> div({Ng, Ng, Ng});
    auto out = div.mutable_unchecked<3>();

    #pragma omp parallel for collapse(3)
    for (int i = 0; i < Ng; ++i) {
        for (int j = 0; j < Ng; ++j) {
            for (int k = 0; k < Ng; ++k) {
                int ip = wrap(i + 1, Ng);
                int im = wrap(i - 1, Ng);
                int jp = wrap(j + 1, Ng);
                int jm = wrap(j - 1, Ng);
                int kp = wrap(k + 1, Ng);
                int km = wrap(k - 1, Ng);

                double dvx = (vxv(ip, j, k) - vxv(im, j, k)) * inv_2dx;
                double dvy = (vyv(i, jp, k) - vyv(i, jm, k)) * inv_2dx;
                double dvz = (vzv(i, j, kp) - vzv(i, j, km)) * inv_2dx;
                out(i, j, k) = dvx + dvy + dvz;
            }
        }
    }

    return div;
}

Vec3Grid pressure_acceleration_grid_impl(const py::array_t<double> rho_b_com_grid,
                                         const py::array_t<double> P_phys,
                                         double a, double dx) {
    auto grads = gradient_central_impl(P_phys, dx);
    auto rho = rho_b_com_grid.unchecked<3>();
    const int Ng = rho.shape(0);

    py::array_t<double> ax({Ng, Ng, Ng});
    py::array_t<double> ay({Ng, Ng, Ng});
    py::array_t<double> az({Ng, Ng, Ng});
    auto gx = grads[0].unchecked<3>();
    auto gy = grads[1].unchecked<3>();
    auto gz = grads[2].unchecked<3>();
    auto ax_out = ax.mutable_unchecked<3>();
    auto ay_out = ay.mutable_unchecked<3>();
    auto az_out = az.mutable_unchecked<3>();

    #pragma omp parallel for collapse(3)
    for (int i = 0; i < Ng; ++i) {
        for (int j = 0; j < Ng; ++j) {
            for (int k = 0; k < Ng; ++k) {
                double denom = std::max(rho(i, j, k), 1e-40);
                ax_out(i, j, k) = -(a / denom) * gx(i, j, k);
                ay_out(i, j, k) = -(a / denom) * gy(i, j, k);
                az_out(i, j, k) = -(a / denom) * gz(i, j, k);
            }
        }
    }

    return {std::move(ax), std::move(ay), std::move(az)};
}

py::array_t<double> gather_vec_impl(const py::array_t<double> ax,
                                    const py::array_t<double> ay,
                                    const py::array_t<double> az,
                                    const py::array_t<double> pos_Mpc,
                                    double Lbox_Mpc, double dx, int n_threads) {
    py::array_t<double> gx = cic_gather(ax, pos_Mpc, Lbox_Mpc, dx, n_threads);
    py::array_t<double> gy = cic_gather(ay, pos_Mpc, Lbox_Mpc, dx, n_threads);
    py::array_t<double> gz = cic_gather(az, pos_Mpc, Lbox_Mpc, dx, n_threads);

    auto gxi = gx.unchecked<1>();
    auto gyi = gy.unchecked<1>();
    auto gzi = gz.unchecked<1>();
    const ssize_t N = gx.shape(0);

    py::array_t<double> result({N, static_cast<ssize_t>(3)});
    auto out = result.mutable_unchecked<2>();

    #pragma omp parallel for
    for (ssize_t i = 0; i < N; ++i) {
        out(i, 0) = gxi(i);
        out(i, 1) = gyi(i);
        out(i, 2) = gzi(i);
    }

    return result;
}

py::array_t<double> curl_mag_cgs_impl(const py::array_t<double> vx,
                                      const py::array_t<double> vy,
                                      const py::array_t<double> vz,
                                      double dx) {
    auto vxv = vx.unchecked<3>();
    auto vyv = vy.unchecked<3>();
    auto vzv = vz.unchecked<3>();
    const int Ng = vxv.shape(0);
    const double inv_2dx = 1.0 / (2.0 * dx);

    py::array_t<double> curl_mag({Ng, Ng, Ng});
    auto out = curl_mag.mutable_unchecked<3>();

    #pragma omp parallel for collapse(3)
    for (int i = 0; i < Ng; ++i) {
        for (int j = 0; j < Ng; ++j) {
            for (int k = 0; k < Ng; ++k) {
                int ip = wrap(i + 1, Ng);
                int im = wrap(i - 1, Ng);
                int jp = wrap(j + 1, Ng);
                int jm = wrap(j - 1, Ng);
                int kp = wrap(k + 1, Ng);
                int km = wrap(k - 1, Ng);

                double dvz_dy = (vzv(i, jp, k) - vzv(i, jm, k)) * inv_2dx;
                double dvy_dz = (vyv(i, j, kp) - vyv(i, j, km)) * inv_2dx;
                double dvx_dz = (vxv(i, j, kp) - vxv(i, j, km)) * inv_2dx;
                double dvz_dx = (vzv(ip, j, k) - vzv(im, j, k)) * inv_2dx;
                double dvy_dx = (vyv(ip, j, k) - vyv(im, j, k)) * inv_2dx;
                double dvx_dy = (vxv(i, jp, k) - vxv(i, jm, k)) * inv_2dx;

                double wx = dvz_dy - dvy_dz;
                double wy = dvx_dz - dvz_dx;
                double wz = dvy_dx - dvx_dy;
                out(i, j, k) = std::sqrt(wx*wx + wy*wy + wz*wz);
            }
        }
    }

    return curl_mag;
}

py::array_t<double> strain_tracefree_norm2_cgs_impl(const py::array_t<double> vx,
                                                    const py::array_t<double> vy,
                                                    const py::array_t<double> vz,
                                                    double dx) {
    auto vxv = vx.unchecked<3>();
    auto vyv = vy.unchecked<3>();
    auto vzv = vz.unchecked<3>();
    const int Ng = vxv.shape(0);
    const double inv_2dx = 1.0 / (2.0 * dx);

    py::array_t<double> S_tf({Ng, Ng, Ng});
    auto out = S_tf.mutable_unchecked<3>();

    #pragma omp parallel for collapse(3)
    for (int i = 0; i < Ng; ++i) {
        for (int j = 0; j < Ng; ++j) {
            for (int k = 0; k < Ng; ++k) {
                int ip = wrap(i + 1, Ng);
                int im = wrap(i - 1, Ng);
                int jp = wrap(j + 1, Ng);
                int jm = wrap(j - 1, Ng);
                int kp = wrap(k + 1, Ng);
                int km = wrap(k - 1, Ng);

                double dvx_dx = (vxv(ip, j, k) - vxv(im, j, k)) * inv_2dx;
                double dvx_dy = (vxv(i, jp, k) - vxv(i, jm, k)) * inv_2dx;
                double dvx_dz = (vxv(i, j, kp) - vxv(i, j, km)) * inv_2dx;

                double dvy_dx = (vyv(ip, j, k) - vyv(im, j, k)) * inv_2dx;
                double dvy_dy = (vyv(i, jp, k) - vyv(i, jm, k)) * inv_2dx;
                double dvy_dz = (vyv(i, j, kp) - vyv(i, j, km)) * inv_2dx;

                double dvz_dx = (vzv(ip, j, k) - vzv(im, j, k)) * inv_2dx;
                double dvz_dy = (vzv(i, jp, k) - vzv(i, jm, k)) * inv_2dx;
                double dvz_dz = (vzv(i, j, kp) - vzv(i, j, km)) * inv_2dx;

                double Sxx = dvx_dx;
                double Syy = dvy_dy;
                double Szz = dvz_dz;
                double Sxy = 0.5 * (dvx_dy + dvy_dx);
                double Sxz = 0.5 * (dvx_dz + dvz_dx);
                double Syz = 0.5 * (dvy_dz + dvz_dy);

                double tr = Sxx + Syy + Szz;
                double one_third_tr = tr / 3.0;
                double Sxx_tf = Sxx - one_third_tr;
                double Syy_tf = Syy - one_third_tr;
                double Szz_tf = Szz - one_third_tr;

                double norm2 = Sxx_tf*Sxx_tf + Syy_tf*Syy_tf + Szz_tf*Szz_tf
                               + 2.0*(Sxy*Sxy + Sxz*Sxz + Syz*Syz);
                out(i, j, k) = norm2;
            }
        }
    }

    return S_tf;
}

ViscosityResult artificial_viscosity_q_cgs_impl(const py::array_t<double> rho_phys,
                                                const py::array_t<double> u_grid,
                                                const py::array_t<double> vx,
                                                const py::array_t<double> vy,
                                                const py::array_t<double> vz,
                                                double dx_cm,
                                                double a,
                                                double C2,
                                                double C1,
                                                double Ctheta,
                                                bool balsara,
                                                double eps) {
    const double dx_phys = a * dx_cm;

    py::array_t<double> div_v = divergence_central_impl(vx, vy, vz, dx_cm);
    py::array_t<double> curl_mag = curl_mag_cgs_impl(vx, vy, vz, dx_cm);
    py::array_t<double> S_tf = strain_tracefree_norm2_cgs_impl(vx, vy, vz, dx_cm);

    auto rho = rho_phys.unchecked<3>();
    auto u   = u_grid.unchecked<3>();
    auto div = div_v.mutable_unchecked<3>();
    auto curl = curl_mag.unchecked<3>();
    auto S_view = S_tf.unchecked<3>();
    auto vxv = vx.unchecked<3>();
    auto vyv = vy.unchecked<3>();
    auto vzv = vz.unchecked<3>();

    const int Ng = rho.shape(0);
    py::array_t<double> q_arr({Ng, Ng, Ng});
    auto q = q_arr.mutable_unchecked<3>();

    double S_max = 0.0;

    #pragma omp parallel for collapse(3) reduction(max:S_max)
    for (int i = 0; i < Ng; ++i) {
        for (int j = 0; j < Ng; ++j) {
            for (int k = 0; k < Ng; ++k) {
                double div_val = div(i, j, k);
                double theta = std::min(div_val, 0.0);
                double cs = std::sqrt(std::max(gamma_ad * (gamma_ad - 1.0) * u(i, j, k), 0.0));
                double theta_min = -Ctheta * cs / dx_phys;
                double theta_clamped = std::max(theta, theta_min);
                double abs_theta = -theta_clamped;

                double abs_theta_sq = abs_theta * abs_theta;
                double denom2 = eps * eps + abs_theta_sq;
                if (balsara) {
                    double curl_sq = curl(i, j, k);
                    curl_sq *= curl_sq;
                    denom2 += curl_sq + S_view(i, j, k);
                }

                double S_fac = balsara ? (denom2 > 0.0 ? abs_theta_sq / denom2 : 0.0) : 1.0;
                S_max = std::max(S_max, S_fac);

                double vx_val = vxv(i, j, k);
                double vy_val = vyv(i, j, k);
                double vz_val = vzv(i, j, k);
                double v_mag = std::sqrt(vx_val*vx_val + vy_val*vy_val + vz_val*vz_val);
                double M = (cs > eps) ? (v_mag / cs) : 0.0;
                double w1 = 1.0 / (1.0 + std::pow(M / 0.7, 4));

                double term2 = C2 * rho(i, j, k) * std::pow(dx_phys * abs_theta, 2);
                double term1 = (C1 * S_fac * w1) * rho(i, j, k) * cs * dx_phys * abs_theta;
                double q_val = (term2 + term1) * S_fac;
                if (div_val >= 0.0) q_val = 0.0;
                q(i, j, k) = q_val;
                div(i, j, k) = div_val;
            }
        }
    }

    return {std::move(q_arr), std::move(div_v), S_max};
}

} // namespace

py::tuple cic_deposit_vec_equal_mass(const py::array_t<double> pos_Mpc,
                                     const py::array_t<double> vec,
                                     double Lbox_Mpc,
                                     int Ng) {
    auto res = cic_deposit_vec_equal_mass_impl(pos_Mpc, vec, Lbox_Mpc, Ng);
    return py::make_tuple(res[0], res[1], res[2]);
}

py::tuple gradient_central(const py::array_t<double> grid, double dx) {
    auto grads = gradient_central_impl(grid, dx);
    return py::make_tuple(grads[0], grads[1], grads[2]);
}

py::array_t<double> divergence_central(const py::array_t<double> vx,
                                       const py::array_t<double> vy,
                                       const py::array_t<double> vz,
                                       double dx) {
    return divergence_central_impl(vx, vy, vz, dx);
}

py::tuple pressure_acceleration_grid(const py::array_t<double> rho_b_com_grid,
                                      const py::array_t<double> P_phys,
                                      double a,
                                      double dx) {
    auto res = pressure_acceleration_grid_impl(rho_b_com_grid, P_phys, a, dx);
    return py::make_tuple(res[0], res[1], res[2]);
}

py::array_t<double> gather_vec(const py::array_t<double> ax,
                               const py::array_t<double> ay,
                               const py::array_t<double> az,
                               const py::array_t<double> pos_Mpc,
                               double Lbox_Mpc,
                               double dx,
                               int n_threads) {
    return gather_vec_impl(ax, ay, az, pos_Mpc, Lbox_Mpc, dx, n_threads);
}

py::array_t<double> curl_mag_cgs(const py::array_t<double> vx,
                                 const py::array_t<double> vy,
                                 const py::array_t<double> vz,
                                 double dx_cm) {
    return curl_mag_cgs_impl(vx, vy, vz, dx_cm);
}

py::array_t<double> strain_tracefree_norm2_cgs(const py::array_t<double> vx,
                                               const py::array_t<double> vy,
                                               const py::array_t<double> vz,
                                               double dx_cm) {
    return strain_tracefree_norm2_cgs_impl(vx, vy, vz, dx_cm);
}

py::tuple artificial_viscosity_q_cgs(const py::array_t<double> rho_phys,
                                     const py::array_t<double> u_grid,
                                     const py::array_t<double> vx,
                                     const py::array_t<double> vy,
                                     const py::array_t<double> vz,
                                     double dx_cm,
                                     double a,
                                     double C2,
                                     double C1,
                                     double Ctheta,
                                     bool balsara,
                                     double eps) {
    auto res = artificial_viscosity_q_cgs_impl(rho_phys, u_grid, vx, vy, vz, dx_cm, a, C2, C1, Ctheta, balsara, eps);
    return py::make_tuple(res.q, res.div_v, py::float_(res.S_max));
}

py::tuple cooling_heating_step_cpp(
    const py::array_t<double> pos_Mpc,
    const py::array_t<double> vel_Mpc_per_Myr,
    const py::array_t<uint8_t, py::array::c_style | py::array::forcecast> baryon_mask,
    double mass_Msun,
    const py::array_t<double> u_baryons,
    double Lbox_Mpc,
    int Ng,
    double dt_Myr,
    double a,
    bool use_TSC = false,
    int n_threads = 8,
    double z_reion = 8.0,
    double dz_trans = 0.5,
    double C_cfl = 0.2,
    double f_dyn = 0.2
) {
    if (!g_cosmo.is_set) {
        throw std::runtime_error("Cosmology parameters not initialised. Call set_cosmology_params before cooling_heating_step.");
    }
    auto pos_all = pos_Mpc.unchecked<2>();
    auto vel_all = vel_Mpc_per_Myr.unchecked<2>();
    auto mask = baryon_mask.unchecked<1>();

    const ssize_t N_total = pos_all.shape(0);
    if (vel_all.shape(0) != N_total || vel_all.shape(1) != 3) {
        throw std::runtime_error("vel_Mpc_per_Myr must have shape (N,3)");
    }
    if (pos_all.shape(1) != 3) {
        throw std::runtime_error("pos_Mpc must have shape (N,3)");
    }
    if (mask.size() != N_total) {
        throw std::runtime_error("baryon_mask shape mismatch");
    }

    std::vector<ssize_t> baryon_indices;
    baryon_indices.reserve(N_total);
    for (ssize_t i = 0; i < N_total; ++i) {
        if (mask(i)) baryon_indices.push_back(i);
    }

    const ssize_t Nb = static_cast<ssize_t>(baryon_indices.size());
    if (u_baryons.ndim() != 1 || u_baryons.shape(0) != Nb) {
        throw std::runtime_error("u_baryons must be a 1D array matching baryon count");
    }

    py::array_t<double> pos_baryon({Nb, static_cast<ssize_t>(3)});
    py::array_t<double> vel_baryon({Nb, static_cast<ssize_t>(3)});
    auto pos_b = pos_baryon.mutable_unchecked<2>();
    auto vel_b = vel_baryon.mutable_unchecked<2>();
    auto u_b = u_baryons.unchecked<1>();

    #pragma omp parallel for num_threads(n_threads)
    for (ssize_t idx = 0; idx < Nb; ++idx) {
        ssize_t i = baryon_indices[idx];
        pos_b(idx, 0) = pos_all(i, 0);
        pos_b(idx, 1) = pos_all(i, 1);
        pos_b(idx, 2) = pos_all(i, 2);
        vel_b(idx, 0) = vel_all(i, 0);
        vel_b(idx, 1) = vel_all(i, 1);
        vel_b(idx, 2) = vel_all(i, 2);
    }

    double z_now = 1.0 / std::max(a, 1e-6) - 1.0;
    double dt_s = dt_Myr * Myr_to_s;

    auto rho_pair = deposit_density(pos_baryon, Lbox_Mpc, Ng, use_TSC);
    py::array_t<double> rho_b_com_grid = std::move(rho_pair.first);
    double dx = rho_pair.second;

    auto rho_b_com = rho_b_com_grid.mutable_unchecked<3>();
    double sum_rho = 0.0;
    const double mass_to_g = mass_Msun * Msun_to_g / Mpc_to_cm_cubed;

    py::array_t<double> od_b_grid({Ng, Ng, Ng});
    auto od_b = od_b_grid.mutable_unchecked<3>();
    const ssize_t Ng3 = static_cast<ssize_t>(Ng) * Ng * Ng;

    #pragma omp parallel for collapse(3) num_threads(n_threads) schedule(static) reduction(+:sum_rho) //TESTING
    for (int i = 0; i < Ng; ++i)
    for (int j = 0; j < Ng; ++j)
    for (int k = 0; k < Ng; ++k) {
        double val = rho_b_com(i, j, k);
        sum_rho += val;
    }

    double mean_rho = sum_rho / Ng3;
    double inv_mean_rho = mean_rho > 0.0 ? 1.0 / mean_rho : 0.0;
    const double a_safe = std::max(a, 1e-12);
    const double inv_a3 = 1.0 / (a_safe * a_safe * a_safe);
    double max_b_od = 0.0;

    py::array_t<double> rho_b_phys_grid({Ng, Ng, Ng});
    auto rho_b_phys = rho_b_phys_grid.mutable_unchecked<3>();
    py::array_t<double> nH_grid({Ng, Ng, Ng});
    auto nH = nH_grid.mutable_unchecked<3>();
    double nH_sum = 0.0;

    #pragma omp parallel for collapse(3) num_threads(n_threads) schedule(static) reduction(+:nH_sum) reduction(max:max_b_od) //TESTING
    for (int i = 0; i < Ng; ++i)
    for (int j = 0; j < Ng; ++j)
    for (int k = 0; k < Ng; ++k) {
        double val_com = rho_b_com(i, j, k);
        double rho_mass = val_com * mass_to_g;
        rho_b_com(i, j, k) = rho_mass;

        double od_val = mean_rho > 0.0 ? val_com * inv_mean_rho : 0.0;
        od_b(i, j, k) = od_val;
        max_b_od = std::max(max_b_od, od_val);

        double rho_phys_val = rho_mass * inv_a3;
        rho_b_phys(i, j, k) = rho_phys_val;

        double nH_val = (X_H * rho_phys_val) / m_p;
        nH(i, j, k) = nH_val;
        nH_sum += nH_val;
    }
    double nH_vol_mean = nH_sum / Ng3;

    auto u_pair = deposit_energy(pos_baryon, u_baryons, Lbox_Mpc, Ng, use_TSC);
    py::array_t<double> u_grid = std::move(u_pair.first);

    double mu_eff = (z_now > z_reion) ? mu_neutral : mu_ion;

    auto u_view = u_grid.mutable_unchecked<3>();
    py::array_t<double> T_grid({Ng, Ng, Ng});
    auto T_view = T_grid.mutable_unchecked<3>();
    double T_CMB = T_CMB_of_z(z_now);

    #pragma omp parallel for collapse(3) num_threads(n_threads) schedule(static) //TESTING
    for (int i = 0; i < Ng; ++i)
    for (int j = 0; j < Ng; ++j)
    for (int k = 0; k < Ng; ++k) {
        double T_val = T_from_u(u_view(i, j, k), mu_eff);
        T_val = std::clamp(T_val, T_CMB, 1e9);
        T_view(i, j, k) = T_val;
        u_view(i, j, k) = u_from_T(T_val, mu_eff);
    }

    double w_hi = 1.0 / (1.0 + std::exp(-(z_now - z_reion) / dz_trans));
    const double w_thres = 1e-5;
    bool epoch_hi = w_hi > w_thres;
    bool epoch_lo = (1.0 - w_hi) > w_thres;

    py::array_t<double> C_vol({Ng, Ng, Ng});
    std::memset(C_vol.mutable_data(), 0, sizeof(double) * Ng3);
    auto C_view = C_vol.mutable_unchecked<3>();

    if (epoch_hi) {
        auto collis_table = require_lambda_collis();
        #pragma omp parallel for collapse(3) num_threads(n_threads) schedule(static)
        for (int i = 0; i < Ng; ++i) {
            for (int j = 0; j < Ng; ++j) {
                for (int k = 0; k < Ng; ++k) {
                    double lambda = collis_table->evaluate(T_view(i, j, k));
                    double nH_sq = nH(i, j, k) * nH(i, j, k);
                    C_view(i, j, k) += w_hi * nH_sq * lambda;
                }
            }
        }
    }

    double Zsol_min = 0.0;
    double Zsol_max = 0.0;
    double Zsol_mean = 0.0;
    if (epoch_lo) {
        auto lambda_table = require_lambda_table();
        double sum_Z = 0.0;
        double min_Z = std::numeric_limits<double>::infinity();
        double max_Z = -std::numeric_limits<double>::infinity();

        #pragma omp parallel for collapse(3) num_threads(n_threads) schedule(static) reduction(+:sum_Z) reduction(min:min_Z) reduction(max:max_Z)
        for (int i = 0; i < Ng; ++i) {
            for (int j = 0; j < Ng; ++j) {
                for (int k = 0; k < Ng; ++k) {
                    double Zsol = Zsolar_of_nH_z(nH(i, j, k), z_now);
                    double lambda = lambda_table->evaluate(z_now, Zsol, nH(i, j, k), T_view(i, j, k));
                    double nH_sq = nH(i, j, k) * nH(i, j, k);
                    C_view(i, j, k) += (1.0 - w_hi) * nH_sq * lambda;
                    sum_Z += Zsol;
                    min_Z = std::min(min_Z, Zsol);
                    max_Z = std::max(max_Z, Zsol);
                }
            }
        }

        Zsol_min = min_Z;
        Zsol_max = max_Z;
        Zsol_mean = sum_Z / Ng3;
    }

    Vec3Grid vec_field = cic_deposit_vec_equal_mass_impl(pos_baryon, vel_baryon, Lbox_Mpc, Ng);
    auto vx = std::move(vec_field[0]);
    auto vy = std::move(vec_field[1]);
    auto vz = std::move(vec_field[2]);
    auto vx_mut = vx.mutable_unchecked<3>();
    auto vy_mut = vy.mutable_unchecked<3>();
    auto vz_mut = vz.mutable_unchecked<3>();
    const double vel_conv = Mpc_to_cm / Myr_to_s;

    #pragma omp parallel for collapse(3) num_threads(n_threads) schedule(static)
    for (int i = 0; i < Ng; ++i) {
        for (int j = 0; j < Ng; ++j) {
            for (int k = 0; k < Ng; ++k) {
                vx_mut(i, j, k) *= vel_conv;
                vy_mut(i, j, k) *= vel_conv;
                vz_mut(i, j, k) *= vel_conv;
            }
        }
    }

    ViscosityResult visc = artificial_viscosity_q_cgs_impl(
        rho_b_phys_grid, u_grid, vx, vy, vz, dx * Mpc_to_cm, a,
        1.5, 0.5, 10.0, true, 1e-30);

    py::array_t<double> P_phys_grid({Ng, Ng, Ng});
    auto P_phys = P_phys_grid.mutable_unchecked<3>();

    auto q_visc = std::move(visc.q);
    auto div_v_s = std::move(visc.div_v);
    double S_min = visc.S_max;

    auto q_visc_view = q_visc.unchecked<3>();
    auto div_view = div_v_s.unchecked<3>();

    #pragma omp parallel for collapse(3) num_threads(n_threads) schedule(static)
    for (int i = 0; i < Ng; ++i)
    for (int j = 0; j < Ng; ++j)
    for (int k = 0; k < Ng; ++k) {
        P_phys(i, j, k) = (gamma_ad - 1.0) * rho_b_phys(i, j, k) * u_view(i, j, k) + q_visc_view(i, j, k);
    }

    Vec3Grid aP_grid = pressure_acceleration_grid_impl(rho_b_com_grid, P_phys_grid, a, dx * Mpc_to_cm);
    auto aPx = std::move(aP_grid[0]);
    auto aPy = std::move(aP_grid[1]);
    auto aPz = std::move(aP_grid[2]);

    const double acc_conv = (Myr_to_s * Myr_to_s) / Mpc_to_cm;
    auto aPx_mut = aPx.mutable_unchecked<3>();
    auto aPy_mut = aPy.mutable_unchecked<3>();
    auto aPz_mut = aPz.mutable_unchecked<3>();

    #pragma omp parallel for collapse(3) num_threads(n_threads) schedule(static)
    for (int i = 0; i < Ng; ++i)
    for (int j = 0; j < Ng; ++j)
    for (int k = 0; k < Ng; ++k) {
        aPx_mut(i, j, k) *= acc_conv;
        aPy_mut(i, j, k) *= acc_conv;
        aPz_mut(i, j, k) *= acc_conv;
    }

    py::array_t<double> aP_new = gather_vec_impl(aPx, aPy, aPz, pos_baryon, Lbox_Mpc, dx, n_threads);

    double c_s_max_cms = 0.0;
    #pragma omp parallel for collapse(3) num_threads(n_threads) schedule(static) reduction(max:c_s_max_cms) //TESTING
    for (int i = 0; i < Ng; ++i)
    for (int j = 0; j < Ng; ++j)
    for (int k = 0; k < Ng; ++k) {
        double cs = std::sqrt(std::max(gamma_ad * (gamma_ad - 1.0) * u_view(i, j, k), 0.0));
        c_s_max_cms = std::max(c_s_max_cms, cs);
    }

    double v_max_Mpc_Myr = 0.0;
    #pragma omp parallel for reduction(max:v_max_Mpc_Myr) num_threads(n_threads)
    for (ssize_t idx = 0; idx < Nb; ++idx) {
        double vx_val = vel_b(idx, 0);
        double vy_val = vel_b(idx, 1);
        double vz_val = vel_b(idx, 2);
        double vmag = std::sqrt(vx_val*vx_val + vy_val*vy_val + vz_val*vz_val);
        v_max_Mpc_Myr = std::max(v_max_Mpc_Myr, vmag);
    }

    double c_s_max_Mpc_Myr = c_s_max_cms * (Myr_to_s / Mpc_to_cm) / std::max(a, 1e-12);
    double dt_cfl = C_cfl * dx / std::max(c_s_max_Mpc_Myr + v_max_Mpc_Myr, 1e-20);

    auto rho_tot_pair = deposit_density(pos_Mpc, Lbox_Mpc, Ng, use_TSC);
    py::array_t<double> rho_tot_com_grid = std::move(rho_tot_pair.first);
    auto rho_tot = rho_tot_com_grid.mutable_unchecked<3>();

    double sum_tot = 0.0;
    double max_tot = 0.0;
    double rho_tot_phys_max = 0.0;
    #pragma omp parallel for collapse(3) num_threads(n_threads) schedule(static) reduction(max:max_tot, rho_tot_phys_max) reduction(+:sum_tot) //TESTING
    for (int i = 0; i < Ng; ++i)
    for (int j = 0; j < Ng; ++j)
    for (int k = 0; k < Ng; ++k) {
        double val = rho_tot(i, j, k);
        sum_tot += val;
        max_tot = std::max(max_tot, val);
        double rho_mass = val * mass_to_g;
        rho_tot(i, j, k) = rho_mass;
        rho_tot_phys_max = std::max(rho_tot_phys_max, rho_mass * inv_a3);
    }
    double mean_tot = sum_tot / Ng3;
    double max_tot_od = (mean_tot > 0.0) ? max_tot / mean_tot : 0.0;

    double t_dyn_s = f_dyn / std::sqrt(std::max(4.0 * M_PI * G_cgs * rho_tot_phys_max, 1e-40));
    double dt_dyn = t_dyn_s / Myr_to_s;
    double dt_max = std::min(dt_cfl, dt_dyn);

    auto lamS = lambda_S_Compton(z_now, mu_eff, 2e-4, 0.079, w_hi);
    double lam_comp = lamS.first;
    double S_comp = lamS.second;

    py::array_t<double> u_grid_new({Ng, Ng, Ng});
    auto u_new_grid = u_grid_new.mutable_unchecked<3>();

    #pragma omp parallel for collapse(3) num_threads(n_threads) schedule(static) //TESTING
    for (int i = 0; i < Ng; ++i)
    for (int j = 0; j < Ng; ++j)
    for (int k = 0; k < Ng; ++k) {
        double rho_phys_val = rho_b_phys(i, j, k);
        double cool_rate = C_view(i, j, k) / std::max(rho_phys_val, 1e-32);
        double t_cool = u_view(i, j, k) / std::max(cool_rate, 1e-40);
        double lam = lam_comp + 1.0 / std::max(t_cool, 1e-30) + (gamma_ad - 1.0) * (div_view(i, j, k) / std::max(a, 1e-12));
        double S_term = S_comp - q_visc_view(i, j, k) / std::max(rho_phys_val, 1e-40) * (div_view(i, j, k) / std::max(a, 1e-12));
        double arg = lam * dt_s;
        if (std::abs(arg) > 1e-10) {
            double e = std::exp(-arg);
            u_new_grid(i, j, k) = u_view(i, j, k) * e + (S_term / lam) * (1.0 - e);
        } else {
            u_new_grid(i, j, k) = u_view(i, j, k) + S_term * dt_s;
        }
    }

    py::array_t<double> u_new = gather(u_grid_new, pos_baryon, Lbox_Mpc, dx, use_TSC, n_threads);
    auto u_new_view = u_new.mutable_unchecked<1>();
    double H_a = g_cosmo.H0_cos * E_of_a(a);

    #pragma omp parallel for num_threads(n_threads)
    for (ssize_t idx = 0; idx < Nb; ++idx) {
        u_new_view(idx) += -2.0 * dt_Myr * H_a * u_b(idx);
    }

    py::array_t<double> T_grid_new({Ng, Ng, Ng});
    auto T_new_grid = T_grid_new.mutable_unchecked<3>();

    #pragma omp parallel for collapse(3) num_threads(n_threads) schedule(static) //TESTING
    for (int i = 0; i < Ng; ++i)
    for (int j = 0; j < Ng; ++j)
    for (int k = 0; k < Ng; ++k) {
        double T_val = T_from_u(u_new_grid(i, j, k), mu_eff);
        T_new_grid(i, j, k) = std::clamp(T_val, 0.0, 1e12);
    }

    py::array_t<double> T_new = gather(T_grid_new, pos_baryon, Lbox_Mpc, dx, use_TSC, n_threads);
    py::array_t<double> P_new = gather(P_phys_grid, pos_baryon, Lbox_Mpc, dx, use_TSC, n_threads);

    double temp_bins_sum[5] = {0,0,0,0,0};
    size_t temp_bins_count[5] = {0,0,0,0,0};
    /*
    #pragma omp parallel
    {
        double local_sum[5] = {0,0,0,0,0};
        size_t local_count[5] = {0,0,0,0,0};

        #pragma omp for collapse(3) schedule(static)
        for (int i = 0; i < Ng; ++i)
        for (int j = 0; j < Ng; ++j)
        for (int k = 0; k < Ng; ++k) {
            double od = od_b(i, j, k);
            double temp = T_view(i, j, k);
            int bin = 0;
            if (od < 1.0) bin = 0;
            else if (od < 10.0) bin = 1;
            else if (od < 100.0) bin = 2;
            else if (od < 1000.0) bin = 3;
            else bin = 4;
            local_sum[bin] += temp;
            local_count[bin] += 1;
        }

        #pragma omp critical
        {
            for (int b = 0; b < 5; ++b) {
                temp_bins_sum[b] += local_sum[b];
                temp_bins_count[b] += local_count[b];
            }
        }
    }
    */
    #pragma omp parallel for collapse(3) schedule(static) reduction(+:temp_bins_sum[:5], temp_bins_count[:5])
    for (int i = 0; i < Ng; ++i)
    for (int j = 0; j < Ng; ++j)
    for (int k = 0; k < Ng; ++k) {
        double od   = od_b(i, j, k);
        double temp = T_view(i, j, k);
        int bin;

        if      (od < 1.0)    bin = 0;
        else if (od < 10.0)   bin = 1;
        else if (od < 100.0)  bin = 2;
        else if (od < 1000.0) bin = 3;
        else                  bin = 4;

        temp_bins_sum[bin]   += temp;
        temp_bins_count[bin] += 1;
    }

    py::list temp_by_b_od;
    for (int i = 0; i < 5; ++i) {
        if (temp_bins_count[i] > 0) {
            temp_by_b_od.append(temp_bins_sum[i] / temp_bins_count[i]);
        } else {
            temp_by_b_od.append(py::float_(std::numeric_limits<double>::quiet_NaN()));
        }
    }

    double Cvol_sum = 0.0;
    #pragma omp parallel for collapse(3) num_threads(n_threads) schedule(static) reduction(+:Cvol_sum) //TESTING
    for (int i = 0; i < Ng; ++i)
    for (int j = 0; j < Ng; ++j)
    for (int k = 0; k < Ng; ++k) {
        Cvol_sum += C_view(i, j, k);
    }
    double Cvol_mean = Cvol_sum / Ng3;

    py::dict diag;
    diag["nH_vol_mean"] = nH_vol_mean;
    diag["Cvol_mean"] = Cvol_mean;
    diag["Zsol_min"] = Zsol_min;
    diag["Zsol_max"] = Zsol_max;
    diag["Zsol_mean"] = Zsol_mean;

    py::list epochs_list;
    epochs_list.append(py::bool_(epoch_hi));
    epochs_list.append(py::bool_(epoch_lo));

    return py::make_tuple(
        u_new,
        T_new,
        aP_new,
        P_new,
        diag,
        epochs_list,
        dt_max,
        max_b_od,
        max_tot_od,
        temp_by_b_od,
        S_min
    );
}

py::tuple combined_step_cpp(
    const py::array_t<double> pos_Mpc,
    const py::array_t<double> vel_Mpc_per_Myr,
    const py::array_t<double> acc_in,
    const py::array_t<double> u_baryons,
    const py::array_t<uint8_t, py::array::c_style | py::array::forcecast> baryon_mask,
    double mass_Msun,
    double Lbox_Mpc,
    int Ng,
    double dt_Myr,
    double a,
    double G_const,
    PMEngine &eng,
    bool use_TSC = false,
    int n_threads = 8,
    double z_reion = 8.0,
    double dz_trans = 0.5,
    double C_cfl = 0.2,
    double f_dyn = 0.2
) {
    if (!g_cosmo.is_set) {
        throw std::runtime_error("Cosmology parameters not initialised. Call set_cosmology_params before combined_step.");
    }

    auto pos_view = pos_Mpc.unchecked<2>();
    auto vel_view = vel_Mpc_per_Myr.unchecked<2>();
    auto acc_view = acc_in.unchecked<2>();
    auto mask_view = baryon_mask.unchecked<1>();

    const ssize_t N_total = pos_view.shape(0);
    if (vel_view.shape(0) != N_total || vel_view.shape(1) != 3)
        throw std::runtime_error("vel_Mpc_per_Myr must have shape (N,3)");
    if (pos_view.shape(1) != 3)
        throw std::runtime_error("pos_Mpc must have shape (N,3)");

    std::vector<ssize_t> baryon_indices;
    baryon_indices.reserve(N_total);
    for (ssize_t i = 0; i < N_total; ++i)
        if (mask_view(i)) baryon_indices.push_back(i);

    const ssize_t Nb = static_cast<ssize_t>(baryon_indices.size());
    if (u_baryons.ndim() != 1 || u_baryons.shape(0) != Nb)
        throw std::runtime_error("u_baryons must be a 1D array matching baryon count");

    py::array_t<double> pos_out({N_total, static_cast<ssize_t>(3)});
    py::array_t<double> vel_out({N_total, static_cast<ssize_t>(3)});

    auto pos_out_mut = pos_out.mutable_unchecked<2>();
    auto vel_out_mut = vel_out.mutable_unchecked<2>();
    std::vector<double> u_half(static_cast<size_t>(N_total) * 3);

    const double H_a = g_cosmo.H0_cos * E_of_a(a);
    const double a_new = a + dt_Myr * a * H_a;

    #pragma omp parallel for num_threads(n_threads)
    for (ssize_t i = 0; i < N_total; ++i) {
        const double vx = vel_view(i, 0);
        const double vy = vel_view(i, 1);
        const double vz = vel_view(i, 2);
        const double gx = acc_view(i, 0);
        const double gy = acc_view(i, 1);
        const double gz = acc_view(i, 2);

        const double uhx = vx + 0.5 * dt_Myr * gx;
        const double uhy = vy + 0.5 * dt_Myr * gy;
        const double uhz = vz + 0.5 * dt_Myr * gz;

        double x_new = pos_view(i, 0) + dt_Myr * uhx;
        double y_new = pos_view(i, 1) + dt_Myr * uhy;
        double z_new = pos_view(i, 2) + dt_Myr * uhz;

        x_new -= Lbox_Mpc * std::floor(x_new / Lbox_Mpc);
        if (x_new < 0.0) x_new += Lbox_Mpc;
        y_new -= Lbox_Mpc * std::floor(y_new / Lbox_Mpc);
        if (y_new < 0.0) y_new += Lbox_Mpc;
        z_new -= Lbox_Mpc * std::floor(z_new / Lbox_Mpc);
        if (z_new < 0.0) z_new += Lbox_Mpc;

        pos_out_mut(i, 0) = x_new;
        pos_out_mut(i, 1) = y_new;
        pos_out_mut(i, 2) = z_new;

        u_half[3 * static_cast<size_t>(i) + 0] = uhx;
        u_half[3 * static_cast<size_t>(i) + 1] = uhy;
        u_half[3 * static_cast<size_t>(i) + 2] = uhz;
    }

        std::atomic<bool> eng_thread_ready{false};
    std::exception_ptr eng_exc;
    py::object step_result_obj;

    std::thread eng_thread([&]() {
        try {
            py::gil_scoped_acquire acquire;
            eng_thread_ready.store(true, std::memory_order_release);
            step_result_obj = eng.step(pos_out, mass_Msun, a, G_const);
        } catch (...) {
            eng_exc = std::current_exception();
            eng_thread_ready.store(true, std::memory_order_release);
        }
    });

    {
        py::gil_scoped_release release;
        while (!eng_thread_ready.load(std::memory_order_acquire)) {
            std::this_thread::sleep_for(std::chrono::microseconds(50));
        }
    }

    py::tuple cooling;
    try {
        cooling = cooling_heating_step_cpp(
            pos_Mpc,
            vel_Mpc_per_Myr,
            baryon_mask,
            mass_Msun,
            u_baryons,
            Lbox_Mpc,
            Ng,
            dt_Myr,
            a,
            use_TSC,
            n_threads,
            z_reion,
            dz_trans,
            C_cfl,
            f_dyn
        );
    } catch (...) {
        {
            py::gil_scoped_release release;
            if (eng_thread.joinable()) {
                eng_thread.join();
            }
        }
        if (eng_exc) {
            std::rethrow_exception(eng_exc);
        }
        throw;
    }

    {
        py::gil_scoped_release release;
        if (eng_thread.joinable()) {
            eng_thread.join();
        }
    }

    if (eng_exc) {
        std::rethrow_exception(eng_exc);
    }

    py::tuple step_result = step_result_obj.cast<py::tuple>();

    py::array_t<double> u_new   = cooling[0].cast<py::array_t<double>>();
    py::array_t<double> T       = cooling[1].cast<py::array_t<double>>();
    py::array_t<double> acc_P   = cooling[2].cast<py::array_t<double>>();
    py::array_t<double> P       = cooling[3].cast<py::array_t<double>>();
    py::dict diag               = cooling[4].cast<py::dict>();
    py::list epochs             = cooling[5].cast<py::list>();
    double dt_target_new        = cooling[6].cast<double>();
    double max_b_od             = cooling[7].cast<double>();
    double max_tot_od           = cooling[8].cast<double>();
    py::list temp_by_b_od       = cooling[9].cast<py::list>();
    double S_min                = cooling[10].cast<double>();

    py::array_t<double> acc_out = step_result[0].cast<py::array_t<double>>();
    double PE = step_result[2].cast<double>();

    auto acc_out_mut = acc_out.mutable_unchecked<2>();
    #pragma omp parallel for num_threads(n_threads)
    for (ssize_t i = 0; i < N_total; ++i) {
        const size_t idx = static_cast<size_t>(i) * 3;
        acc_out_mut(i, 0) -= 2.0 * H_a * u_half[idx + 0];
        acc_out_mut(i, 1) -= 2.0 * H_a * u_half[idx + 1];
        acc_out_mut(i, 2) -= 2.0 * H_a * u_half[idx + 2];

        vel_out_mut(i, 0) = u_half[idx + 0] + 0.5 * dt_Myr * acc_out_mut(i, 0);
        vel_out_mut(i, 1) = u_half[idx + 1] + 0.5 * dt_Myr * acc_out_mut(i, 1);
        vel_out_mut(i, 2) = u_half[idx + 2] + 0.5 * dt_Myr * acc_out_mut(i, 2);
    }
   
    auto accP_view = acc_P.unchecked<2>();
    #pragma omp parallel for num_threads(n_threads)
    for (ssize_t idx = 0; idx < Nb; ++idx) {
        const ssize_t i = baryon_indices[static_cast<size_t>(idx)];
        vel_out_mut(i, 0) += dt_Myr * accP_view(idx, 0);
        vel_out_mut(i, 1) += dt_Myr * accP_view(idx, 1);
        vel_out_mut(i, 2) += dt_Myr * accP_view(idx, 2);
    }

    return py::make_tuple(
        pos_out,
        vel_out,
        acc_out,
        py::float_(a_new),
        py::float_(PE),
        u_new,
        T,
        acc_P,
        P,
        diag,
        epochs,
        py::float_(dt_target_new),
        py::float_(max_b_od),
        py::float_(max_tot_od),
        temp_by_b_od,
        py::float_(S_min)
    );
}

py::list fof_groups_cpp(const py::array_t<double, py::array::c_style | py::array::forcecast>& positions, 
                        double linking_length, 
                        py::ssize_t min_group_size = 20
) {
    auto buf = positions.request();
    if (buf.ndim != 2 || buf.shape[1] != 3) {
        throw std::runtime_error("positions must have shape (N, 3)");
    }

    const py::ssize_t N = buf.shape[0];
    const double* pos_ptr = static_cast<const double*>(buf.ptr);
    std::vector<std::vector<py::ssize_t>> groups;

    {
        py::gil_scoped_release release;

        if (N == 0 || linking_length <= 0.0) {
            // nothing to do (Python version would return empty after size filter)
        } else {
            const double inv_cell = 1.0 / linking_length;
            const double ll2 = linking_length * linking_length;

            std::vector<CellKey> particle_cells(static_cast<size_t>(N));
            #pragma omp parallel for schedule(static)
            for (py::ssize_t i = 0; i < N; ++i) {
                double x = pos_ptr[3 * static_cast<size_t>(i) + 0];
                double y = pos_ptr[3 * static_cast<size_t>(i) + 1];
                double z = pos_ptr[3 * static_cast<size_t>(i) + 2];
                particle_cells[static_cast<size_t>(i)] = {
                    static_cast<int64_t>(std::floor(x * inv_cell)),
                    static_cast<int64_t>(std::floor(y * inv_cell)),
                    static_cast<int64_t>(std::floor(z * inv_cell))
                };
            }

            struct CellEntry {
                CellKey key;
                py::ssize_t index;
            };

            std::vector<CellEntry> entries(static_cast<size_t>(N));
            #pragma omp parallel for schedule(static)
            for (py::ssize_t i = 0; i < N; ++i) {
                entries[static_cast<size_t>(i)] = CellEntry{particle_cells[static_cast<size_t>(i)], i};
            }

            std::sort(entries.begin(), entries.end(), [](const CellEntry& a, const CellEntry& b) {
                return cell_key_less(a.key, b.key);
            });

            std::vector<CellKey> unique_keys;
            std::vector<std::pair<size_t, size_t>> key_ranges;
            unique_keys.reserve(entries.size());
            key_ranges.reserve(entries.size());

            for (size_t start = 0; start < entries.size();) {
                size_t end = start + 1;
                while (end < entries.size() && cell_key_equal(entries[start].key, entries[end].key)) {
                    ++end;
                }
                unique_keys.push_back(entries[start].key);
                key_ranges.emplace_back(start, end);
                start = end;
            }

            auto find_cell_range = [&](const CellKey& key) -> const std::pair<size_t, size_t>* {
                auto it = std::lower_bound(unique_keys.begin(), unique_keys.end(), key,
                                           [](const CellKey& a, const CellKey& b) {
                                               return cell_key_less(a, b);
                                           });
                if (it == unique_keys.end()) {
                    return nullptr;
                }
                if (!cell_key_equal(*it, key)) {
                    return nullptr;
                }
                size_t idx = static_cast<size_t>(std::distance(unique_keys.begin(), it));
                return &key_ranges[idx];
            };

            std::vector<CellKey> neighbor_offsets;
            neighbor_offsets.reserve(27);
            for (int dx = -1; dx <= 1; ++dx) {
                for (int dy = -1; dy <= 1; ++dy) {
                    for (int dz = -1; dz <= 1; ++dz) {
                        neighbor_offsets.push_back(CellKey{dx, dy, dz});
                    }
                }
            }

            std::vector<char> visited(static_cast<size_t>(N), 0);
            std::vector<py::ssize_t> stack;
            stack.reserve(128);

            for (py::ssize_t i = 0; i < N; ++i) {
                if (visited[static_cast<size_t>(i)]) {
                    continue;
                }

                stack.clear();
                stack.push_back(i);
                visited[static_cast<size_t>(i)] = 1;

                std::vector<py::ssize_t> group;
                group.reserve(32);

                while (!stack.empty()) {
                    py::ssize_t idx = stack.back();
                    stack.pop_back();
                    group.push_back(idx);

                    const double base_x = pos_ptr[3 * static_cast<size_t>(idx) + 0];
                    const double base_y = pos_ptr[3 * static_cast<size_t>(idx) + 1];
                    const double base_z = pos_ptr[3 * static_cast<size_t>(idx) + 2];
                    const CellKey& base_cell = particle_cells[static_cast<size_t>(idx)];

                    for (const CellKey& offset : neighbor_offsets) {
                        CellKey neighbor_cell{
                            base_cell.x + offset.x,
                            base_cell.y + offset.y,
                            base_cell.z + offset.z
                        };

                        const auto* range = find_cell_range(neighbor_cell);
                        if (!range) {
                            continue;
                        }

                        for (size_t p = range->first; p < range->second; ++p) {
                            py::ssize_t nb = entries[p].index;
                            if (visited[static_cast<size_t>(nb)]) {
                                continue;
                            }
                            double dx = pos_ptr[3 * static_cast<size_t>(nb) + 0] - base_x;
                            double dy = pos_ptr[3 * static_cast<size_t>(nb) + 1] - base_y;
                            double dz = pos_ptr[3 * static_cast<size_t>(nb) + 2] - base_z;
                            double dist2 = dx*dx + dy*dy + dz*dz;
                            if (dist2 <= ll2) {
                                visited[static_cast<size_t>(nb)] = 1;
                                stack.push_back(nb);
                            }
                        }
                    }
                }

                if (group.size() >= static_cast<std::size_t>(std::max<py::ssize_t>(min_group_size, 1))) {
                    groups.push_back(std::move(group));
                }
            }

            std::stable_sort(groups.begin(), groups.end(), [](const auto& a, const auto& b) {
                return a.size() > b.size();
            });
        }
    }

    py::list result;
    for (const auto& group : groups) {
        py::array_t<py::ssize_t> arr(group.size());
        auto ptr = arr.mutable_data();
        std::memcpy(ptr, group.data(), group.size() * sizeof(py::ssize_t));
        result.append(std::move(arr));
    }
    return result;
}

PYBIND11_MODULE(grad_phi, m) {
    m.doc() = "Compute comoving potential gradient in C++ with OpenMP";
    m.def("set_cosmology_params", &set_cosmology_params_cpp,
          py::arg("H0_cos"),
          py::arg("Omega_m"),
          py::arg("Omega_r"),
          py::arg("Omega_k"),
          py::arg("Omega_lambda"),
          "Set cosmology parameters used by cooling_heating_step.");
    m.def("set_lambda_table", &set_lambda_table_cpp,
          py::arg("z_grid"),
          py::arg("Z_grid"),
          py::arg("nH_grid"),
          py::arg("T_grid"),
          py::arg("lambda_table"),
          "Initialise 4D cooling table (z, Z, nH, T). Pass physical grids; nH/T are logged internally.");
    m.def("set_lambda_collis", &set_lambda_collis_cpp,
          py::arg("T_grid"),
          py::arg("lambda_table"),
          "Initialise 1D collisional cooling table over temperature.");
    m.def("Lambda_T_nH_Z_z", &lambda_T_nH_Z_z_cpp,
          py::arg("T"),
          py::arg("nH"),
          py::arg("Z_solar"),
          py::arg("z"),
          py::arg("n_threads") = py::none(),
          "Evaluate interpolated cooling coefficient Λ(T, nH, Z, z). Scalars or matching arrays supported.");
    m.def("Lambda_collis", &lambda_collis_cpp,
          py::arg("T"),
          py::arg("n_threads") = py::none(),
          "Evaluate collisional cooling Λ as a function of temperature.");
    m.def("cooling_heating_step", &cooling_heating_step_cpp,
          py::arg("pos_Mpc"),
          py::arg("vel_Mpc_per_Myr"),
          py::arg("baryon_mask"),
          py::arg("mass_Msun"),
          py::arg("u_baryons"),
          py::arg("Lbox_Mpc"),
          py::arg("Ng"),
          py::arg("dt_Myr"),
          py::arg("a"),
          py::arg("use_TSC") = false,
          py::arg("n_threads") = 8,
          py::arg("z_reion") = 8.0,
          py::arg("dz_trans") = 0.5,
          py::arg("C_cfl") = 0.2,
          py::arg("f_dyn") = 0.2,
          "Full mesh-based cooling/heating update with pressure forces and timestep estimates.");
    m.def("fof_groups", &fof_groups_cpp,
          py::arg("positions"),
          py::arg("linking_length"),
          py::arg("min_group_size") = 20,
          "Friends-of-friends groups using a hashed grid search (returns list of index arrays).");
    m.def("combined_step", &combined_step_cpp,
          py::arg("pos_Mpc"),
          py::arg("vel_Mpc_per_Myr"),
          py::arg("acc_in"),
          py::arg("u_baryons"),
          py::arg("baryon_mask"),
          py::arg("mass_Msun"),
          py::arg("Lbox_Mpc"),
          py::arg("Ng"),
          py::arg("dt_Myr"),
          py::arg("a"),
          py::arg("G"),
          py::arg("eng"),
          py::arg("use_TSC") = false,
          py::arg("n_threads") = 8,
          py::arg("z_reion") = 8.0,
          py::arg("dz_trans") = 0.5,
          py::arg("C_cfl") = 0.2,
          py::arg("f_dyn") = 0.2,
          "Compute cooling/heating and leapfrog update entirely in C++ with OpenMP parallel loops.");
    m.def("compute_phi_grad", &compute_phi_grad,
          py::arg("x"), py::arg("eps"), py::arg("n_threads"), py::arg("L") = 0.0,
          "Compute ∇phi for all particles with softening eps");
    m.def("compute_energies", &compute_energies,
          py::arg("x"),
          py::arg("u"),
          py::arg("center"),
          py::arg("eps"),
          py::arg("H_a"),
          py::arg("n_threads"),
          "Compute kinetic and potential energy for cosmological N-body system. Multiply KE by mass and PE by G * mass");
    m.def("velocity_stat", &velocity_stat,
          py::arg("u"),
          py::arg("n_threads"),
          "Compute the ratio of pec velocities exceeding v_thres");
    m.def("compute_delta_field_CIC", &compute_delta_field_CIC,
          py::arg("positions"),
          py::arg("Ngrid"),
          py::arg("L"),
          py::arg("n_threads") = 8,
          "Compute delta field using CIC assignment");
    m.def("compute_delta_field", &compute_delta_field,
          py::arg("positions"),
          py::arg("Ngrid"),
          py::arg("L"),
          py::arg("use_TSC"),
          py::arg("n_threads") = 8,
          "Compute delta field using CIC or TSC assignment");
    m.def("interpolate_forces", &interpolate_forces_CIC,
          py::arg("Fx"),
          py::arg("Fy"),
          py::arg("Fz"),
          py::arg("pod"),
          py::arg("Ngrid"),
          py::arg("L"),
          py::arg("n_threads") = 8,
          "Trilinear interpolation of force fields");
    m.def("interpolate_forces_CIC", &interpolate_forces_CIC,
          py::arg("Fx"),
          py::arg("Fy"),
          py::arg("Fz"),
          py::arg("pod"),
          py::arg("Ngrid"),
          py::arg("L"),
          py::arg("n_threads") = 8,
          "Trilinear interpolation of force fields");
    m.def("interpolate_forces_TSC", &interpolate_forces_TSC,
          py::arg("Fx"),
          py::arg("Fy"),
          py::arg("Fz"),
          py::arg("pod"),
          py::arg("Ngrid"),
          py::arg("L"),
          py::arg("n_threads") = 8,
          "Trilinear interpolation of force fields using TSC");
    m.def("interpolate_potential", &interpolate_potential_CIC,
          py::arg("phi_in"),
          py::arg("pod"),
          py::arg("Ngrid"),
          py::arg("L"),
          py::arg("n_threads") = 8,
          "Trilinear interpolation of potential fields");
    m.def("interpolate_potential_CIC", &interpolate_potential_CIC,
          py::arg("phi_in"),
          py::arg("pod"),
          py::arg("Ngrid"),
          py::arg("L"),
          py::arg("n_threads") = 8,
          "Trilinear interpolation of potential fields");
    m.def("compute_delta_field_TSC", &compute_delta_field_TSC,
          py::arg("positions"),
          py::arg("Ngrid"),
          py::arg("L"),
          py::arg("n_threads") = 8,
          "Compute delta field using TSC assignment");
    py::class_<PMEngine>(m, "PMEngine")
        .def(py::init<int, double, bool, int>(),
             py::arg("Ngrid"), py::arg("box_size"), py::arg("use_TSC") = true, py::arg("n_threads") = 8)
        .def("step", &PMEngine::step, py::arg("pod"), py::arg("mass"), py::arg("a"), py::arg("G"));
    m.def("cic_deposit_density", &cic_deposit_density,
          py::arg("pos_Mpc"), py::arg("Lbox_Mpc"), py::arg("Ng"),
          "CIC mass deposition");
    m.def("cic_deposit_energy", &cic_deposit_energy,
          py::arg("pos_Mpc"), py::arg("scalar"), py::arg("Lbox_Mpc"), py::arg("Ng"),
          "CIC scalar energy deposition");
    m.def("tsc_deposit_density", &tsc_deposit_density,
          py::arg("pos_Mpc"), py::arg("Lbox_Mpc"), py::arg("Ng"),
          "TSC mass deposition");
    m.def("tsc_deposit_energy", &tsc_deposit_energy,
          py::arg("pos_Mpc"), py::arg("scalar"), py::arg("Lbox_Mpc"), py::arg("Ng"),
          "TSC scalar energy deposition");
    m.def("deposit_density", &deposit_density,
          py::arg("pos_Mpc"), py::arg("Lbox_Mpc"), py::arg("Ng"), py::arg("use_TSC"),
          "Mass deposition using CIC or TSC based on use_TSC flag.");
    m.def("deposit_energy", &deposit_energy,
          py::arg("pos_Mpc"), py::arg("scalar"), py::arg("Lbox_Mpc"), py::arg("Ng"), py::arg("use_TSC"),
          "Scalar deposition using CIC or TSC based on use_TSC flag.");
    m.def("cic_gather", &cic_gather,
          py::arg("grid"), py::arg("pos_Mpc"), py::arg("Lbox_Mpc"), py::arg("dx"), py::arg("n_threads") = 8,
          "CIC gather (interpolation)");
    m.def("gather", &gather,
          py::arg("grid"), py::arg("pos_Mpc"), py::arg("Lbox_Mpc"), py::arg("dx"), py::arg("use_TSC"), py::arg("n_threads") = 8,
          "Gather scalar field using CIC or TSC based on use_TSC");
    m.def("cic_deposit_vec_equal_mass", &cic_deposit_vec_equal_mass,
          py::arg("pos_Mpc"), py::arg("vec"), py::arg("Lbox_Mpc"), py::arg("Ng"),
          "Mass-weighted CIC deposition of vector field (equal-mass particles).");
    m.def("gradient_central", &gradient_central,
          py::arg("grid"), py::arg("dx"),
          "Central-difference gradient of a periodic scalar field.");
    m.def("divergence_central", &divergence_central,
          py::arg("vx"), py::arg("vy"), py::arg("vz"), py::arg("dx"),
          "Central-difference divergence of a periodic vector field.");
    m.def("pressure_acceleration_grid", &pressure_acceleration_grid,
          py::arg("rho_b_com_grid"), py::arg("P_phys"), py::arg("a"), py::arg("dx"),
          "Compute pressure acceleration components on the mesh.");
    m.def("gather_vec", &gather_vec,
          py::arg("ax"), py::arg("ay"), py::arg("az"),
          py::arg("pos_Mpc"), py::arg("Lbox_Mpc"), py::arg("dx"), py::arg("n_threads") = 8,
          "Gather vector field from mesh to particle positions via CIC.");
    m.def("curl_mag_cgs", &curl_mag_cgs,
          py::arg("vx"), py::arg("vy"), py::arg("vz"), py::arg("dx_cm"),
          "Magnitude of curl using centred differences (periodic).");
    m.def("strain_tracefree_norm2_cgs", &strain_tracefree_norm2_cgs,
          py::arg("vx"), py::arg("vy"), py::arg("vz"), py::arg("dx_cm"),
          "Trace-free strain tensor norm squared (periodic centred differences).");
    m.def("artificial_viscosity_q_cgs", &artificial_viscosity_q_cgs,
          py::arg("rho_phys"), py::arg("u_grid"), py::arg("vx"), py::arg("vy"), py::arg("vz"),
          py::arg("dx_cm"), py::arg("a"), py::arg("C2") = 1.5, py::arg("C1") = 0.5,
          py::arg("Ctheta") = 10.0, py::arg("balsara") = true, py::arg("eps") = 1e-30,
          "Monotonic artificial viscosity (returns q, div_v, shear limiter).");
}
