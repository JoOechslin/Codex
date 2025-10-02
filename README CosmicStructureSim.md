# 🌌 CosmicStructureSim — A Physically Grounded Cosmological Simulation Code

**CosmicStructureSim** is a fully self-consistent cosmological simulation framework that evolves **dark matter and baryons** from the early universe (z ≈ 100) to the present day.  
It implements **gravity**, **hydrodynamics**, and **radiative thermochemistry** to study the formation and evolution of **cosmic structures** — including dark matter halos, gas collapse, and baryonic disk formation — in a ΛCDM cosmology.

---

## ✨ Key Features

- **ΛCDM cosmological evolution**
  - Comoving coordinates and full Hubble expansion
  - Correct Layzer–Irvine energy balance validation
  - Dark energy (Λ) and matter included consistently

- **Hydrodynamics**
  - Finite-volume mesh-based solver for baryons
  - Artificial viscosity with adaptive Balsara limiter
  - Accurate energy-conserving implementation of the PdV term

- **Thermochemistry**
  - Radiative cooling and heating using **Wiersma et al. (2009)** tables  
    (`Net_Cooling` for metal-free and metal-enriched gas)
  - Temperature floor based on the **Jeans criterion**
  - **Self-shielding correction** for dense regions
  - **Compton cooling** active pre-reionization (z > 9)
  - Smooth transition between pre- and post-reionization regimes

- **Metallicity modeling**
  - Physically motivated density–metallicity relation  
    \( Z/Z_\odot = Z_{\text{floor}} + (Z_{\text{max}} - Z_{\text{floor}})\,(n_H/n_0)^\alpha / (1 + (n_H/n_0)^\alpha) \)
  - Enables local enhancement of cooling efficiency in dense gas

- **Numerical accuracy**
  - Mass- and energy-conserving CIC (cloud-in-cell) deposition and gathering
  - Fully verified **Layzer–Irvine energy conservation**
  - Adaptive timestep control based on cooling, dynamical, and CFL limits
  - Parallelized OpenMP kernels for high performance

---

## 🧩 Physical Modules

| Module | Description |
|--------:|:-------------|
| **Gravity** | Poisson solver for total matter density field |
| **Hydro** | Baryonic momentum and energy update on comoving grid |
| **Cooling** | Wiersma-based Λ(T, nH, Z, z) interpolator + Compton and UV heating |
| **Viscosity** | Artificial viscosity (C1, C2, Cθ, Balsara limiter) with shock control |
| **Metallicity** | Density-based Z model mimicking chemical enrichment |
| **Diagnostics** | Halo finder, virial parameters, spin, temperature, overdensity, LI test |

---

## 🪐 Scientific Goals

- Reproduce large-scale **structure formation** from primordial density fluctuations  
- Investigate **baryonic collapse** and **disk formation** inside dark matter halos  
- Study **cooling, angular momentum, and thermal feedback** in galaxy-scale structures  
- Benchmark thermodynamics and dynamics against the **Layzer–Irvine equation**

---

## ⚙️ Requirements

- **C++17** (core simulation)
- **OpenMP** (for parallelization)
- **Python 3.10+** (for visualization and post-processing)
- **NumPy / Matplotlib / CAMB** (for analysis)

---

## 🚀 Running the Simulation

1. **Copy all files on one directory**  
   Open the main Jupyter notebook called DFS-v11-0.ipynb. The grad_phi.cpython-312-darwin.so files as well as the 2 .npz files containing the cooling tables
   need to sit in the same directory such that the notebook finds these files.
   The .so file is compiled for a macOS environment. If you are running things on a different platform, you need to compile
   the grad_phi-v2.cpp sorce file to get your .so file.

2. **Run**  
   Open and run the DFS-v11-0.ipynb notebook. You may need to install certain modules (eg CAMB) for the notebook to run.

3. **Visualize**  
   Use the provided Python notebooks in to inspect:
   - Halo growth and temperature maps  
   - Press–Schechter mass function  
   - Disk structure and baryonic spin evolution  
   - Layzer–Irvine energy conservation diagnostics

---

## 📈 Example Results

- Formation of virialized halos with realistic baryon fractions  
- Cooling-driven baryonic collapse and disk-like gas distributions  
- Evolution of temperature–density phase diagrams from z = 100 → 0  
- Verified Layzer–Irvine energy conservation throughout cosmic time

---

## 🧠 References

- Wiersma, Schaye & Smith (2009), *MNRAS*, 393, 99

---

## 🧩 Author & License

Developed by **Jo Oechslin**, 2025  
Licensed under the **MIT License**

---

## 🌠 Future Work

- Implement explicit **star formation and feedback models**
- Add **radiative transfer / UV background self-consistency**
- Extend **magnetohydrodynamics (MHD)** module
- Improve **halo finder** and add **galaxy morphology diagnostics**
