# Macroturbulence and rotation-only v sin i in OB stars

Tools for interpreting projected rotational velocities of OB stars measured from low-resolution spectra (e.g. SDSS-V
BOSS, R ≈ 2000) with rotation as the only broadening: how macroturbulence biases them, and what macroturbulence to
expect for a star of given T_eff and L.

## Contents

| Folder | What it is |
|---|---|
| [`notebooks/`](notebooks) | `vsini_macroturbulence.ipynb`: a step-by-step calculation of what a rotation-only fit returns for a star that also has radial-tangential macroturbulence. Built and executed by `build_vsini_macroturbulence.py`. |
| [`vmac_table/`](vmac_table) | 890 Galactic OB stars with measured macroturbulence (IACOB and related studies), with luminosities, masses and evolutionary stages from MIST tracks, and a fit of v_mac as a function of (T_eff, log L). See its [README](vmac_table/README.md). |

## Main results

**1. Macroturbulence adds in quadrature, not linearly.** A rotation-only fit to a line broadened by rotation and
radial-tangential macroturbulence ζ returns

    v sin i_fit ≈ sqrt( (v sin i)² + k² ζ² )

with k ≈ 0.75 at R = 22 500 and k ≈ 0.82–0.88 at R = 2000, for equal radial and tangential fractions. The quadrature
form follows from the exact additivity of the profile variances; least-squares fits give a smaller k than the
variance argument (1.05) because they follow the line core. Subtracting v_mac from the fitted value is wrong: for a
fitted 116 km/s and ζ = 80 km/s it gives 36 km/s instead of the true 100 km/s.

**2. The correction is small at high v sin i and depends on the geometry of the velocity field.** At R = 2000 and
v sin i = 150 km/s, ζ = 80 km/s adds about 14 km/s for equal radial and tangential fractions, 22 km/s for purely
radial and 6 km/s for purely tangential motions. Relative to the IACOB ζ, k ranges from about 0.5 (purely
tangential, as for gravity waves) to about 1.0 (purely radial). A k measured on stars with known v sin i and ζ
therefore also constrains the geometry.

**3. Resolution alone does not make a pile-up.** At R = 2000 and S/N 15–100 the fitted values are unbiased (apart
from the macroturbulence term) with wider scatter. A pile-up of fitted values near ~130 km/s must come from the fitting
pipeline itself and has to be calibrated empirically.

**4. Expected v_mac.** Among Galactic OB stars v_mac rises from ≈ 20–30 km/s at log L/L☉ = 3–4.5 to ≈ 60–80 km/s
above log L/L☉ = 5, with an observed star-to-star scatter (measurement errors included) of 0.09 dex at high
luminosity and ≈ 0.18 dex below log L/L☉ = 4.5.
`vmac_table/vmac_fit_grid.csv` gives the mean relation, its uncertainty and the scatter on a grid, with a sampling
snippet in `vmac_table/README.md`.

## Using it in a forward model of v sin i distributions

1. Draw equatorial velocities from the distribution being tested, and isotropic inclinations.
2. Draw v_mac for each star from `vmac_fit_grid.csv` at its (T_eff, log L), including the scatter.
3. Convert to the value the pipeline would report: √(v sin i² + k² v_mac²) plus measurement noise, and the
   pipeline's own behaviour at low v sin i (from a validation sample).
4. Apply the sample cuts to these reported values and compare with the observed distribution.

## Reproducing

```bash
pip install -r requirements.txt
cd notebooks && python build_vsini_macroturbulence.py      # ~4 min; or open the notebook and run it
cd ../vmac_table && MIST_DIR=/path/to/MIST_v1.2_feh_p0.00_afe_p0.0_vvcrit0.4_EEPS python build_vmac_table.py
```

## Data sources

All stellar parameters come from published tables on VizieR: Simón-Díaz et al. (2017, A&A 597, A22), de Burgos et al.
(2024, A&A 687, A228), Holgado et al. (2018, A&A 613, A65; 2022, A&A 665, A150) and Burssens et al. (2020, A&A 639,
A81). Evolutionary tracks: MIST v1.2 (Choi et al. 2016, ApJ 823, 102; Dotter 2016, ApJS 222, 8). Please cite the
original papers when using the table.

Contact: Matteo Cantiello.
