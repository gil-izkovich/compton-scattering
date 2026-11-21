# Compton Scattering

Analysis code and data for Compton scattering lab experiment.

We measured the energy of 662 keV gamma rays from a Cs-137 source after they scattered at different angles, using a NaI(Tl) detector. In Part A the target was an aluminum rod. In Part B the target was a plastic scintillator, and the NaI(Tl) spectrum was recorded only in coincidence with a pulse in the plastic. From the scattered photon energy as a function of angle we extracted the electron rest energy, and we compared the measured angular distribution with the Thomson and Klein-Nishina cross sections.

## Requirements

Python 3 with numpy, scipy, matplotlib, pandas and openpyxl.

Run every script from this folder, for example:

```
python angle_calibration_fit.py
```

Plots are saved under `plots/`.

## Data

| Folder / file | Content |
|---|---|
| `deg_cal_files/` | Spectra of the zero-angle scan (direct beam, no target), one file per angle on the board |
| `source_cal_files/` | Spectra of the calibration sources (Ba-133, Co-57, Na-22, Cs-137) |
| `background_files/` | Room background spectrum, subtracted from the calibration spectra |
| `scattering_al_files/` | Scattered spectra, Part A (aluminum rod), one file per angle |
| `scattering_plastic_files/` | Scattered spectra, Part B (plastic scintillator in coincidence), one file per angle |
| `angle_calibration.xlsx` | Peak area and live time for each angle of the zero-angle scan |
| `energy_calibration.xlsx` | Peak channel of each calibration line and its known energy |
| `efficiency_calibration.xlsx` | Peak area, live time and source data (activity, intensity, half-life) for each calibration line, and the geometry (distance, detector radius) |
| `scattering_al_results.csv`, `scattering_plastic_results.csv` | Results of the scattered peak fits (made by `scattering_peak_fits.py`) |

## Scripts

| Script | What it does | Input | Output | In the paper |
|---|---|---|---|---|
| `peak_fit.py` | Fits one photopeak with a Gaussian plus a linear background and prints the center, width and area. Set `FILE`, `LOW` and `HIGH` at the top. | one `.mca` file | `plots/peak_fits/<folder>/` | Fig. 10, Table VII |
| `angle_calibration_fit.py` | Fits the count rate vs. board angle with a Gaussian plus a constant to find the true zero angle θ₀ | `angle_calibration.xlsx` | `plots/calibration/angle_calibration_fit.png` | Fig. 4, Table I |
| `energy_calibration_fit.py` | Linear fit of energy vs. channel | `energy_calibration.xlsx` | `plots/calibration/energy_calibration_fit.png` | Fig. 5, Table II |
| `efficiency_calibration_fit.py` | Photopeak efficiency of each calibration line and a quadratic fit of ln ε vs. ln E | `efficiency_calibration.xlsx` | `plots/calibration/efficiency_calibration_fit.png` | Fig. 6, Table III |
| `scattering_peak_fits.py` | Fits the scattered peak in every spectrum of one part (`PART = 1` or `2` at the top) and converts its center to energy | `scattering_al_files/` or `scattering_plastic_files/` | `scattering_al_results.csv` or `scattering_plastic_results.csv`, `plots/peak_fits/<folder>/` | |
| `energy_angle_combined.py` | Linear fit of 1/E′ vs. 1 − cos θ for both parts, gives mₑc² and E | both results files | `plots/scattering/energy_angle_fit_combined.png` | Fig. 7, Table IV |
| `cross_section_fit.py` | Count rate corrected for solid angle and efficiency vs. angle, fitted with the Thomson and Klein-Nishina cross sections (`PART = 1` or `2` at the top) | results file of the chosen part | `plots/scattering/cross_section_fit.png` or `cross_section_fit_plastic.png` | Figs. 8-9, Tables V-VI |

## Order

1. `peak_fit.py` on the calibration spectra, then copy the values into the Excel files.
2. `angle_calibration_fit.py`, `energy_calibration_fit.py`, `efficiency_calibration_fit.py`.
3. `scattering_peak_fits.py` with `PART = 1` and with `PART = 2`.
4. `energy_angle_combined.py` and `cross_section_fit.py`.
