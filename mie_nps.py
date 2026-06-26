import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.special import spherical_jn, spherical_yn
from tkinter import Tk
from tkinter.filedialog import askopenfilename

#parameters

sizes_nm = [23, 30, 37, 43, 60]   # radius (nm)
n_medium = 1.333
nmax = 10

#load_data

# Open file dialog in the script directory
script_dir = os.path.dirname(os.path.abspath(__file__))

Tk().withdraw()

file_path = askopenfilename(
    title="Select a text file",
    initialdir=script_dir,
    filetypes=[
        ("Text files", "*.txt"),
        ("All files", "*.*")
    ]
)

data = np.loadtxt(file_path)

print("Selected file:", file_path)
print(data)

# Akan terbuka File Explorer kemudian pilih Permittivity_Gold_JohnsonChristy

energy_raw = data[:,0]
eps1_raw = data[:,1]
eps2_raw = data[:,2]

##constants

e_charge = 1.60217646e-19
h_planck = 6.626068e-34
c_light = 2.99792458e8

#interpolation

energy = np.linspace(
    energy_raw.min(),
    energy_raw.max(),
    1000
)

cs1 = CubicSpline(energy_raw, eps1_raw)
cs2 = CubicSpline(energy_raw, eps2_raw)

eps1 = cs1(energy)
eps2 = cs2(energy)

#energy-to_wavelength

wavelength_m = h_planck * c_light / (e_charge * energy)
wavelength_nm = wavelength_m * 1e9

# sort wavelength
sort_idx = np.argsort(wavelength_nm)
wavelength_nm = wavelength_nm[sort_idx]
wavelength_m = wavelength_m[sort_idx]

#refractive_index

abs_eps = np.sqrt(eps1**2 + eps2**2)

n_particle = np.sqrt((abs_eps + eps1)/2)
k_particle = np.sqrt((abs_eps - eps1)/2)

m = (n_particle + 1j*k_particle) / n_medium
m = m[sort_idx]

#plotting

plt.figure(figsize=(9,6))

for radius_nm in sizes_nm:

    radius = radius_nm * 1e-9

    #wavevector
    k = 2 * np.pi * n_medium / wavelength_m

    #size parameter
    x = k * radius
    mx = m * x

    #initialize
    scaele = np.zeros(len(wavelength_nm), dtype=complex)
    extele = np.zeros(len(wavelength_nm))

    #mie_calculation

    for n in range(1, nmax + 1):

        jnx = spherical_jn(n, x)
        jnminx = spherical_jn(n-1, x)

        hnx = spherical_jn(n, x) + 1j*spherical_yn(n, x)
        hnminx = spherical_jn(n-1, x) + 1j*spherical_yn(n-1, x)

        jnmx = spherical_jn(n, mx)
        jnminmx = spherical_jn(n-1, mx)

        xjnxdiff = x * jnminx - n * jnx
        mxjnmxdiff = mx * jnminmx - n * jnmx
        xhnxdiff = x * hnminx - n * hnx

        an = (
            m**2 * jnmx * xjnxdiff -
            jnx * mxjnmxdiff
        ) / (
            m**2 * jnmx * xhnxdiff -
            hnx * mxjnmxdiff
        )

        bn = (
            jnmx * xjnxdiff -
            jnx * mxjnmxdiff
        ) / (
            jnmx * xhnxdiff -
            hnx * mxjnmxdiff
        )

        scaele += (2*n + 1) * (
            np.abs(an)**2 +
            np.abs(bn)**2
        )

        extele += (2*n + 1) * np.real(an + bn)

    #extinction

    Cext = 2 * np.pi / k**2 * extele

    #peak
    peak_idx = np.argmax(Cext)
    peak_wl = wavelength_nm[peak_idx]
    peak_val = Cext[peak_idx]

    print(f"{radius_nm} nm → Peak = {peak_wl:.1f} nm")

    #plot_spectrum
    plt.plot(
        wavelength_nm,
        Cext,
        linewidth=2,
        label=f'{radius_nm} nm'
    )

    #peak_marker
    plt.scatter(
        peak_wl,
        peak_val,
        s=40
    )

    #peak_label
    plt.text(
        peak_wl + 5,
        peak_val,
        f'{peak_wl:.0f} nm',
        fontsize=9
    )

plt.title('Au Nanosphere Extinction Spectra')

plt.xlabel('Wavelength (nm)')
plt.ylabel('Extinction Cross-section (m²)')

plt.xlim(400, 900)

# ==========================================
# DATA
# ==========================================

size = [14, 19, 27, 30, 33]

lspr_peaks = [
    522.3,
    525.0,
    531.7,
    535.7,
    538.5
]

# warna mengikuti grafik extinction
colors = [
    'black',        # 14 nm
    'red',          # 19 nm
    'dodgerblue',   # 27 nm
    'green',        # 30 nm
    'hotpink'       # 33 nm
]

plt.legend()

plt.tight_layout()
# ==========================================
# PLOT
# ==========================================

plt.figure(figsize=(8,6))

for x, y, c in zip(size, lspr_peaks, colors):

    plt.scatter(
        x,
        y,
        color=c,
        s=90
    )

# garis penghubung (opsional)
plt.plot(
    size,
    lspr_peaks,
    color='gray',
    linewidth=1,
    alpha=0.6
)

# ==========================================
# LABELS
# ==========================================

plt.xlabel(
    'Au Core Diameter (nm)',
    fontsize=10
)

plt.ylabel(
    'LSPR Peak Position (nm)',
    fontsize=10
)

# ==========================================
# AXIS
# ==========================================

plt.xlim(12, 35)
plt.xticks(size)

plt.ylim(520, 540)
plt.yticks(np.arange(520, 541, 5))

# ==========================================
# STYLE
# ==========================================

ax = plt.gca()

plt.tight_layout()
plt.show()
