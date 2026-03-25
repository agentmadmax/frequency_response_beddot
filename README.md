# Frequency Response Analysis of Beddot

## Overview
This repository contains the original nominal-bin(fft_response.py) and measured-bin(comboned_code.py) FFT analysis workflow used to estimate the frequency response of a geophone-based sensing system from shaker table data.

The code processes both horizontal and vertical datasets, handles multiple geophone sensors, saves all outputs into a separate `Results` directory, and generates frequency-response plots for each sensor as well as combined comparison plots.

The implemented method is the **nominal-bin FFT method and measured-bin FFT method**.
---

## Repository Structure

```text
FREQ_RESP/
├── Code/
│   └── nominal_fft_multi_sensor.py
    └──combined_code.py
│
├── Data/
│   ├── horizontal/
│   │   ├── 1.55 0.1hz/
│   │   ├── 2.01 0.2hz/
│   │   ├── 2.07 0.5hz/
│   │   ├── 2.09 0.8hz/
│   │   ├── 2.11 1hz/
│   │   ├── 2.14 2hz/
│   │   ├── 2.16 4hz/
│   │   ├── 2.17 8hz/
│   │   └── 2.19 16hz/
│   │
│   └── vertical/
│       ├── 2.27 0.1hz/
│       ├── 2.30 0.2hz/
│       ├── 2.32 0.5hz/
│       ├── 2.34 0.8hz/
│       ├── 2.36 1hz/
│       ├── 2.40 0.6hz/
│       ├── 2.42 0.7hz/
│       ├── 2.46 1.5hz/
│       ├── 2.49 2hz/
│       ├── 2.50 4hz/
│       ├── 2.51 8hz/
│       └── 2.53 16hz/
│
├── Results/
│   ├── horizontal/
│   ├── vertical/
│   └── combined/
│
└── README.md
