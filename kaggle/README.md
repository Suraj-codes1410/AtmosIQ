# AtmosIQ: Delhi NCR Daily PM2.5 Air Quality Dataset (2020–2024)

## Overview
AtmosIQ is an explainable AI and atmospheric intelligence platform for Delhi NCR air quality forecasting.
This dataset contains **1,827 continuous daily observations** spanning 5 complete calendar years (January 1, 2020 to December 31, 2024).

## Data Sources & Provenance
- **Ground Air Quality**: CPCB / OpenAQ Delhi station network
- **Meteorology**: Open-Meteo ERA5 historical reanalysis
- **Satellite Biomass Fires**: NASA FIRMS MODIS/VIIRS active fire hotspots
- **Calendar & Seasonality**: Festival proximity (Diwali) and agricultural stubble burning windows

## Leakage Prevention
For 24-hour forecast models ($t-1 ightarrow t$), all historical lag and rolling statistics are strictly shifted by $\ge 1$ day.

## License
Creative Commons Attribution 4.0 International (CC-BY-4.0).
