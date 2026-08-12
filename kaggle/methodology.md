# AtmosIQ Dataset v2 Technical Methodology

## 1. Ground Air Quality Aggregation
Station-level PM2.5 readings from Delhi CPCB monitoring stations (ITO, Anand Vihar, RK Puram, Punjabi Bagh, Mandir Marg) were aggregated into daily city-wide regional averages.

## 2. Satellite Fire Exposure Proxies
Regional upwind satellite active fire hotspots across Punjab, Haryana, Rajasthan, and Delhi NCR are processed into daily hotspot sums and wind-weighted transport exposure scores.

## 3. Strict Leakage Protection
Target-derived rolling statistics use $t-1 \dots t-k$ historical windows exclusively.
