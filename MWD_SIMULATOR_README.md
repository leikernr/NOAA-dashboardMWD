# 🛢️ Advanced MWD Pulse Telemetry Simulator

## Overview
This enhanced MWD (Measurement While Drilling) simulator uses **real configuration data** from uploaded MWD tool configuration files to create an authentic mud pulse telemetry visualization.

## Features Implemented

### 1. **Real MWD Sequences**
Based on your actual config file:

#### Survey Sequence (SSq1)
- **Inc**: 12-bit Inclination with Parity
- **Azm**: 12-bit Azimuth with Parity
- **DipA**: 12-bit Dip Angle with Parity
- **Temp**: 8-bit Temperature with Parity
- **Grav**: 12-bit Gravity with Parity
- **MagF**: 12-bit Magnetic Field with Parity
- **BatV**: 8-bit Battery Voltage with Parity

#### Tool Sequence (TSq1)
- **aTFA**: 6-bit Azimuth Tool Face Angle with Parity
- **Gamma**: 8-bit Gamma Ray with Parity
- **RINC**: 11-bit Rate of Inclination Change
- **RAZM**: 11-bit Rate of Azimuth Change
- **LRMS**: 11-bit Lateral Shock RMS
- **ARMS**: 11-bit Axial Shock RMS
- **LMxG**: 11-bit Lateral Max G
- **AMxG**: 11-bit Axial Max G

### 2. **Three Pulse Types**
- **Positive Pulse**: Valve restriction increases standpipe pressure (most common)
- **Negative Pulse**: Valve opening decreases pressure (faster data rate)
- **Continuous Wave**: Phase-shift keying using oscillating valve

### 3. **Realistic Parameters**
From your config file:
- Pulse widths: 0.6 - 1.0 seconds (PW1-PW6)
- Data rates: 1-12 bits per second
- Standpipe pressure: 2000-5000 psi
- Pressure change: 100-800 psi
- Mud flow rate: 200-800 gpm

### 4. **Binary Encoding Visualization**
- Displays current sensor values
- Shows bit allocation and parity
- Calculates frame duration
- Visualizes binary stream with sync pattern

### 5. **Pressure Waveform**
- Real-time pressure signal generation
- Sync pattern highlighting (green region)
- Baseline pressure reference
- Realistic noise and signal attenuation
- Signal strength effects on quality

### 6. **Surface Decoding Process**
Real-time metrics:
- **Sync Status**: Frame synchronization lock
- **Bit Error Rate**: Quality metric based on signal strength
- **Parity Check**: Data integrity validation
- **Frame Status**: Overall frame validity

### 7. **Live Telemetry Stream**
- Animated 10-second streaming window
- Continuous pulse train visualization
- Real-time frame counter
- Start/stop controls

## Technical Details

### Encoding Process
1. **Sensor Reading** → Analog values from downhole sensors
2. **Digital Conversion** → Binary encoding with specified bit precision
3. **Parity Addition** → Error detection bit added to each parameter
4. **Sync Pattern** → 8-bit sync sequence (11001100) for frame alignment
5. **Pulse Generation** → Binary stream modulates mud pressure
6. **Transmission** → Pressure pulses travel through drilling fluid
7. **Surface Detection** → Standpipe transducer measures pressure changes
8. **Decoding** → Binary stream reconstructed and converted back to values

### Signal Quality Factors
- **Mud Flow Rate**: Higher flow = better signal transmission
- **Signal Strength**: Accounts for depth, mud properties, vibration
- **Noise**: Realistic pump noise and mechanical vibration
- **Attenuation**: Signal degradation over distance

## How to Use

### In Sidebar:
1. **Select Pulse Type**: Choose positive, negative, or continuous wave
2. **Select Sequence**: Survey (directional) or Tool (formation evaluation)
3. **Adjust Pulse Width**: 0.15-1.0 seconds (default 0.60s from config)
4. **Set Data Rate**: 1-12 bps (typical MWD range)
5. **Configure Drilling Parameters**: Mud flow, standpipe pressure, etc.

### In Main Dashboard:
1. **View Sensor Values**: See current readings with bit precision
2. **Monitor Telemetry Settings**: Frame size, duration, and timing
3. **Analyze Pressure Waveform**: Visualize actual mud pressure pulses
4. **Check Binary Stream**: Inspect sync pattern and data bits
5. **Monitor Decoding**: Real-time quality metrics
6. **Watch Live Stream**: Start animation to see continuous telemetry

## Real-World Applications

### What This Simulates:
- **Directional Drilling**: Real-time inclination/azimuth for wellbore positioning
- **Formation Evaluation**: Gamma ray for lithology identification
- **Tool Performance**: Shock/vibration monitoring for BHA health
- **Safety Monitoring**: Temperature and pressure for kick detection

### Why It Matters:
- **Cost**: MWD telemetry costs $10k-$50k per day
- **Speed**: Faster data = better drilling decisions
- **Safety**: Real-time monitoring prevents costly incidents
- **Precision**: Accurate directional control hits geological targets

## Configuration File Integration

Your MWD config includes:
```
Survey Sequences: SSq1-SSq6
Tool Sequences: TSq1-TSq6
Pulse Widths: PW1=0.6, PW2=0.6, PW3=0.6, PW4=0.8, PW5=1.0, PW6=1.0
Telemetry: TxDT=45, FEvT=5.00, BThr=21.00
Shock Levels: 40G, 60G, 80G
Vibe Levels: 5 GRMS, 10 GRMS, 15 GRMS
```

These real parameters drive the simulator's accuracy!

## Future Enhancements

Potential additions:
- [ ] Multi-frame sequences showing all SSq1-6 and TSq1-6
- [ ] Downlink command simulation (surface to downhole)
- [ ] Memory dump visualization for logged data
- [ ] Error correction algorithms (Reed-Solomon, etc.)
- [ ] Waveform FFT analysis for signal processing
- [ ] Real-time comparison with uploaded LAS file data

## Technical Notes

### Frame Structure:
```
[SYNC 8-bit] [DATA 64+ bits] [PARITY bits]
```

### Timing:
- Frame duration = Total bits / Data rate
- Example: 80 bits @ 6 bps = 13.3 seconds per frame
- Transmission dead time: 45 seconds (from config)

### Quality Thresholds:
- Signal strength > 70%: Valid frames
- Signal strength > 75%: Parity pass
- Signal strength > 80%: Low bit error rate

---

This simulator demonstrates authentic mud pulse telemetry technology used in offshore drilling operations worldwide! 🌊⛽
