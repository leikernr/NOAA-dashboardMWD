# 📤 MWD Config File Upload Guide

## Overview
The MWD Pulse Simulator now supports uploading your own MWD configuration files (.mwd) for analysis and validation. This feature helps you:
- Validate your config file before deployment
- Identify potential issues or warnings
- Visualize telemetry sequences
- Test pulse patterns with real parameters

## How to Use

### 1. Upload Your Config File
In the sidebar under "🔧 Advanced MWD Pulse Simulator":
- Click **"📤 Upload MWD Config File (.mwd)"**
- Select your .mwd or .txt config file
- The file will be parsed automatically

### 2. Review Configuration Analysis
After upload, an expandable section appears showing:

#### ✅ Validation Status
- **Critical Issues (❌)**: Problems that must be fixed
- **Warnings (⚠️)**: Non-critical issues to review
- **Success (✅)**: File is valid

#### 📋 Parsed Information
- **Survey Sequences (SSq1-6)**: Directional drilling parameters
- **Tool Sequences (TSq1-6)**: Formation evaluation parameters
- **Pulse Widths (PW1-6)**: Timing parameters
- **Telemetry Parameters**: TxDT, FEvT, BThr
- **Alert Thresholds**: Shock and vibration levels

### 3. Simulator Adjusts Automatically
When a valid config is uploaded:
- **Pulse Width**: Uses average from PW1-6 values
- **Sequences**: Shows actual parameters from your config
- **Thresholds**: Displays your configured alert levels
- **Telemetry**: Reflects your transmission dead time

### 4. Default Behavior (No Upload)
If no file is uploaded, the simulator uses:
- Default Survey Sequence (SSq1)
- Default Tool Sequence (TSq1)
- Standard pulse width: 0.60 seconds
- Typical MWD parameters

## Config File Format

Your .mwd file should contain lines like:

```
SSq1: "Inc:12:P Azm:12:P DipA:12:P Temp:8:P Grav:12:P MagF:12:P BatV:8:P"
TSq1: "3{2{aTFA:6:P} Gama:8:P aTFA:6:P Gama:8:P} RINC:11:P RAZM:11:P..."
PW1: 0.6
PW2: 0.6
TxDT: 45
FEvT: 5.00
BThr: 21.00
sLvl1: 40
sLvl2: 60
sLvl3: 80
vLvl1: 5
vLvl2: 10
vLvl3: 15
```

## Validation Checks

### Critical Issues
- **No sequences found**: File must contain SSq or TSq definitions
- **Invalid numeric values**: PW, TxDT, etc. must be numbers
- **Parse errors**: File format is unreadable

### Warnings
- **Missing critical parameters**: SSq should have Inc/Azm/Temp
- **No pulse widths**: Will use defaults
- **Unusual values**: PW outside 0.6-1.0s range

## What Gets Analyzed

### Survey Sequences (SSq)
- **Inc**: Inclination (bits:encoding)
- **Azm**: Azimuth
- **DipA**: Dip Angle
- **Temp**: Temperature
- **Grav**: Gravity
- **MagF**: Magnetic Field
- **BatV**: Battery Voltage

### Tool Sequences (TSq)
- **aTFA**: Azimuth Tool Face Angle
- **Gama**: Gamma Ray
- **RINC**: Rate of Inclination
- **RAZM**: Rate of Azimuth
- **LRMS**: Lateral Shock RMS
- **ARMS**: Axial Shock RMS
- **LMxG**: Lateral Max G
- **AMxG**: Axial Max G

### Pulse Parameters
- **PW1-PW6**: Pulse widths in seconds
- **TxDT**: Transmission dead time
- **FEvT**: Flow event time
- **BThr**: Bit threshold

### Thresholds
- **sLvl1-3**: Shock levels in G
- **vLvl1-3**: Vibration levels in GRMS

## Example: Analyzing Your Config

Let's say you upload `BT_Al_San_Roman_VCL_STD_GAMMA_1-18-26.mwd`:

### ✅ Results You'll See:
```
✅ Configuration file validated successfully!

Survey Sequences: 6
Tool Sequences: 6
Total Sequences: 12
Avg Pulse Width: 0.73s
Tx Dead Time: 45s

Shock Levels:
  sLvl1: 40G
  sLvl2: 60G
  sLvl3: 80G

Vibration Levels:
  vLvl1: 5 GRMS
  vLvl2: 10 GRMS
  vLvl3: 15 GRMS
```

### 📊 Simulator Updates:
- Pressure waveforms use 0.73s pulses
- Binary encoding shows your exact bit allocations
- Telemetry timing reflects 45s dead time
- Alert thresholds match your configuration

## Common Issues & Solutions

### Issue: "No sequences found"
**Cause**: File doesn't contain SSq or TSq lines
**Solution**: Ensure file has lines starting with `SSq1:` or `TSq1:`

### Issue: "Invalid numeric value"
**Cause**: Non-numeric data in PW, TxDT, or threshold fields
**Solution**: Check that numbers don't have units or extra text

### Issue: "Unusual pulse width"
**Cause**: PW value < 0.1s or > 2.0s
**Solution**: Typical MWD uses 0.6-1.0s. Verify your value is correct.

### Issue: "Missing critical parameters"
**Cause**: Survey sequence doesn't include Inc, Azm, or Temp
**Solution**: Most drilling needs these. Add if missing.

## Benefits of Validation

### Before Deployment:
✅ Catch configuration errors early
✅ Verify parameter ranges
✅ Ensure all required sequences exist
✅ Check threshold consistency

### During Operations:
✅ Understand expected telemetry behavior
✅ Visualize pulse patterns
✅ Preview data rates and frame timing
✅ Compare actual vs. configured performance

## Advanced Features

### Sequence Complexity Metrics
The analyzer calculates:
- Total number of sequences
- Average pulse width across PW1-6
- Estimated frame duration
- Data rate implications

### Real-Time Visualization
Once validated, your config drives:
- Pressure pulse waveforms
- Binary encoding patterns
- Surface decoding simulation
- Live telemetry streams

## Tips for Best Results

1. **Upload Early**: Check your config before rig-up
2. **Review Warnings**: Even non-critical issues matter
3. **Compare Sequences**: Use SSq for directional, TSq for logging
4. **Test Thresholds**: Ensure shock/vibe levels are appropriate
5. **Monitor Frame Size**: Large sequences = longer transmission time

## Integration with MWD Logging

If you also upload:
- **Survey LAS file**: Compare actual vs. configured inclination/azimuth
- **MWD logging data**: Cross-reference gamma, shock, vibration with thresholds

This provides complete validation of your MWD system!

---

**Note**: This feature analyzes configuration files only. It does not modify your files or deploy settings to actual tools. Always validate configurations with your MWD service provider before field operations.
