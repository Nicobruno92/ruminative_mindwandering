# Connectivity Circle Plot Montage Setup

## Channel Layout

The circular connectivity plots use **topographic positioning** based on the standard 10-20 EEG system with 64 channels (BioSemi layout).

### Channel Arrangement

Channels are positioned on the circle according to their **anatomical location** on the scalp:
- **Anterior (top)**: Frontal channels (Fp, AF, F)
- **Posterior (bottom)**: Occipital channels (O, PO)
- **Left hemisphere**: Negative angles (left side of circle)
- **Right hemisphere**: Positive angles (right side of circle)
- **Midline**: Central vertical axis (90° = top, 270° = bottom)

### Angular Positioning (degrees)

The `CHANNEL_ANGLES` dictionary in `plot_connectivity.py` defines the angular position of each channel on the circle. Angles are measured clockwise from the top (90° = Fz position).

**Coverage**: All 64 channels from the config ROI definitions are included.

### ROI Grouping

ROIs are color-coded and grouped with outer arcs:
- **frontal_left** (red): 60°
- **frontal_right** (orange): 120°
- **central_left** (green): 20°
- **central_right** (teal): 160°
- **posterior_left** (blue): 310°
- **posterior_right** (purple): 230°
- **midline** (yellow): 270°

## Validation

Run `test_imports.py` to verify:
```bash
cd Statistics_connectivity
python3 test_imports.py
```

Expected output:
```
✓ Basic imports OK
✓ lmm_connectivity imports OK
✓ plot_connectivity imports OK
✓ All config channels covered
✓ All ROIs covered
All tests passed! Ready to run pipeline.
```

## Notes

- The montage matches the **CACS-64_REF.bvef** montage used in preprocessing
- Channel positions are approximate but follow standard 10-20 topography
- Extra channels (e.g., FCz) in `CHANNEL_ANGLES` won't cause issues
- The `_get_default_angles()` helper handles any unknown channels by spacing them evenly
