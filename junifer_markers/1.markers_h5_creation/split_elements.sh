#!/bin/bash
# Split elements.csv into state, evoked, and sleep files

# Always resolve paths relative to this script's directory so that the script
# works regardless of the current working directory from which it is called.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ELEMENTS_FILE="${SCRIPT_DIR}/elements.csv"
ELEMENTS_STATE="${SCRIPT_DIR}/elements_state.csv"
ELEMENTS_EVOKED="${SCRIPT_DIR}/elements_evoked.csv"
ELEMENTS_SLEEP="${SCRIPT_DIR}/elements_sleep.csv"

# Check if elements.csv exists
if [ ! -f "$ELEMENTS_FILE" ]; then
    echo "ERROR: $ELEMENTS_FILE not found"
    exit 1
fi

# Get header
head -n 1 "$ELEMENTS_FILE" > "$ELEMENTS_STATE"
head -n 1 "$ELEMENTS_FILE" > "$ELEMENTS_EVOKED"
head -n 1 "$ELEMENTS_FILE" > "$ELEMENTS_SLEEP"

# Filter state elements (desc column = state)
awk -F',' 'NR>1 && $3=="state"' "$ELEMENTS_FILE" >> "$ELEMENTS_STATE"

# Filter evoked elements (desc column = evoked)
awk -F',' 'NR>1 && $3=="evoked"' "$ELEMENTS_FILE" >> "$ELEMENTS_EVOKED"

# Filter sleep elements (desc column = sleep)
awk -F',' 'NR>1 && $3=="sleep"' "$ELEMENTS_FILE" >> "$ELEMENTS_SLEEP"

# Count results
STATE_COUNT=$(tail -n +2 "$ELEMENTS_STATE" | wc -l | tr -d ' ')
EVOKED_COUNT=$(tail -n +2 "$ELEMENTS_EVOKED" | wc -l | tr -d ' ')
SLEEP_COUNT=$(tail -n +2 "$ELEMENTS_SLEEP" | wc -l | tr -d ' ')

echo "Split complete:"
echo "  State elements:  $STATE_COUNT -> $ELEMENTS_STATE"
echo "  Evoked elements: $EVOKED_COUNT -> $ELEMENTS_EVOKED"
echo "  Sleep elements:  $SLEEP_COUNT -> $ELEMENTS_SLEEP"
