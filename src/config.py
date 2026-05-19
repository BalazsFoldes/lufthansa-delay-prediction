import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "aviation_delay.csv")
FIGURES_DIR = os.path.join(BASE_DIR, "outputs", "figures")
RESULTS_DIR = os.path.join(BASE_DIR, "outputs", "results")

# target & date
TARGET = "delay_over_15m"
DATE_COL = "flight_datetime"

# --- Leakage columns ------------------------------------------------------
# These columns are excluded because they are only available AFTER the flight
# has already departed or landed — using them would constitute data leakage.
#
# actual_delay_minutes          → the target itself is derived from this
# actual_gate_out_time_diff     → measured after gate-out, not available pre-flight
# maintenance_closed_after_pushback → post-pushback event
# final_delay_reason            → assigned after the delay occurs; also a near-
#                               perfect predictor (NONE = no delay, else = delayed)
# sched_buffer_mins_latest      → 100% identical to turnaround_minutes (redundant)
#
# ops_delay_prediction_v2 is kept: correlation with target is only 0.12,
# so it is NOT leakage, it is simply a weak feature and will be evaluated.

LEAKAGE_COLS = [
    "actual_delay_minutes",
    "actual_gate_out_time_diff",
    "maintenance_closed_after_pushback",
    "final_delay_reason",
    "sched_buffer_mins_latest",
]

# feature groups
CATEGORICAL_COLS = ["origin", "destination", "aircraft_type"]

NUMERIC_COLS = [
    "scheduled_departure_hour",
    "route_distance",
    "aircraft_age",
    "turnaround_minutes",
    "passenger_load_factor",
    "visibility",
    "wind_speed",
    "precipitation",
    "thunderstorm_flag",
    "airport_congestion_index",
    "runway_utilization",
    "gate_availability",
    "crew_rest_hours",
    "crew_change_last_minute",
    "previous_leg_delay",
    "maintenance_events_last_30d",
    "crew_status_new_FINAL",
    "ops_delay_prediction_v2",
]

# train / test split
# time-based split: we sort by date and use the last TEST_MONTHS as test set.
# random splitting would leak future information into training, forbidden
# for time-series-like flight data.
TEST_MONTHS = 1          # last 1 month held out as test set (May 2024)
VALIDATION_MONTHS = 1    # month before that used as validation (April 2024)

# modelling
RANDOM_STATE = 42
CV_FOLDS = 5             # timeSeriesSplit folds for cross-validation

# class imbalance
# target ratio is ~96% / 4% → models need scale_pos_weight or class_weight
POS_CLASS_RATIO = 24     # approx. 5762 / 238 ≈ 24

# metric of interest
# recall is prioritised over precision:
# missing a real delay (false negative) has higher operational cost than
# a false alarm (false positive).
MAIN_METRIC = "f1"