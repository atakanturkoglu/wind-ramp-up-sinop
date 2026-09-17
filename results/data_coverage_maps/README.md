# Data coverage maps

One calendar map per station per variable (pressure, sea-level pressure, temperature, precipitation, wind speed, wind direction — OB station additionally has u/v components). Each map is a year × month × hour-of-day grid: light blue = data present, dark red = missing.

These are the actual evidence behind the quality-control numbers in the paper — for example, the Sinop Merkez wind speed map shows dense, scattered short dropouts through 2021, a multi-week outage in early 2022, and a hard stop in mid-2025 (end of the data pull for this study), rather than one clean continuous record. This is exactly the kind of real-world messiness the quality-control step (`src/quality_control/`) has to handle, and why ramp-up detection deliberately runs before cleaning rather than after (see the main [README](../../README.md)).

Folders: `sinop/`, `inceburun/`, `airport/` (MGM stations, 2014–2025), `ob/` (personal station, starts August 2023).
