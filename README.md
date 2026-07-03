# Indoor Run Analysis

Upload `.fit` files from treadmill runs to analyze heart rate, speed, cadence, and steady-state metrics.

## Features

- **Upload .fit files** — Drag-and-drop or click to upload
- **Session management** — Toggle inclusion, delete individual or batch
- **Filtering** — Filter by speed, duration, date, and analysis window
- **Charts** — HR, Speed, Cadence, Distance over time; HR distribution & zones
- **Steady-State Analysis** — Block-averaged HR, HR drift over 5-min blocks
- **Leaderboards** — Compare steady-state HR and cadence across sessions

## Quick Start

```bash
pip install -r requirements.txt
python app.py
```

Open http://localhost:5000

## Docker

```bash
docker compose up
```

## Stack

- **Backend**: Flask + SQLite + fitparse + numpy
- **Frontend**: Vanilla JS + Chart.js
