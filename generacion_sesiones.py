import random
from dataclasses import dataclass
from datetime import datetime, timedelta, time, date, timezone
from typing import List, Tuple, Optional, Dict

# -------------------------
# Config
# -------------------------

@dataclass
class Config:
    seed: int = 43
    start_date: date = date(2026, 5, 15)
    end_date: date = date(2026, 6, 16)


    ejemplar_ids: List[int] = None


    UBI_CLINICA: int = 1
    UBI_BEBEDERO: int = 2
    UBI_ENTRENAMIENTO: int = 3
    UBI_HENERAS: Tuple[int, int, int] = (13, 14, 19)

    tz = timezone.utc

    clinic_horses_per_day: Tuple[int, int] = (0, 3)
    clinic_duration_min: Tuple[int, int] = (15, 30)
    clinic_detections: Tuple[int, int] = (5, 10)
    clinic_window: Tuple[time, time] = (time(7, 30), time(16, 0))

    training_horses_per_day: Tuple[int, int] = (8, 10)
    training_duration_min: Tuple[int, int] = (20, 40)
    training_detections: Tuple[int, int] = (5, 10)
    training_window: Tuple[time, time] = (time(7, 30), time(16, 0))


    water_times_per_day: Tuple[int, int] = (4, 8)
    water_duration_sec: Tuple[int, int] = (40, 120)
    water_detections: Tuple[int, int] = (10, 30)
    water_window: Tuple[time, time] = (time(6, 0), time(0, 59))


    eat_times_per_day: Tuple[int, int] = (3, 6)
    eat_duration_min: Tuple[int, int] = (15, 90)
    eat_detections: Tuple[int, int] = (10, 150)
    eat_window: Tuple[time, time] = (time(6, 0), time(23, 59))


def fmt_dt(dt: datetime) -> str:

    s = dt.strftime("%Y-%m-%d %H:%M:%S%z")
    if s.endswith("+0000"):
        s = s[:-5] + "+00"
    return s


def rand_datetime_in_window(day: date, start_t: time, end_t: time, tz) -> datetime:
    """
    If end_t < start_t, treat as window spanning midnight.
    """
    start_dt = datetime.combine(day, start_t, tzinfo=tz)

    if (end_t.hour, end_t.minute, end_t.second) < (start_t.hour, start_t.minute, start_t.second):
        end_dt = datetime.combine(day + timedelta(days=1), end_t, tzinfo=tz)
    else:
        end_dt = datetime.combine(day, end_t, tzinfo=tz)

    total_seconds = int((end_dt - start_dt).total_seconds())
    offset = random.randint(0, max(total_seconds, 0))
    return start_dt + timedelta(seconds=offset)


def random_detections(det_range: Tuple[int, int], last_value: Optional[int] = None) -> int:

    lo, hi = det_range
    if lo >= hi:
        return lo

    if last_value is not None:
        for _ in range(6):
            v = random.randint(lo, hi)
            if v != last_value:
                return v
    return random.randint(lo, hi)


def generate_rows(cfg: Config) -> List[dict]:
    random.seed(cfg.seed)

    if cfg.ejemplar_ids is None:
        cfg.ejemplar_ids = list(range(200, 224))

    rows = []
    next_id = 1

    last_det: Dict[Tuple[int, int], int] = {}

    def add_event(ejemplar_id: int, ubi_id: int, start: datetime, duration_sec: int, det_range: Tuple[int, int]):
        nonlocal next_id
        end = start + timedelta(seconds=duration_sec)

        key = (ejemplar_id, ubi_id)
        det = random_detections(det_range, last_det.get(key))
        last_det[key] = det

        rows.append({
            "id": next_id,
            "entrada": fmt_dt(start),
            "salida": fmt_dt(end),
            "duracion": duration_sec,
            "detecciones": det,
            "ejemplar_id": ejemplar_id,
            "ubi_id": ubi_id,
        })
        next_id += 1


    day = cfg.start_date
    while day <= cfg.end_date:

        k = random.randint(cfg.clinic_horses_per_day[0], cfg.clinic_horses_per_day[1])
        clinic_horses = random.sample(cfg.ejemplar_ids, k=min(k, len(cfg.ejemplar_ids)))
        for hid in clinic_horses:
            start = rand_datetime_in_window(day, cfg.clinic_window[0], cfg.clinic_window[1], cfg.tz)
            dur_min = random.randint(*cfg.clinic_duration_min)
            add_event(hid, cfg.UBI_CLINICA, start, dur_min * 60, cfg.clinic_detections)


        k = random.randint(cfg.training_horses_per_day[0], cfg.training_horses_per_day[1])
        training_horses = random.sample(cfg.ejemplar_ids, k=min(k, len(cfg.ejemplar_ids)))
        for hid in training_horses:
            start = rand_datetime_in_window(day, cfg.training_window[0], cfg.training_window[1], cfg.tz)
            dur_min = random.randint(*cfg.training_duration_min)
            add_event(hid, cfg.UBI_ENTRENAMIENTO, start, dur_min * 60, cfg.training_detections)


        for hid in cfg.ejemplar_ids:
            n_water = random.randint(*cfg.water_times_per_day)
            for _ in range(n_water):
                start = rand_datetime_in_window(day, cfg.water_window[0], cfg.water_window[1], cfg.tz)
                dur_sec = random.randint(*cfg.water_duration_sec)
                add_event(hid, cfg.UBI_BEBEDERO, start, dur_sec, cfg.water_detections)


            n_eat = random.randint(*cfg.eat_times_per_day)
            for _ in range(n_eat):
                start = rand_datetime_in_window(day, cfg.eat_window[0], cfg.eat_window[1], cfg.tz)
                dur_min = random.randint(*cfg.eat_duration_min)
                ubi = random.choice(cfg.UBI_HENERAS)
                add_event(hid, ubi, start, dur_min * 60, cfg.eat_detections)

        day += timedelta(days=1)

    rows.sort(key=lambda r: (r["entrada"], r["id"]))

    for i, r in enumerate(rows, start=1):
        r["id"] = i

    return rows


def write_csv(rows: List[dict], out_path: str):
    headers = ["id", "entrada", "salida", "duracion", "detecciones", "ejemplar_id", "ubi_id"]
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(";".join(headers) + "\n")
        for r in rows:
            f.write(";".join(str(r[h]) for h in headers) + "\n")


if __name__ == "__main__":
    cfg = Config()
    rows = generate_rows(cfg)
    write_csv(
        rows,
        "registros_sesiones_generados.csv"
    )
    print(f"Generadas {len(rows)} filas -> registros_sesiones_generados.csv")
