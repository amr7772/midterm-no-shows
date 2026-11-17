"""One booking, as the front desk enters it."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd
LEAD_DAYS_RANGE = (0, 120)
AGE_RANGE = (12, 110)
PRIOR_NO_SHOWS_RANGE = (0, 30)
HOUR_RANGE = (8, 17)
DISTANCE_KM_RANGE = (0.1, 150)

WEEKDAYS = ("mon", "tue", "wed", "thu", "fri")
BOOKINGS = ("online", "phone", "front_desk")
INSURANCES = ("private", "provincial", "self_pay")
TREATMENTS = ("physio", "massage", "acupuncture")


def _check_number(name: str, value: object, low: float, high: float) -> None:
    """Raise ValueError, naming the field, if value is not a number in range."""
    is_number = isinstance(value, int | float) and not isinstance(value, bool)
    if not is_number or not low <= value <= high:
        raise ValueError(
            f"{name} must be a number from {low} to {high}, got {value!r}"
        )


def _check_choice(name: str, value: object, allowed: tuple[str, ...]) -> None:
    """Raise ValueError, naming the field, if value is not an allowed choice."""
    if value not in allowed:
        options = ", ".join(allowed)
        raise ValueError(f"{name} must be one of {options}, got {value!r}")


@dataclass(frozen=True)
class Appointment:
    """One appointment. Frozen, so it cannot be changed after creation."""

    lead_days: int
    age: int
    prior_no_shows: int
    hour: int
    weekday: str
    booking: str
    insurance: str
    treatment: str
    first_visit: int
    distance_km: float | None = None

    def __post_init__(self) -> None:
        _check_number("lead_days", self.lead_days, *LEAD_DAYS_RANGE)
        _check_number("age", self.age, *AGE_RANGE)
        _check_number("prior_no_shows", self.prior_no_shows, *PRIOR_NO_SHOWS_RANGE)
        _check_number("hour", self.hour, *HOUR_RANGE)
        if self.first_visit not in (0, 1):
            raise ValueError(f"first_visit must be 0 or 1, got {self.first_visit!r}")
        if self.distance_km is not None:
            _check_number("distance_km", self.distance_km, *DISTANCE_KM_RANGE)
        _check_choice("weekday", self.weekday, WEEKDAYS)
        _check_choice("booking", self.booking, BOOKINGS)
        _check_choice("insurance", self.insurance, INSURANCES)
        _check_choice("treatment", self.treatment, TREATMENTS)


    def to_frame(self) -> pd.DataFrame:
        """Return this booking as a one-row table with the training columns."""
        row = asdict(self)
        if row["distance_km"] is None:
            row["distance_km"] = float("nan")
        return pd.DataFrame([row])