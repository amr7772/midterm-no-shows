"""Tests for the Appointment class."""

import math
from dataclasses import FrozenInstanceError

import pytest

from noshows.appointment import Appointment

VALID = {
    "lead_days": 10,
    "age": 40,
    "prior_no_shows": 0,
    "hour": 9,
    "weekday": "mon",
    "booking": "online",
    "insurance": "private",
    "treatment": "physio",
    "first_visit": 1,
    "distance_km": 5.0,
}

COLUMNS = [
    "lead_days",
    "age",
    "prior_no_shows",
    "hour",
    "weekday",
    "booking",
    "insurance",
    "treatment",
    "first_visit",
    "distance_km",
]


def make(**changes):
    """Build an Appointment from the valid example, changing some fields."""
    return Appointment(**{**VALID, **changes})


def test_valid_booking_is_accepted():
    appointment = make()
    assert appointment.age == 40


def test_missing_distance_is_allowed():
    appointment = make(distance_km=None)
    assert appointment.distance_km is None


def test_appointment_cannot_be_changed():
    appointment = make()
    with pytest.raises(FrozenInstanceError):
        appointment.age = 50


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("lead_days", -1),
        ("lead_days", 121),
        ("lead_days", "ten"),
        ("age", 11),
        ("age", 111),
        ("prior_no_shows", -1),
        ("prior_no_shows", 31),
        ("hour", 7),
        ("hour", 18),
        ("first_visit", 2),
        ("distance_km", 0),
        ("distance_km", 151),
    ],
)
def test_impossible_numbers_are_rejected_with_field_name(field, value):
    with pytest.raises(ValueError, match=field):
        make(**{field: value})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("weekday", "sat"),
        ("weekday", "Mon"),
        ("booking", "walk_in"),
        ("insurance", "none"),
        ("treatment", "yoga"),
    ],
)
def test_unseen_categories_are_rejected_with_field_name(field, value):
    with pytest.raises(ValueError, match=field):
        make(**{field: value})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("lead_days", 0),
        ("lead_days", 120),
        ("age", 12),
        ("age", 110),
        ("prior_no_shows", 30),
        ("hour", 8),
        ("hour", 17),
        ("distance_km", 0.1),
        ("distance_km", 150),
    ],
)
def test_limits_themselves_are_accepted(field, value):
    appointment = make(**{field: value})
    assert getattr(appointment, field) == value


def test_to_frame_has_one_row_and_training_columns():
    frame = make().to_frame()
    assert frame.shape == (1, 10)
    assert list(frame.columns) == COLUMNS


def test_to_frame_turns_missing_distance_into_nan():
    frame = make(distance_km=None).to_frame()
    assert math.isnan(frame.loc[0, "distance_km"])
