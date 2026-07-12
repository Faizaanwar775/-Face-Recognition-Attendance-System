"""Unit tests for src.matching.face_matcher.FaceMatcher."""

from __future__ import annotations

from src.matching.face_matcher import FaceMatcher
from src.storage.models import EmployeeRecord, FaceEmbedding


def test_identical_embedding_is_a_match() -> None:
    matcher = FaceMatcher(distance_threshold=0.55)
    vector = [0.1] * 128
    enrolled = [EmployeeRecord(employee_id="E1", full_name="Alice", embedding=vector)]

    result = matcher.match(FaceEmbedding(vector=vector), enrolled)

    assert result.is_match is True
    assert result.employee_id == "E1"
    assert result.distance == 0.0


def test_very_different_embedding_is_rejected() -> None:
    matcher = FaceMatcher(distance_threshold=0.55)
    enrolled = [EmployeeRecord(employee_id="E1", full_name="Alice", embedding=[0.0] * 128)]

    far_vector = [10.0] * 128
    result = matcher.match(FaceEmbedding(vector=far_vector), enrolled)

    assert result.is_match is False
    assert result.employee_id is None


def test_no_enrolled_employees_never_matches() -> None:
    matcher = FaceMatcher()
    result = matcher.match(FaceEmbedding(vector=[0.1] * 128), [])
    assert result.is_match is False
    assert result.confidence == 0.0


def test_best_of_multiple_candidates_is_chosen() -> None:
    matcher = FaceMatcher(distance_threshold=1.0)
    enrolled = [
        EmployeeRecord(employee_id="FAR", full_name="Far Person", embedding=[5.0] * 128),
        EmployeeRecord(employee_id="CLOSE", full_name="Close Person", embedding=[0.05] * 128),
    ]
    result = matcher.match(FaceEmbedding(vector=[0.0] * 128), enrolled)
    assert result.employee_id == "CLOSE"
