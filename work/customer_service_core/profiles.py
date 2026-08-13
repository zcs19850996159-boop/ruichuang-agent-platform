from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ProfileDefinition:
    name: str
    competition_patches_enabled: bool
    source: str


class CompetitionPatchRegistry:
    """Single compatibility boundary for competition-only behavior.

    Phase 1 keeps legacy behavior enabled in the competition profile. Rules are
    migrated here one at a time, with a regression run after every migration.
    """

    SUPPORTED = {"default", "competition", "enterprise"}

    def resolve(self, profile: str) -> ProfileDefinition:
        name = profile if profile in self.SUPPORTED else "default"
        return ProfileDefinition(
            name=name,
            competition_patches_enabled=name == "competition",
            source="legacy_compatibility_adapter" if name == "competition" else "profile_configuration",
        )

    def status(self, profile: str) -> dict[str, object]:
        resolved = self.resolve(profile)
        return {
            "profile": resolved.name,
            "competition_patches_enabled": resolved.competition_patches_enabled,
            "source": resolved.source,
        }
