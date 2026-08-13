from __future__ import annotations

import os
from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class ReleaseVersions:
    application_version: str
    knowledge_version: str
    model_configuration_version: str
    prompt_version: str

    @classmethod
    def from_environment(cls) -> "ReleaseVersions":
        return cls(
            application_version=os.environ.get("APPLICATION_VERSION", "3.5.1"),
            knowledge_version=os.environ.get("KNOWLEDGE_VERSION", "competition-kb-v1"),
            model_configuration_version=os.environ.get(
                "MODEL_CONFIGURATION_VERSION",
                "model-config-v1",
            ),
            prompt_version=os.environ.get("PROMPT_VERSION", "prompt-v1"),
        )

    def as_dict(self) -> dict[str, str]:
        return asdict(self)
