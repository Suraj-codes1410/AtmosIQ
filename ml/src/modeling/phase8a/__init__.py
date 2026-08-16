"""
AtmosIQ Phase 8A: Large-Scale Synthetic Data Generation Infrastructure & Controlled Expansion.
"""

from .config import GenerationConfigPhase8A
from .firewall import EvaluationFirewall, EvaluationFirewallViolation
from .provenance import Phase8AProvenanceManager
from .validation import Phase8APhysicsValidator
from .filtering import ExtremeTailFilter, RejectionRecord
from .ood_support import OODSupportScorer
from .memorization import MemorizationScreen
from .generator import ProductionTrajectoryGenerator
from .sharding import DatasetSharder
from .manifest import DatasetManifestGenerator
from .runner import Phase8ARunner

__all__ = [
    "GenerationConfigPhase8A",
    "EvaluationFirewall",
    "EvaluationFirewallViolation",
    "Phase8AProvenanceManager",
    "Phase8APhysicsValidator",
    "ExtremeTailFilter",
    "RejectionRecord",
    "OODSupportScorer",
    "MemorizationScreen",
    "ProductionTrajectoryGenerator",
    "DatasetSharder",
    "DatasetManifestGenerator",
    "Phase8ARunner",
]
