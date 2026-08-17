"""
AtmosIQ Phase 8C: Final Synthetic Corpus Consolidation, Governance & Production Training Release Package.
"""

from .config import ReleaseConfigPhase8C
from .governance import ExtremeTailGovernanceEngine
from .consolidation import CorpusConsolidationEngine
from .provenance import Phase8CProvenanceManager
from .audits import IntegrityAndIsolationAuditor
from .policy import SyntheticAugmentationPolicyEngine
from .contract import Phase9TrainingContractEngine
from .runner import Phase8CRunner

__all__ = [
    "ReleaseConfigPhase8C",
    "ExtremeTailGovernanceEngine",
    "CorpusConsolidationEngine",
    "Phase8CProvenanceManager",
    "IntegrityAndIsolationAuditor",
    "SyntheticAugmentationPolicyEngine",
    "Phase9TrainingContractEngine",
    "Phase8CRunner",
]
