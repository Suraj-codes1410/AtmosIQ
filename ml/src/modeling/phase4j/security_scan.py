import re
import sys
from pathlib import Path
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ml.src.utils.logger import setup_logger

logger = setup_logger("SecurityScanPhase4J")


class ReleaseSecurityScannerPhase4J:
    """
    Release Safety & Secret Scanner for Phase 4J.
    Scans repository artifacts, documentation, and manifests for credentials, private keys, tokens, and secrets.
    """

    SECRET_PATTERNS = [
        (r"(?i)api[_-]?key\s*[:=]\s*['\"][a-zA-Z0-9_\-]{16,}['\"]", "API Key Assignment"),
        (r"(?i)password\s*[:=]\s*['\"][^'\"]{6,}['\"]", "Plaintext Password Assignment"),
        (r"(?i)secret[_-]?key\s*[:=]\s*['\"][a-zA-Z0-9_\-]{16,}['\"]", "Secret Key Assignment"),
        (r"-----BEGIN\s+(RSA|OPENSSH|DSA|EC|PRIVATE)\s+KEY-----", "Private Key Block"),
        (r"(?i)bearer\s+[a-zA-Z0-9_\-\.]{20,}", "Bearer Token"),
        (r"(?i)ghp_[a-zA-Z0-9]{36}", "GitHub Personal Access Token"),
        (r"(?i)aws[_-]?secret[_-]?access[_-]?key", "AWS Secret Key Reference")
    ]

    SCAN_EXTENSIONS = {".py", ".json", ".md", ".csv", ".txt", ".yml", ".yaml", ".cff"}

    def __init__(self, root_dir: Path = ROOT_DIR):
        self.root_dir = root_dir

    def scan_release_tree(self, output_csv: Path) -> pd.DataFrame:
        logger.info("Executing Release Security & Secret Scan...")
        output_csv.parent.mkdir(parents=True, exist_ok=True)

        scan_targets = [
            self.root_dir / "ml" / "models" / "production",
            self.root_dir / "ml" / "releases",
            self.root_dir / "kaggle" / "v3",
            self.root_dir / "docs" / "phase4",
            self.root_dir / "ml" / "src" / "modeling" / "phase4j"
        ]

        findings = []
        scanned_files_count = 0

        for target in scan_targets:
            if not target.exists():
                continue

            for p in target.rglob("*"):
                if p.is_file() and p.suffix in self.SCAN_EXTENSIONS:
                    scanned_files_count += 1
                    try:
                        with open(p, "r", encoding="utf-8", errors="ignore") as f:
                            content = f.read()

                        for pat, pat_name in self.SECRET_PATTERNS:
                            matches = re.findall(pat, content)
                            if matches:
                                findings.append({
                                    "file_path": str(p.relative_to(self.root_dir)),
                                    "pattern_matched": pat_name,
                                    "severity": "CRITICAL",
                                    "status": "FAIL"
                                })
                    except Exception as e:
                        logger.warning(f"Could not scan file {p}: {e}")

        if not findings:
            df_scan = pd.DataFrame([{
                "scanned_files_count": scanned_files_count,
                "secrets_detected": 0,
                "api_keys_detected": 0,
                "private_keys_detected": 0,
                "security_status": "CLEAN",
                "status": "PASS"
            }])
        else:
            df_scan = pd.DataFrame(findings)

        df_scan.to_csv(output_csv, index=False)

        assert df_scan['status'].iloc[0] == "PASS", f"Security scan failed! Detected {len(findings)} potential secrets."
        logger.info(f"Security Scan PASSED cleanly ({scanned_files_count} files scanned, 0 secrets detected).")
        return df_scan
