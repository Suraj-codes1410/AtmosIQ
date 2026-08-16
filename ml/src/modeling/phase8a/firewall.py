"""
AtmosIQ Phase 8A: Locked Evaluation Data Isolation Firewall.
"""

from typing import Optional
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class EvaluationFirewallViolation(Exception):
    """Raised when evaluation-period data breaches the generator firewall."""
    pass


class EvaluationFirewall:
    """
    Code-level firewall preventing accidental leakage of the locked 2022–2024
    evaluation fold into training data or generator parameter estimation.
    """

    def __init__(self, locked_eval_start_date: str = "2022-01-01"):
        self.locked_eval_start_date = locked_eval_start_date

    def verify_training_partition_isolation(self, df_input: pd.DataFrame, source_name: str = "input_dataframe") -> bool:
        """
        Ensures input dataframe contains ONLY development data (date < 2022-01-01).
        Fails loudly with EvaluationFirewallViolation if violated.
        """
        if "date" not in df_input.columns:
            logger.warning(f"Firewall check on {source_name}: 'date' column not found.")
            return True

        dates = pd.to_datetime(df_input["date"])
        eval_start = pd.to_datetime(self.locked_eval_start_date)

        violations = df_input[dates >= eval_start]
        if len(violations) > 0:
            msg = (
                f"CRITICAL FIREWALL VIOLATION in {source_name}: "
                f"Found {len(violations)} records with date >= {self.locked_eval_start_date}. "
                f"Earliest violation: {violations['date'].min()}, Latest violation: {violations['date'].max()}. "
                f"Locked evaluation fold (2022–2024) is strictly isolated and cannot be used for synthetic generation!"
            )
            logger.error(msg)
            raise EvaluationFirewallViolation(msg)

        logger.info(f"Evaluation Firewall verified for {source_name}: 0 evaluation records found (Max Date: {df_input['date'].max()}).")
        return True
