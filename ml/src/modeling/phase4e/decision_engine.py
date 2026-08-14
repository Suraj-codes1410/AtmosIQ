import sys
from pathlib import Path
from typing import List, Dict
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ml.src.modeling.phase4e.data_loader import DataLoaderPhase4E
from ml.src.modeling.phase4e.cache import CachePhase4E
from ml.src.modeling.phase4e.prediction_service import PredictionServicePhase4E
from ml.src.modeling.phase4e.shap_service import SHAPServicePhase4E
from ml.src.modeling.phase4e.validation_service import ValidationServicePhase4E
from ml.src.modeling.phase4e.counterfactual_service import CounterfactualServicePhase4E
from ml.src.modeling.phase4e.confidence_service import ConfidenceServicePhase4E
from ml.src.modeling.phase4e.event_service import EventServicePhase4E
from ml.src.modeling.phase4e.response_schema import (
    DecisionSupportResponse,
    SCIENTIFIC_DISCLAIMER
)


class DecisionEnginePhase4E:
    """
    AtmosIQ Phase 4E Master Decision Support Engine.
    Integrates prediction, SHAP attribution, environmental validation, counter-evidence, counterfactual sensitivity, and confidence.
    """

    def __init__(self, data_loader: DataLoaderPhase4E = None, cache: CachePhase4E = None):
        self.loader = data_loader or DataLoaderPhase4E()
        self.cache = cache or CachePhase4E(self.loader)

        self.pred_service = PredictionServicePhase4E(self.loader, self.cache)
        self.shap_service = SHAPServicePhase4E(self.loader, self.cache)
        self.val_service = ValidationServicePhase4E(self.loader, self.cache)
        self.cf_service = CounterfactualServicePhase4E(self.loader, self.cache)
        self.conf_service = ConfidenceServicePhase4E(self.loader, self.cache)
        self.event_service = EventServicePhase4E(self.loader, self.cache)

    def generate_decision_support(self, date_str: str) -> DecisionSupportResponse:
        """Generates unified decision-support report for a single YYYY-MM-DD date."""
        pred = self.pred_service.predict_date(date_str)
        shap_res = self.shap_service.explain_prediction(date_str)
        val = self.val_service.validate_attribution(date_str)
        conf = self.conf_service.calculate_confidence(date_str)
        extreme_summary = self.event_service.analyze_extreme_event(date_str)

        # Counterfactual scenario evaluation
        scenarios_to_evaluate = ["biomass_low", "wind_dispersion", "combined_all_favorable"]
        cf_dict = {}
        for s in scenarios_to_evaluate:
            cf_dict[s] = self.cf_service.run_counterfactual(date_str, s)

        # Formulate conservative scientific interpretation
        dom = shap_res.dominant_group
        pred_val = pred.predicted_pm25

        bio_delta = cf_dict.get("biomass_low", None)
        bio_txt = f"Under the defined biomass-low feature intervention, the model prediction drops by {abs(bio_delta.delta_prediction):.1f} µg/m³." if bio_delta else ""

        interp = (
            f"For {date_str}, the frozen AtmosIQ model predicts next-day PM2.5 of {pred_val:.1f} µg/m³ ({pred.pollution_category}). "
            f"The primary model attribution signal is '{dom}'. "
            f"Independent environmental validation status: {val.validation_status}. {bio_txt}"
        )

        return DecisionSupportResponse(
            date=date_str,
            prediction=pred,
            attribution=shap_res,
            validation=val,
            counterfactual_scenarios=cf_dict,
            confidence=conf,
            extreme_event_analysis=extreme_summary,
            scientific_interpretation=interp,
            scientific_limitations=SCIENTIFIC_DISCLAIMER
        )

    def analyze_dates(self, dates: List[str]) -> List[DecisionSupportResponse]:
        """Analyzes a list of YYYY-MM-DD dates."""
        return [self.generate_decision_support(d) for d in dates]

    def analyze_period(self, start_date: str, end_date: str) -> List[DecisionSupportResponse]:
        """Analyzes all daily observations within a date range."""
        dates_dt = pd.to_datetime(self.loader.df_v2["date"])
        mask = (dates_dt >= pd.to_datetime(start_date)) & (dates_dt <= pd.to_datetime(end_date))
        matched_dates = self.loader.df_v2.loc[mask, "date"].tolist()
        return self.analyze_dates(matched_dates)


if __name__ == "__main__":
    engine = DecisionEnginePhase4E()
    report = engine.generate_decision_support("2024-11-16")
    print(report.model_dump_json(indent=2))
