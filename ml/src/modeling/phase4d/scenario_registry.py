import sys
import json
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ml.src.utils.logger import setup_logger

logger = setup_logger("ScenarioRegistryPhase4D")


class ScenarioRegistryPhase4D:
    """
    AtmosIQ Phase 4D Machine-Readable Counterfactual Scenario Registry.
    Defines predefined single-group and multi-group counterfactual scenarios with explicit physical rationales and limitations.
    """

    def __init__(self, exp_dir: str = "ml/experiments/phase4d"):
        self.exp_dir = Path(exp_dir)
        self.exp_dir.mkdir(parents=True, exist_ok=True)

        self.scenarios = {
            "biomass_low": {
                "group": "biomass_burning",
                "method": "q25_reference",
                "quantile": 0.25,
                "description": "Reduce biomass-burning signal to low historical background reference conditions (25th percentile).",
                "physical_rationale": "Simulates substantial reduction in upwind stubble burning and open fire emissions.",
                "limitations": "Model sensitivity estimate only; does not account for atmospheric chemical transformation dynamics."
            },
            "biomass_median": {
                "group": "biomass_burning",
                "method": "q50_reference",
                "quantile": 0.50,
                "description": "Replace biomass-burning signal with median historical conditions (50th percentile).",
                "physical_rationale": "Simulates typical background biomass burning signal.",
                "limitations": "Model sensitivity estimate."
            },
            "biomass_high": {
                "group": "biomass_burning",
                "method": "q75_reference",
                "quantile": 0.75,
                "description": "Increase biomass-burning signal to high historical conditions (75th percentile).",
                "physical_rationale": "Simulates peak post-monsoon stubble burning episode.",
                "limitations": "Model sensitivity estimate."
            },
            "wind_stagnant": {
                "group": "wind_ventilation",
                "method": "q25_reference",
                "quantile": 0.25,
                "description": "Set ventilation features to low surface wind speed / poor ventilation reference state (25th percentile).",
                "physical_rationale": "Simulates winter boundary layer atmospheric stagnation.",
                "limitations": "Model sensitivity estimate."
            },
            "wind_normal": {
                "group": "wind_ventilation",
                "method": "q50_reference",
                "quantile": 0.50,
                "description": "Replace ventilation features with median historical wind conditions.",
                "physical_rationale": "Simulates baseline atmospheric dispersion conditions.",
                "limitations": "Model sensitivity estimate."
            },
            "wind_dispersion": {
                "group": "wind_ventilation",
                "method": "q75_reference",
                "quantile": 0.75,
                "description": "Set ventilation features to high surface wind speed / strong dispersion reference state (75th percentile).",
                "physical_rationale": "Simulates strong atmospheric ventilation blowing away localized pollutants.",
                "limitations": "Model sensitivity estimate."
            },
            "meteorology_normal": {
                "group": "meteorology",
                "method": "q50_reference",
                "quantile": 0.50,
                "description": "Replace meteorological features with median historical weather conditions.",
                "physical_rationale": "Normalizes temperature and humidity to baseline historical averages.",
                "limitations": "Model sensitivity estimate."
            },
            "combined_biomass_wind": {
                "group": "multi_group",
                "groups": ["biomass_burning", "wind_ventilation"],
                "method": "combined_favorable",
                "description": "Combine low biomass burning (Q25) with high atmospheric ventilation (Q75).",
                "physical_rationale": "Simulates reduced agricultural fires co-occurring with strong dispersion winds.",
                "limitations": "Multi-group model counterfactual sensitivity."
            },
            "combined_all_favorable": {
                "group": "multi_group",
                "groups": ["biomass_burning", "wind_ventilation", "meteorology"],
                "method": "all_favorable",
                "description": "Combine low biomass burning (Q25), high ventilation (Q75), and median weather (Q50).",
                "physical_rationale": "Simulates optimal environmental control and dispersion scenario.",
                "limitations": "Multi-group model counterfactual sensitivity."
            }
        }

    def export_registry_json(self) -> Path:
        """Exports scenario_registry.json to experiment directory."""
        output_file = self.exp_dir / "scenario_registry.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(self.scenarios, f, indent=4)
        logger.info(f"Scenario registry exported to {output_file}.")
        return output_file


if __name__ == "__main__":
    registry = ScenarioRegistryPhase4D()
    registry.export_registry_json()
