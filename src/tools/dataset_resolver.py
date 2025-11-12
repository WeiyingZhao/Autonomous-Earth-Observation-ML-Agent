"""
Dataset Resolver Tool - Finds and matches open datasets for EO ML tasks.
Searches HuggingFace, Radiant MLHub, Kaggle, and TorchGeo.
"""

from typing import List, Dict, Any, Optional
from langchain_core.tools import tool
from langchain_core.prompts import PromptTemplate
import yaml
import json
import os


class DatasetResolver:
    """
    Resolves paper requirements to available open datasets.
    Implements the mapping logic from readme.md Phase 2.
    """

    def __init__(self, llm=None, prompts_config: Optional[str] = None):
        """
        Initialize Dataset Resolver.

        Args:
            llm: Language model for intelligent matching
            prompts_config: Path to prompts.yml
        """
        self.llm = llm
        self.prompts_config = prompts_config or "src/config/prompts.yml"
        self.prompts = self._load_prompts()
        self.dataset_catalog = self._build_dataset_catalog()

    def _load_prompts(self) -> Dict[str, Any]:
        """Load prompts from YAML configuration."""
        if os.path.exists(self.prompts_config):
            with open(self.prompts_config, 'r') as f:
                return yaml.safe_load(f)
        return {}

    def _build_dataset_catalog(self) -> List[Dict[str, Any]]:
        """
        Build catalog of known EO datasets.
        Based on readme.md knowledge base of popular EO datasets.
        """
        return [
            # Sentinel-2 datasets
            {
                "name": "BigEarthNet",
                "source": "torchgeo",
                "dataset_id": "torchgeo.datasets.BigEarthNet",
                "task_type": "classification",
                "modality": ["optical"],
                "sensor": "Sentinel-2",
                "resolution_m": 10,
                "num_classes": 43,
                "size_gb": 65,
                "bands": ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"],
                "license": "Community Data License Agreement",
                "splits": {"train": 470000, "val": 117500, "test": 117500},
                "loader_type": "torchgeo",
                "citation": "Sumbul et al., BigEarthNet: A Large-Scale Benchmark Archive..., 2019",
                "url": "https://bigearth.net"
            },
            {
                "name": "EuroSAT",
                "source": "torchgeo",
                "dataset_id": "torchgeo.datasets.EuroSAT",
                "task_type": "classification",
                "modality": ["optical"],
                "sensor": "Sentinel-2",
                "resolution_m": 10,
                "num_classes": 10,
                "size_gb": 1,
                "bands": ["B02", "B03", "B04", "B08"],
                "license": "CC-BY-4.0",
                "splits": {"train": 21600, "val": 5400, "test": 5400},
                "loader_type": "torchgeo",
                "citation": "Helber et al., EuroSAT: A Novel Dataset..., 2019"
            },
            # Multisensor datasets
            {
                "name": "SEN12MS",
                "source": "custom",
                "dataset_id": "https://mediatum.ub.tum.de/1474000",
                "task_type": "segmentation",
                "modality": ["optical", "SAR"],
                "sensor": "Sentinel-1 + Sentinel-2",
                "resolution_m": 10,
                "num_classes": 17,
                "size_gb": 100,
                "bands": ["S1_VV", "S1_VH", "S2_B01-B12"],
                "license": "CC-BY-4.0",
                "splits": {"train": 162556, "val": 20320, "test": 20320},
                "loader_type": "custom",
                "citation": "Schmitt et al., SEN12MS - A Curated Dataset..., 2019"
            },
            # Segmentation datasets
            {
                "name": "LoveDA",
                "source": "huggingface",
                "dataset_id": "AWSAF/LoveDA",
                "task_type": "segmentation",
                "modality": ["optical"],
                "sensor": "aerial",
                "resolution_m": 0.3,
                "num_classes": 7,
                "size_gb": 2.5,
                "bands": ["R", "G", "B"],
                "license": "Apache-2.0",
                "splits": {"train": 2522, "val": 1669, "test": 1711},
                "loader_type": "huggingface",
                "citation": "Wang et al., LoveDA: A Remote Sensing Land-Cover Dataset..., 2021"
            },
            {
                "name": "Inria Aerial Image Labeling",
                "source": "torchgeo",
                "dataset_id": "torchgeo.datasets.InriaAerialImageLabeling",
                "task_type": "segmentation",
                "modality": ["optical"],
                "sensor": "aerial",
                "resolution_m": 0.3,
                "num_classes": 2,
                "size_gb": 10,
                "bands": ["R", "G", "B"],
                "license": "Open Data Commons",
                "splits": {"train": 180, "test": 180},
                "loader_type": "torchgeo",
                "citation": "Maggiori et al., Can Semantic Labeling Methods Generalize..., 2017"
            },
            # Detection datasets
            {
                "name": "DOTA",
                "source": "custom",
                "dataset_id": "https://captain-whu.github.io/DOTA/",
                "task_type": "detection",
                "modality": ["optical"],
                "sensor": "aerial",
                "resolution_m": 0.5,
                "num_classes": 15,
                "size_gb": 20,
                "bands": ["R", "G", "B"],
                "license": "Research only",
                "splits": {"train": 1411, "val": 458, "test": 937},
                "loader_type": "custom",
                "citation": "Xia et al., DOTA: A Large-scale Dataset for Object Detection..., 2018"
            },
            {
                "name": "xView",
                "source": "custom",
                "dataset_id": "http://xviewdataset.org/",
                "task_type": "detection",
                "modality": ["optical"],
                "sensor": "aerial",
                "resolution_m": 0.3,
                "num_classes": 60,
                "size_gb": 15,
                "bands": ["R", "G", "B"],
                "license": "CC-BY-NC-SA-4.0",
                "splits": {"train": 846, "val": 282, "test": None},
                "loader_type": "custom",
                "citation": "Lam et al., xView: Objects in Context..., 2018"
            },
            # SAR datasets
            {
                "name": "Sen1Floods11",
                "source": "custom",
                "dataset_id": "https://github.com/cloudtostreet/Sen1Floods11",
                "task_type": "segmentation",
                "modality": ["SAR", "optical"],
                "sensor": "Sentinel-1",
                "resolution_m": 10,
                "num_classes": 2,
                "size_gb": 5,
                "bands": ["VV", "VH"],
                "license": "CC-BY-4.0",
                "splits": {"train": 4385, "val": 559, "test": 2143},
                "loader_type": "custom",
                "citation": "Bonafilia et al., Sen1Floods11: A Georeferenced Dataset..., 2020"
            },
        ]

    def find_matching_datasets(
        self,
        task_type: str,
        modality: List[str],
        resolution_m: Optional[float] = None,
        num_classes: Optional[int] = None,
        sensor: Optional[str] = None,
        top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Find datasets matching the requirements using heuristic search.

        Args:
            task_type: classification, segmentation, detection
            modality: List of modalities (optical, SAR, etc.)
            resolution_m: Target resolution in meters
            num_classes: Number of classes
            sensor: Specific sensor requirement
            top_k: Number of matches to return

        Returns:
            List of matching datasets with match scores
        """
        matches = []

        for dataset in self.dataset_catalog:
            score = 0
            reasons = []

            # Task match (weight: 40)
            if dataset["task_type"] == task_type:
                score += 40
                reasons.append("exact_task_match")
            elif task_type in dataset["task_type"]:
                score += 20
                reasons.append("partial_task_match")

            # Modality match (weight: 30)
            modality_overlap = set(modality) & set(dataset["modality"])
            if len(modality_overlap) == len(modality):
                score += 30
                reasons.append("full_modality_match")
            elif modality_overlap:
                score += 15
                reasons.append("partial_modality_match")

            # Resolution match (weight: 15)
            if resolution_m and dataset["resolution_m"]:
                ratio = max(resolution_m, dataset["resolution_m"]) / min(resolution_m, dataset["resolution_m"])
                if ratio <= 2:  # Within 2x
                    score += 15
                    reasons.append("close_resolution")
                elif ratio <= 5:  # Within 5x
                    score += 7
                    reasons.append("acceptable_resolution")

            # Sensor match (weight: 10)
            if sensor and sensor.lower() in dataset["sensor"].lower():
                score += 10
                reasons.append("sensor_match")

            # License (weight: 5)
            if "open" in dataset["license"].lower() or "cc-by" in dataset["license"].lower():
                score += 5
                reasons.append("open_license")

            if score > 0:
                matches.append({
                    **dataset,
                    "match_score": score,
                    "match_reasons": reasons
                })

        # Sort by score
        matches.sort(key=lambda x: x["match_score"], reverse=True)

        return matches[:top_k]

    def resolve_with_llm(
        self,
        paper_spec: Dict[str, Any],
        candidate_datasets: List[Dict[str, Any]],
        llm=None
    ) -> Dict[str, Any]:
        """
        Use LLM to select the best dataset and explain rationale.

        Args:
            paper_spec: Paper specification dict
            candidate_datasets: List of candidate datasets from heuristic search
            llm: Language model to use

        Returns:
            Recommendation with rationale
        """
        if llm is None:
            if self.llm is None:
                from src.agent.router import get_router
                router = get_router()
                llm = router.get_model("dataset_resolution")
            else:
                llm = self.llm

        # Get prompt
        resolver_prompt = self.prompts.get("dataset_resolver", {}).get("prompt", "")

        if not resolver_prompt:
            resolver_prompt = """
            Given the paper requirements and candidate datasets, select the best match.
            Return JSON with recommended dataset and rationale.
            """

        # Format prompt
        prompt = PromptTemplate.from_template(
            "{prompt}\n\nPaper: {paper_spec}\n\nCandidates: {candidates}\n\nReturn JSON."
        )

        formatted_prompt = prompt.format(
            prompt=resolver_prompt,
            paper_spec=json.dumps(paper_spec, indent=2),
            candidates=json.dumps(candidate_datasets, indent=2)
        )

        # Invoke LLM
        response = llm.invoke(formatted_prompt)

        # Parse response
        try:
            if hasattr(response, 'content'):
                response_text = response.content
            else:
                response_text = str(response)

            # Extract JSON
            import re
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_str = response_text

            result = json.loads(json_str)
            return result

        except json.JSONDecodeError:
            # Fallback: return best match
            if candidate_datasets:
                return {
                    "recommended": candidate_datasets[0],
                    "rationale": "Selected based on highest match score",
                    "alternatives": candidate_datasets[1:]
                }
            else:
                return {
                    "recommended": None,
                    "rationale": "No suitable datasets found",
                    "alternatives": []
                }

    def resolve_dataset(
        self,
        paper_spec: Dict[str, Any],
        use_llm: bool = True,
        llm=None
    ) -> Dict[str, Any]:
        """
        Complete dataset resolution pipeline.

        Args:
            paper_spec: Paper specification from paper ingestor
            use_llm: Whether to use LLM for final selection
            llm: Language model to use

        Returns:
            Dataset recommendation with alternatives
        """
        # Extract requirements
        task_type = paper_spec.get("tasks", ["classification"])[0]
        sensors = paper_spec.get("sensors", [])
        data_req = paper_spec.get("data_requirements", {})

        # Infer modality from sensors
        modality = []
        for sensor in sensors:
            if "SAR" in sensor.upper() or "S1" in sensor:
                modality.append("SAR")
            elif "Sentinel-2" in sensor or "Landsat" in sensor or "optical" in sensor.lower():
                modality.append("optical")

        if not modality:
            modality = ["optical"]  # Default

        resolution_m = data_req.get("gsd_m") or data_req.get("resolution_m")

        # Step 1: Heuristic search
        candidates = self.find_matching_datasets(
            task_type=task_type,
            modality=modality,
            resolution_m=resolution_m,
            sensor=sensors[0] if sensors else None,
            top_k=3
        )

        # Step 2: LLM refinement
        if use_llm and candidates:
            result = self.resolve_with_llm(paper_spec, candidates, llm)
        else:
            # Return best heuristic match
            if candidates:
                result = {
                    "recommended": candidates[0],
                    "rationale": f"Best heuristic match (score: {candidates[0]['match_score']})",
                    "alternatives": candidates[1:]
                }
            else:
                result = {
                    "recommended": None,
                    "rationale": "No matching datasets found in catalog",
                    "alternatives": []
                }

        return result


# LangChain tool wrapper
@tool
def resolve_dataset_tool(paper_spec: Dict[str, Any]) -> Dict[str, Any]:
    """
    Resolve paper requirements to an appropriate open dataset.

    Args:
        paper_spec: Paper specification dict with tasks, sensors, and data requirements

    Returns:
        Dataset recommendation with source, ID, and rationale
    """
    resolver = DatasetResolver()
    result = resolver.resolve_dataset(paper_spec)
    return result


# Example usage
if __name__ == "__main__":
    resolver = DatasetResolver()

    # Example paper spec
    paper_spec = {
        "title": "Deep Learning for Land Cover Classification",
        "tasks": ["segmentation"],
        "sensors": ["Sentinel-2"],
        "data_requirements": {
            "gsd_m": 10,
            "bands": ["B02", "B03", "B04", "B08"]
        }
    }

    result = resolver.resolve_dataset(paper_spec, use_llm=False)
    print(json.dumps(result, indent=2))
