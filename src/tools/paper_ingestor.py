"""
Paper Ingestor Tool - Converts PDF papers to structured JSON.
Uses PDF parsing libraries and LLM for extraction.
"""

from typing import Optional, Dict, Any
from langchain_core.tools import tool
from langchain_core.prompts import PromptTemplate
import fitz  # PyMuPDF
import json
import yaml
import os
from pathlib import Path
import re


class PaperIngestor:
    """
    Extracts structured information from ML research papers.
    Combines PDF parsing with LLM-based extraction.
    """

    def __init__(self, llm=None, prompts_config: Optional[str] = None):
        """
        Initialize the Paper Ingestor.

        Args:
            llm: Language model for extraction (uses router if None)
            prompts_config: Path to prompts.yml file
        """
        self.llm = llm
        self.prompts_config = prompts_config or "src/config/prompts.yml"
        self.prompts = self._load_prompts()

    def _load_prompts(self) -> Dict[str, Any]:
        """Load prompts from YAML configuration."""
        if os.path.exists(self.prompts_config):
            with open(self.prompts_config, 'r') as f:
                return yaml.safe_load(f)
        return {}

    def extract_text_from_pdf(self, pdf_path: str) -> Dict[str, str]:
        """
        Extract text from PDF and split into sections.

        Args:
            pdf_path: Path to PDF file

        Returns:
            Dict with sections: abstract, introduction, methods, experiments, etc.
        """
        try:
            doc = fitz.open(pdf_path)
            full_text = ""

            # Extract text from all pages
            for page in doc:
                full_text += page.get_text()

            doc.close()

            # Split into sections using heuristics
            sections = self._split_into_sections(full_text)

            return sections

        except Exception as e:
            raise ValueError(f"Failed to extract text from PDF: {str(e)}")

    def _split_into_sections(self, text: str) -> Dict[str, str]:
        """
        Split paper text into sections using common headers.

        Args:
            text: Full paper text

        Returns:
            Dict mapping section names to content
        """
        # Common section headers (case-insensitive patterns)
        section_patterns = {
            "abstract": r"(?i)\n\s*abstract\s*\n",
            "introduction": r"(?i)\n\s*(?:1\.?\s*)?introduction\s*\n",
            "related_work": r"(?i)\n\s*(?:2\.?\s*)?related\s+work\s*\n",
            "methods": r"(?i)\n\s*(?:3\.?\s*)?(?:methods?|methodology|approach)\s*\n",
            "experiments": r"(?i)\n\s*(?:4\.?\s*)?(?:experiments?|results)\s*\n",
            "conclusion": r"(?i)\n\s*(?:5\.?\s*)?conclusions?\s*\n",
            "references": r"(?i)\n\s*references\s*\n"
        }

        sections = {}
        positions = []

        # Find all section headers
        for section_name, pattern in section_patterns.items():
            match = re.search(pattern, text)
            if match:
                positions.append((match.start(), section_name))

        # Sort by position
        positions.sort()

        # Extract section content
        for i, (start_pos, section_name) in enumerate(positions):
            if i < len(positions) - 1:
                end_pos = positions[i + 1][0]
            else:
                end_pos = len(text)

            sections[section_name] = text[start_pos:end_pos].strip()

        # If no sections found, use full text
        if not sections:
            sections["full_text"] = text

        return sections

    def extract_metadata_with_llm(
        self,
        sections: Dict[str, str],
        llm=None
    ) -> Dict[str, Any]:
        """
        Use LLM to extract structured metadata from paper sections.

        Args:
            sections: Dict of paper sections
            llm: Language model to use (uses self.llm if None)

        Returns:
            Structured paper specification
        """
        if llm is None:
            if self.llm is None:
                # Import here to avoid circular dependency
                from src.agent.router import get_router
                router = get_router()
                llm = router.get_model("paper_parsing")
            else:
                llm = self.llm

        # Combine relevant sections
        relevant_text = "\n\n".join([
            f"# {section.upper()}\n{content}"
            for section, content in sections.items()
            if section in ["abstract", "methods", "experiments", "introduction"]
        ])

        # Get extraction prompt
        extraction_prompt = self.prompts.get("method_extraction", {}).get("prompt", "")

        if not extraction_prompt:
            # Fallback prompt
            extraction_prompt = """
            Extract the following information from this ML paper:
            - Title
            - Tasks (classification, segmentation, etc.)
            - Sensors/modalities
            - Method details
            - Metrics
            - Datasets mentioned

            Return valid JSON.
            """

        # Create prompt
        prompt = PromptTemplate.from_template(
            "{prompt}\n\nPaper content:\n{paper_text}\n\nReturn only valid JSON."
        )

        # Invoke LLM
        formatted_prompt = prompt.format(
            prompt=extraction_prompt,
            paper_text=relevant_text[:50000]  # Limit to 50K chars
        )

        response = llm.invoke(formatted_prompt)

        # Extract JSON from response
        try:
            # Try to parse as JSON
            if hasattr(response, 'content'):
                response_text = response.content
            else:
                response_text = str(response)

            # Find JSON in response (may be wrapped in markdown code blocks)
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # Try to find JSON directly
                json_str = response_text

            metadata = json.loads(json_str)
            return metadata

        except json.JSONDecodeError as e:
            # Return partial structure with raw text
            return {
                "title": "Unknown",
                "raw_extraction": response_text,
                "error": f"Failed to parse JSON: {str(e)}"
            }

    def ingest_paper(self, paper_path: str, llm=None) -> Dict[str, Any]:
        """
        Complete paper ingestion pipeline.

        Args:
            paper_path: Path to paper PDF
            llm: Language model to use

        Returns:
            Structured paper specification
        """
        # Step 1: Extract text from PDF
        sections = self.extract_text_from_pdf(paper_path)

        # Step 2: Extract structured metadata with LLM
        metadata = self.extract_metadata_with_llm(sections, llm)

        # Step 3: Add raw sections for reference
        metadata["_raw_sections"] = {
            k: v[:1000] + "..." if len(v) > 1000 else v
            for k, v in sections.items()
        }

        return metadata


# LangChain tool wrapper
@tool
def ingest_paper_tool(paper_path: str) -> Dict[str, Any]:
    """
    Ingest a research paper and extract structured information.

    Args:
        paper_path: Path to PDF file or arXiv link

    Returns:
        Structured paper specification including tasks, methods, datasets, and metrics
    """
    ingestor = PaperIngestor()
    result = ingestor.ingest_paper(paper_path)
    return result


def download_arxiv_paper(arxiv_id: str, output_dir: str = "/tmp") -> str:
    """
    Download paper from arXiv.

    Args:
        arxiv_id: arXiv ID (e.g., "2103.14030")
        output_dir: Directory to save PDF

    Returns:
        Path to downloaded PDF
    """
    import requests

    # Clean arxiv ID
    arxiv_id = arxiv_id.replace("arXiv:", "").strip()

    # arXiv PDF URL
    url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"

    # Download
    response = requests.get(url)
    response.raise_for_status()

    # Save to file
    output_path = os.path.join(output_dir, f"{arxiv_id}.pdf")
    with open(output_path, 'wb') as f:
        f.write(response.content)

    return output_path


# Example usage
if __name__ == "__main__":
    # Example: Ingest a paper
    ingestor = PaperIngestor()

    # Test with a local PDF (replace with actual path)
    # result = ingestor.ingest_paper("/path/to/paper.pdf")
    # print(json.dumps(result, indent=2))

    print("Paper Ingestor Tool initialized successfully")
    print("Use ingest_paper_tool(paper_path) to process papers")
