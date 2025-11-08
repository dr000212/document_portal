import os
import sys
from utils.LLM_Loader import ModelLoader
from logger.custom_logging import CustomLogger
from exception.custom_exception import DocumentPortalException
from model.model import *
from langchain_core.output_parsers import JsonOutputParser
from langchain.output_parsers import OutputFixingParser
from prompt_template.prompt import PROMPT_REGISTRY


class DocumentAnalyzer:
    """
    Analyzes documents using a pre-trained LLM model.
    Automatically logs all actions and supports session-based organization.
    """

    def __init__(self):
        self.log = CustomLogger().get_logger(__name__)
        try:
            self.loader = ModelLoader()
            self.llm = self.loader.load_llm()

            # Prepare output parsers
            self.parser = JsonOutputParser(pydantic_object=Metadata)
            self.fixing_parser = OutputFixingParser.from_llm(parser=self.parser, llm=self.llm)

            # Load the analysis prompt
            self.prompt = PROMPT_REGISTRY["document_analysis"]

            self.log.info("✅ DocumentAnalyzer initialized successfully")

        except Exception as e:
            self.log.error(f"❌ Error initializing DocumentAnalyzer: {e}")
            raise DocumentPortalException("Error in DocumentAnalyzer initialization", sys)

    def analyze_document(self, document_text: str) -> dict:
        """
        Analyze a document's text and extract structured metadata & summary.
        """
        try:
            # Build analysis chain
            chain = self.prompt | self.llm | self.fixing_parser
            self.log.info("🔗 Metadata analysis chain initialized")

            # Invoke LLM chain
            response = chain.invoke({
                "format_instructions": self.parser.get_format_instructions(),
                "document_text": document_text
            })

            # Log success
            self.log.info(f"✅ Metadata extraction successful | keys={list(response.keys())}")
            return response

        except Exception as e:
            self.log.error(f"❌ Metadata analysis failed: {e}")
            raise DocumentPortalException("Metadata extraction failed", sys)
