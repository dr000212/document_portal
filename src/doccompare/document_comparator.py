import sys
from dotenv import load_dotenv
import pandas as pd
from logger.custom_logging import CustomLogger
from exception.custom_exception import DocumentPortalException
from model.model import *
from prompt_template.prompt import PROMPT_REGISTRY
from utils.LLM_Loader import ModelLoader
from langchain_core.output_parsers import JsonOutputParser
from langchain.output_parsers import OutputFixingParser


class DocumentComparatorLLM:
    """
    Compares two documents using an LLM and returns structured results as a DataFrame.
    """

    def __init__(self):
        load_dotenv()
        self.log = CustomLogger().get_logger(__name__)

        try:
            self.loader = ModelLoader()
            self.llm = self.loader.load_llm()

            self.parser = JsonOutputParser(pydantic_object=SummaryResponse)
            self.fixing_parser = OutputFixingParser.from_llm(parser=self.parser, llm=self.llm)

            # pick the correct prompt from registry
            self.prompt = PROMPT_REGISTRY.get(
                PromptType.DOCUMENT_COMPARISON.value,
                PROMPT_REGISTRY.get("document_comparison")
            )

            # construct the LLM chain
            self.chain = self.prompt | self.llm | self.parser

            self.log.info(f"DocumentComparatorLLM initialized successfully. Model: {self.llm}")

        except Exception as e:
            self.log.error(f"Error initializing DocumentComparatorLLM: {e}")
            raise DocumentPortalException("Initialization failed", sys)

    def compare_documents(self, combined_docs: str) -> pd.DataFrame:
        """
        Invokes the LLM chain to compare documents and returns a DataFrame.
        """
        try:
            inputs = {
                "combined_docs": combined_docs,
                "format_instruction": self.parser.get_format_instructions()
            }

            self.log.info("Invoking document comparison LLM chain.")
            response = self.chain.invoke(inputs)
            self.log.info(f"Chain invoked successfully. Response preview: {str(response)[:300]}")

            return self._format_response(response)

        except Exception as e:
            self.log.error(f"Error in compare_documents: {e}")
            raise DocumentPortalException("Error comparing documents", sys)

    def _format_response(self, response_parsed: list[dict]) -> pd.DataFrame:
        """
        Formats the LLM's JSON output into a structured DataFrame.
        """
        try:
            df = pd.DataFrame(response_parsed)
            self.log.info(f"Response formatted into DataFrame successfully. Rows: {len(df)}, Columns: {list(df.columns)}")
            return df

        except Exception as e:
            self.log.error(f"Error formatting response into DataFrame: {e}")
            raise DocumentPortalException("Error formatting response", sys)
