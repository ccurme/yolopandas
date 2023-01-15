import os
import unittest
from unittest.mock import Mock, patch

from llpandas.llm_accessor import pd
import llpandas.llm_accessor
from langchain.chains.base import Chain
from tests import TEST_DIRECTORY


def _get_mock_chain(response: str) -> Chain:
    """Make mock Chain for unit tests."""
    mock_chain = Mock(spec=Chain)
    mock_chain.run.return_value = response

    return mock_chain


class TestLLMAccessor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        test_data_path = os.path.join(TEST_DIRECTORY, "data", "product_df.json")
        cls.product_df = pd.read_json(test_data_path)

    @patch('llpandas.llm_accessor.get_chain')
    def test_basic_use(self, mock):
        mock.return_value = _get_mock_chain("df[df['type'] == 'book']['price'].max()")
        result = self.product_df.llm.query(
            "What is the price of the highest-priced book?",
            verify=False,
        )
        expected_result = 15
        self.assertEqual(expected_result, result)
