import os
import unittest
from unittest import mock

from llpandas.llm_accessor import pd
from llpandas.chains import OpenAI
from tests import TEST_DIRECTORY


class TestLLMAccessor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        test_data_path = os.path.join(TEST_DIRECTORY, "data", "product_df.json")
        cls.product_df = pd.read_json(test_data_path)

    def test_basic_use(self):

        with mock.patch.object(OpenAI, "run") as mock_run:
            mock_run.return_value = "abc"
            result = self.product_df.llm.query(
                "What is the price of the highest-priced book?",
                verify=False,
            )
        expected_result = 15
        self.assertEqual(expected_result, result)
