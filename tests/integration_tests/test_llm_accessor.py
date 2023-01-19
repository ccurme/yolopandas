import os
import unittest

from llpandas.llm_accessor import pd
from tests import TEST_DIRECTORY


class TestLLMAccessor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        test_data_path = os.path.join(TEST_DIRECTORY, "data", "product_df.json")
        cls.product_df = pd.read_json(test_data_path)

    def test_basic_use(self):
        self.product_df.llm.reset_chain(use_memory=False)
        result = self.product_df.llm.query(
            "What is the price of the highest-priced book?",
            verify=False,
        )
        expected_result = 15
        self.assertEqual(expected_result, result)

        result = self.product_df.llm.query(
            "What is the average price of products grouped by type?",
            verify=False,
        )
        expected = self.product_df.groupby("type")["price"].mean()
        pd.testing.assert_series_equal(expected, result)

        result = self.product_df.llm.query(
            "Give me products that are not books.",
            verify=False,
        )
        expected = self.product_df[self.product_df["type"] != "book"]
        pd.testing.assert_frame_equal(expected, result)

    def test_memory(self):
        self.product_df.llm.reset_chain(use_memory=True)
        _ = self.product_df.llm.query(
            "Show me all products that are books.",
            verify=False,
        )
        result = self.product_df.llm.query(
            "Of these, which has the fewest items stocked?",
            verify=False,
        )
        expected = (
            self.product_df[self.product_df["type"] == "book"]
            .sort_values(by="quantity")
            .head(1)
        )
        pd.testing.assert_frame_equal(expected, result)
