import os
import unittest

from tests import TEST_DIRECTORY
from yolopandas import pd


class TestLLMAccessor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        test_data_path = os.path.join(TEST_DIRECTORY, "data", "product_df.json")
        cls.product_df = pd.read_json(test_data_path)

    def test_basic_use(self):
        self.product_df.llm.reset_chain(use_memory=False)
        result = self.product_df.llm.query(
            "What is the price of the highest-priced book?",
            yolo=True,
        )
        expected_result = 15
        self.assertEqual(expected_result, result)

        result = self.product_df.llm.query(
            "What is the average price of products grouped by type?",
            yolo=True,
        )
        expected = self.product_df.groupby("type")["price"].mean()
        pd.testing.assert_series_equal(expected, result)

        result = self.product_df.llm.query(
            "Give me products that are not books.",
            yolo=True,
        )
        expected = self.product_df[self.product_df["type"] != "book"]
        pd.testing.assert_frame_equal(expected, result)

    def test_memory(self):
        self.product_df.llm.reset_chain(use_memory=True)
        _ = self.product_df.llm.query(
            "Show me all products that are books.",
            yolo=True,
        )
        result = self.product_df.llm.query(
            "Of these, which has the fewest items stocked?",
            yolo=True,
        )
        expected = (
            self.product_df[self.product_df["type"] == "book"]
            .sort_values(by="quantity")
            .head(1)
        )
        pd.testing.assert_frame_equal(expected, result)
