import os
import unittest
from unittest.mock import Mock, patch

from yolopandas import pd
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

    @patch("yolopandas.llm_accessor.get_chain")
    def test_basic_use(self, mock):
        mock.return_value = _get_mock_chain("df[df['type'] == 'book']['price'].max()")
        result = self.product_df.llm.query(
            "What is the price of the highest-priced book?",
            yolo=True,
        )
        expected_result = 15
        self.assertEqual(expected_result, result)

        mock.return_value = _get_mock_chain("df.groupby('type')['price'].mean()")
        self.product_df.llm.reset_chain()
        result = self.product_df.llm.query(
            "What is the average price of products grouped by type?",
            yolo=True,
        )
        expected_result = self.product_df.groupby("type")["price"].mean()
        pd.testing.assert_series_equal(expected_result, result)

        mock.return_value = _get_mock_chain("df[df['type'] != 'book']")
        self.product_df.llm.reset_chain()
        result = self.product_df.llm.query(
            "Give me products that are not books.",
            yolo=True,
        )
        expected = self.product_df[self.product_df["type"] != "book"]
        pd.testing.assert_frame_equal(expected, result)

    @patch("yolopandas.llm_accessor.get_chain")
    def test_sliced(self, mock):
        mock.return_value = _get_mock_chain("df[df['type'] == 'book']['price'].max()")
        self.product_df.llm.reset_chain()
        result = self.product_df[["name", "type", "price", "rating"]].llm.query(
            "What is the price of the highest-priced book?", yolo=True
        )
        expected_result = 15
        self.assertEqual(expected_result, result)

    @patch("yolopandas.llm_accessor.get_chain")
    def test_multi_line(self, mock):
        """Test that we can accommodate multiple lines in the LLM response."""
        query = """
        Add a column `new_column` to the dataframe which is range 1 - number of rows,
        then return the mean of this column by each type.
        """
        mock_response = (
            "df['new_column'] = range(1, len(df) + 1)\n"
            "df.groupby('type')['new_column'].mean()"
        )
        mock.return_value = _get_mock_chain(mock_response)
        self.product_df.llm.reset_chain()
        result = self.product_df.llm.query(query, yolo=True)
        expected = (
            self.product_df.assign(new_column=range(1, len(self.product_df) + 1))
            .groupby("type")["new_column"]
            .mean()
        )
        pd.testing.assert_series_equal(expected, result)

    @patch("yolopandas.llm_accessor.get_chain")
    def test_multiline_exec(self, mock):
        """Test a multiline command when the final line should be exec'd not eval'd."""
        query = """
        Add a column `new_column` to the dataframe which is range 1 - number of rows,
        then add a column `foo` which is always the value 1
        """
        mock_response = "df['new_column'] = range(1, len(df) + 1)\n" "df['foo'] = 1"
        mock.return_value = _get_mock_chain(mock_response)
        self.product_df.llm.reset_chain()
        self.product_df.llm.query(query, yolo=True)
        expected_df = self.product_df.assign(
            new_column=range(1, len(self.product_df) + 1)
        ).assign(foo=1)
        pd.testing.assert_frame_equal(expected_df, self.product_df)
