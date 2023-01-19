import unittest

from llpandas.llm_accessor import pd


class TestLLMAccessor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.product_df = pd.DataFrame(
            [
                {
                    "name": "The Da Vinci Code",
                    "type": "book",
                    "price": 15,
                    "quantity": 300,
                    "rating": 4,
                },
                {
                    "name": "Jurassic Park",
                    "type": "book",
                    "price": 12,
                    "quantity": 400,
                    "rating": 4.5,
                },
                {
                    "name": "Jurassic Park",
                    "type": "film",
                    "price": 8,
                    "quantity": 6,
                    "rating": 5,
                },
                {
                    "name": "Matilda",
                    "type": "book",
                    "price": 6,
                    "quantity": 80,
                    "rating": 4,
                },
            ],
        )

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

    def test_sliced(self):
        result = self.product_df[["name", "type", "price", "rating"]].llm.query(
            "What is the price of the highest-priced book?", verify=False
        )
        expected_result = 15
        self.assertEqual(expected_result, result)

    def test_multi_line(self):
        query = """
        Add a column `new_column` to the dataframe which is range 1 - number of rows,
        then return the mean of this column by each type.
        """
        # Here we test that we can accommodate multiple lines. e.g., here we will have:
        # df['new_column'] = range(1, len(df) + 1)
        # df.groupby('type')['new_column'].mean()
        result = self.product_df.llm.query(query, verify=False)
        expected = (
            self.product_df.assign(new_column=range(1, len(self.product_df) + 1))
            .groupby("type")["new_column"]
            .mean()
        )
        pd.testing.assert_series_equal(expected, result)

    def test_multiline_exec(self):
        """Test a multiline command when the final line should be exec'd not eval'd."""
        query = """
        Add a column `new_column` to the dataframe which is range 1 - number of rows,
        then add a column `foo` which is always the value 1
        """
        self.product_df.llm.query(query, verify=False)
        expected_df = self.product_df.assign(
            new_column=range(1, len(self.product_df) + 1)
        ).assign(foo=1)
        pd.testing.assert_frame_equal(expected_df, self.product_df)

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
