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
                    "price": 5,
                    "quantity": 80,
                    "rating": 4,
                },
            ],
        )

    def test_basic_use(self):
        result = self.product_df.llm.query(
            "What is the price of the highest-priced book?",
            verify=False,
        )
        expected_result = 15
        self.assertEqual(expected_result, result)

    def test_sliced(self):
        result = self.product_df[["name", "type", "price", "rating"]].llm.query(
            "What is the price of the highest-priced book?", verify=False
        )
        expected_result = 15
        self.assertEqual(expected_result, result)
