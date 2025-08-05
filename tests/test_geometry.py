import unittest
from src.geometry_utils import convert_to_linestring

class TestGeometryUtils(unittest.TestCase):
    def test_convert_to_linestring(self):
        sample_input = "{'paths': [[[1,2], [3,4], [5,6]]]}"
        expected_output = "LINESTRING (1 2, 3 4, 5 6)"
        result = str(convert_to_linestring(sample_input))
        self.assertEqual(result, expected_output)

if __name__ == "__main__":
    unittest.main()
