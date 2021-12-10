import os
import unittest


class TestRoutes(unittest.TestCase):

    routes = ['models', 'results']

    def test_routes_exist(self):

        for route in self.routes:

            # Assert destination folders exist
            self.assertTrue(os.path.isdir(route))


if __name__ == '__main__':
    unittest.main()
