import os
import unittest


class TestModels(unittest.TestCase):
    routes = ['models']

    def test_routes_exist(self):
        for route in self.routes:
            # Assert destination folders exist
            self.assertTrue(os.path.isdir(route))

    def test_there_is_at_least_one_model(self):
        models = [m for m in os.listdir('models') if m.endswith('.pkl')]
        self.assertTrue(len(models) > 0)
        print(f'Sent {len(models)} models to container')


if __name__ == '__main__':
    unittest.main()
