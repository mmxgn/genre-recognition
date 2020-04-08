import unittest
import classify


class Test_Classify(unittest.TestCase):
    def test_ovo(self):
        self.assertEqual(
            classify.classify_audio(
                "../data/genres/blues/blues.00002.wav",
                "../models/features_classifier.pkl",
            ),
            "blues",
        )

    def test_ovr(self):
        self.assertEqual(
            classify.classify_audio(
                "../data/genres/blues/blues.00002.wav",
                "../models/features_classifier_ovr.pkl",
            ),
            "blues",
        )


if __name__ == "__main__":
    unittest.main()
