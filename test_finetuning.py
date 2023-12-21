import unittest
import finetuning


class TestFinetuning(unittest.TestCase):
    t_10 = list(range(1,11))
    t_300 = list(range(1,301))

    def test_truncation_case_a(self):
        comparable = ['[CLS]'] + [v for v in self.t_10] + ['[SEP]'] + [v for v in  self.t_10] + ['[SEP]'] + [v for v in self.t_10] + ['[SEP]'] + 478 * ['[PAD]'] 
        self.assertEqual(finetuning.truncate(self.t_10, self.t_10, self.t_10), comparable)

    def test_truncation_case_b(self):
        comparable = ['[CLS]'] + [v for v in self.t_300[-104:]] + ['[SEP]'] + [v for v in  self.t_300] + ['[SEP]'] + [v for v in self.t_300[:104]] + ['[SEP]']
        self.assertEqual(finetuning.truncate(self.t_300, self.t_300, self.t_300), comparable)

    def test_truncation_case_c(self):
        comparable = ['[CLS]'] + [v for v in self.t_300[-198:]] + ['[SEP]'] + [v for v in  self.t_300] + ['[SEP]'] + [v for v in self.t_10] + ['[SEP]']
        self.assertEqual(finetuning.truncate(self.t_300, self.t_300, self.t_10), comparable)

    def test_truncation_case_d(self):
        comparable = ['[CLS]'] + [v for v in self.t_10] + ['[SEP]'] + [v for v in  self.t_300] + ['[SEP]'] + [v for v in self.t_300[:198]] + ['[SEP]']
        self.assertEqual(finetuning.truncate(self.t_10, self.t_300, self.t_300), comparable)
    
    def test_truncation_two_parameters(self):
        comparable = ['[CLS]'] + [v for v in self.t_10] + ['[SEP]'] + [v for v in  self.t_10] + ['[SEP]'] + 489 * ['[PAD]']
        self.assertEqual(finetuning.truncate(tokens_mid=self.t_10, tokens_pre=self.t_10), comparable)
        self.assertEqual(finetuning.truncate(tokens_mid=self.t_10, tokens_post=self.t_10), comparable)



