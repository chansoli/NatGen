import os
import unittest
import uuid
from unittest.mock import patch

from tree_sitter import Language

from src.data_preprocessors.transformations.inline_snippet_inserter import InlineSnippetInserter


def _get_parser_so():
    prebuilt = "parser/languages.so"
    if os.path.exists(prebuilt):
        return prebuilt
    sitter_lib_path = "sitter-libs"
    libs = [os.path.join(sitter_lib_path, d) for d in os.listdir(sitter_lib_path)]
    os.makedirs("parser", exist_ok=True)
    Language.build_library(prebuilt, libs)
    return prebuilt

class InlineSnippetInserterTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.parser_so = _get_parser_so()
        cls.skip_reason = None
        try:
            cls.transformer = InlineSnippetInserter(
                parser_path=cls.parser_so,
                language="cpp",
                inline_func=(
                    "int add(int a, int b)\n"
                    "{\n"
                    "    int sum = 0;\n"
                    "    sum += a;\n"
                    "    sum += b;\n"
                    "    return sum;\n"
                    "}"
                ),
            )
        except OSError as exc:
            cls.skip_reason = f"parser load failed: {exc}"
            cls.transformer = None

    def _transform(self, code: str):
        if self.skip_reason:
            self.skipTest(self.skip_reason)
        return self.transformer.transform_code(code)

    def _assert_expected(self, transformed: str):
        expected_lines = [
            "int result;",
            "int a;",
            "int b;",
            "int sum = 0;",
            "sum += a;",
            "sum += b;",
            "result = sum;",
        ]
        for line in expected_lines:
            self.assertIn(line, transformed)
        indices = [transformed.find(l) for l in expected_lines]
        self.assertTrue(all(indices[i] < indices[i + 1] for i in range(len(indices) - 1)))

    def test_plain_function(self):
        code = """
        int foo() {
            int x = 1;
            int y = 2;
            int z = x + y;
            return z;
        }
        """
        with patch.object(uuid, "uuid4", return_value=type("U", (), {"hex": "deadbeef"})()):
            transformed, meta = self._transform(code)
        print("\n[plain_function] transformed:\n", transformed)
        self.assertTrue(meta["success"])
        self._assert_expected(transformed)

    def test_with_loop(self):
        code = """
        int sum_to_n(int n) {
            int acc = 0;
            for (int i = 0; i < n; ++i) {
                acc += i;
            }
            return acc;
        }
        """
        with patch.object(uuid, "uuid4", return_value=type("U", (), {"hex": "deadbeef"})()):
            transformed, meta = self._transform(code)
        print("\n[with_loop] transformed:\n", transformed)
        self.assertTrue(meta["success"])
        self._assert_expected(transformed)

    def test_name_collision_safe(self):
        code = """
        int add(int a, int b) {
            return a + b;
        }

        int caller() {
            int add_result = add(1, 2);
            return add_result;
        }
        """
        with patch.object(uuid, "uuid4", return_value=type("U", (), {"hex": "deadbeef"})()):
            transformed, meta = self._transform(code)
        print("\n[name_collision] transformed:\n", transformed)
        self.assertTrue(meta["success"])
        self._assert_expected(transformed)
        # Ensure original add call remains
        self.assertIn("add(1, 2)", transformed)

class InlineSnippetInserterTestOneArg(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.parser_so = _get_parser_so()
        cls.skip_reason = None
        try:
            cls.transformer = InlineSnippetInserter(
                parser_path=cls.parser_so,
                language="cpp",
                inline_func=(
                    "int power3(int a)\n"
                    "{\n"
                    "    int prod = 1;\n"
                    "    for(int i = 0 ; i< 3; ++i)\n"
                    "    {\n"
                    "        prod *= a;\n"
                    "    }\n"
                    "    return prod;\n"
                    "}"
                ),
            )
        except OSError as exc:
            cls.skip_reason = f"parser load failed: {exc}"
            cls.transformer = None

    def _transform(self, code: str):
        if self.skip_reason:
            self.skipTest(self.skip_reason)
        return self.transformer.transform_code(code)

    def _assert_expected(self, transformed: str):
        expected_lines = [
            "int result;",
            "int a;",
            "int prod = 1;",
            "for(int i = 0 ; i< 3; ++i)",
            "prod *= a;",
            "result = prod;",
        ]
        for line in expected_lines:
            self.assertIn(line, transformed)
        indices = [transformed.find(l) for l in expected_lines]
        self.assertTrue(all(indices[i] < indices[i + 1] for i in range(len(indices) - 1)))

    def test_plain_function(self):
        code = """
        int foo() {
            int x = 1;
            int y = 2;
            int z = x + y;
            return z;
        }
        """
        with patch.object(uuid, "uuid4", return_value=type("U", (), {"hex": "c0ffee"})()):
            transformed, meta = self._transform(code)
        print("\n[one_arg_plain_function] transformed:\n", transformed)
        self.assertTrue(meta["success"])
        self._assert_expected(transformed)

    def test_with_loop(self):
        code = """
        int sum_to_n(int n) {
            int acc = 0;
            for (int i = 0; i < n; ++i) {
                acc += i;
            }
            return acc;
        }
        """
        with patch.object(uuid, "uuid4", return_value=type("U", (), {"hex": "c0ffee"})()):
            transformed, meta = self._transform(code)
        print("\n[one_arg_with_loop] transformed:\n", transformed)
            
        self.assertTrue(meta["success"])
        self._assert_expected(transformed)

    def test_name_collision_safe(self):
        code = """
        int add(int a, int b) {
            return a + b;
        }

        int caller() {
            int add_result = add(1, 2);
            return add_result;
        }
        """
        with patch.object(uuid, "uuid4", return_value=type("U", (), {"hex": "c0ffee"})()):
            transformed, meta = self._transform(code)
        print("\n[one_arg_name_collision] transformed:\n", transformed)
        self.assertTrue(meta["success"])
        self._assert_expected(transformed)
        self.assertIn("add(1, 2)", transformed)

class InlineSnippetConverterTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.parser_so = _get_parser_so()
        cls.skip_reason = None
        try:
            cls.transformer = InlineSnippetInserter(
                parser_path=cls.parser_so,
                language="cpp",
            )
        except OSError as exc:
            cls.skip_reason = f"parser load failed: {exc}"
            cls.transformer = None

    def _convert(self, helper_code: str):
        if self.skip_reason:
            self.skipTest(self.skip_reason)
        return self.transformer._convert_helper_function(helper_code, "", "deadbeef")

    def test_convert_add_helper(self):
        helper = (
            "int add(int a, int b)\n"
            "{\n"
            "\tint sum = 0;\n"
            "\tsum += a;\n"
            "\tsum += b;\n"
            "\treturn sum;\n"
            "}"
        )
        converted = self._convert(helper)
        self.assertIsNotNone(converted)
        expected_order = [
            "int result;",
            "int a;",
            "int b;",
            "int sum = 0;",
            "sum += a;",
            "sum += b;",
            "result = sum;",
        ]
        for line in expected_order:
            self.assertIn(line, converted)
        indices = [converted.find(l) for l in expected_order]
        self.assertTrue(all(indices[i] < indices[i+1] for i in range(len(indices)-1)))

    def test_convert_power3_helper(self):
        helper = (
            "int power3(int a)\n"
            "{\n"
            "\tint prod = 1;\n"
            "\tfor(int i = 0 ; i< 3; ++i)\n"
            "\t{\n"
            "\t\tprod *= a;\n"
            "\t}\n"
            "\n"
            "\treturn prod;\n"
            "}"
        )
        converted = self._convert(helper)
        self.assertIsNotNone(converted)
        expected_substrings = [
            "int result;",
            "int a;",
            "int prod = 1;",
            "for(int i = 0 ; i< 3; ++i)",
            "prod *= a;",
            "result = prod;",
        ]
        for line in expected_substrings:
            self.assertIn(line, converted)
        indices = [converted.find(l) for l in expected_substrings]
        self.assertTrue(all(indices[i] < indices[i+1] for i in range(len(indices)-1)))

if __name__ == "__main__":
    unittest.main()