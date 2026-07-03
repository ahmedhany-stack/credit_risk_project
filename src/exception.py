import sys
from typing import Any


class CustomException(Exception):
    """
    Custom exception class that provides detailed error information,
    including the file name and line number where the exception occurred.
    """

    def __init__(self, error_message: Any, error_detail=sys):
        super().__init__(str(error_message))
        self.error_message = self._get_detailed_error_message(
            error_message,
            error_detail
        )

    @staticmethod
    def _get_detailed_error_message(error_message: Any, error_detail) -> str:
        _, _, exc_tb = error_detail.exc_info()

        # Fallback if traceback is unavailable
        if exc_tb is None:
            return f"Error: {error_message}"

        file_name = exc_tb.tb_frame.f_code.co_filename
        line_number = exc_tb.tb_lineno

        return (
            f"\n{'=' * 80}\n"
            f"Exception occurred\n"
            f"{'-' * 80}\n"
            f"File       : {file_name}\n"
            f"Line       : {line_number}\n"
            f"Error Type : {type(error_message).__name__}\n"
            f"Message    : {error_message}\n"
            f"{'=' * 80}"
        )

    def __str__(self) -> str:
        return self.error_message